import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  api,
  ApiError,
  type Me,
  type Movie,
  type Picks,
  type MoviePage,
  type PreferencePage,
  type Watchlist,
  type EventBody,
} from "./client";
import "./style.css";

type Page = "discover" | "onboarding" | "search" | "watchlist" | "account";
type Value = -1 | 0 | 1;

function Card({
  movie,
  children,
  shown,
}: {
  movie: Movie;
  children?: React.ReactNode;
  shown?: () => Promise<void>;
}) {
  const element = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!shown || !element.current) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        clearTimeout(timer);
        if (entries[0].intersectionRatio >= 0.5)
          timer = setTimeout(() => {
            void shown().catch(() => {});
          }, 800);
      },
      { threshold: 0.5 },
    );
    observer.observe(element.current);
    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [shown]);
  return (
    <article className="movie" ref={element}>
      <div className="movie-meta">
        <span>{movie.year ?? "Year unknown"}</span>
        <span>
          {movie.genres
            .filter((g) => g !== "(no genres listed)")
            .slice(0, 2)
            .join(" / ")}
        </span>
      </div>
      <h3>{movie.title.replace(/\s*\(\d{4}\)$/, "")}</h3>
      {children && <div className="movie-actions">{children}</div>}
    </article>
  );
}

function App() {
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [page, setPage] = useState<Page>("discover");
  const [adult, setAdult] = useState(false);
  const [research, setResearch] = useState(false);
  const [query, setQuery] = useState("");
  const [movies, setMovies] = useState<MoviePage | null>(null);
  const [prefs, setPrefs] = useState<Record<number, Value>>({});
  const [picks, setPicks] = useState<Picks | null>(null);
  const [watchlist, setWatchlist] = useState<Watchlist | null>(null);
  const [deleting, setDeleting] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const pendingRecommendation = useRef<{
    request_id: string;
    k: number;
  } | null>(null);
  const intents = useRef(new Map<string, unknown>());
  const shown = useRef(new Set<string>());
  const showing = useRef(new Map<string, Promise<void>>());
  const liked = Object.values(prefs).filter((value) => value === 1).length;

  function fail(reason: unknown) {
    if (reason instanceof ApiError && reason.status === 401) {
      setMe(null);
      setPicks(null);
      setPrefs({});
      setWatchlist(null);
      setMovies(null);
      setError("Your session has ended. Please sign in again.");
    } else
      setError(
        reason instanceof Error
          ? reason.message
          : "Something went wrong. Please try again.",
      );
  }

  async function perform(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (reason) {
      fail(reason);
    } finally {
      setBusy(false);
    }
  }

  async function loadMe() {
    setLoading(true);
    setError("");
    try {
      setMe(await api<Me>("/api/v2/me"));
    } catch (reason) {
      if (!(reason instanceof ApiError && reason.status === 401)) fail(reason);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadMe();
  }, []);
  useEffect(() => {
    heading.current?.focus();
  }, [page]);

  async function loadPreferences() {
    const response = await api<PreferencePage>("/api/v2/preferences");
    setPrefs(
      Object.fromEntries(
        response.items.map((item) => [item.item_id, item.value]),
      ),
    );
  }

  async function search(cursor = 0) {
    const response = await api<MoviePage>(
      `/api/v2/movies?query=${encodeURIComponent(query)}&cursor=${cursor}`,
    );
    setMovies((previous) =>
      cursor && previous
        ? { ...response, items: [...previous.items, ...response.items] }
        : response,
    );
  }

  async function discover(fresh = false) {
    if (fresh || !pendingRecommendation.current)
      pendingRecommendation.current = {
        request_id: crypto.randomUUID(),
        k: 10,
      };
    setPicks(
      await api<Picks>(
        "/api/v2/recommendations",
        "POST",
        pendingRecommendation.current,
      ),
    );
    pendingRecommendation.current = null;
  }

  async function navigate(next: Page) {
    setPage(next);
    setError("");
    setMovies(null);
    await perform(async () => {
      if (next === "discover") await discover();
      if (next === "search" || next === "onboarding") {
        await loadPreferences();
        await search();
      }
      if (next === "watchlist")
        setWatchlist(await api<Watchlist>("/api/v2/watchlist"));
    });
  }

  async function preference(movie: Movie, value: Value) {
    const key = `preference:${movie.id}:${value}`;
    if (!intents.current.has(key))
      intents.current.set(key, {
        request_id: crypto.randomUUID(),
        changes: [{ item_id: movie.id, value }],
      });
    await api("/api/v2/preferences", "PUT", intents.current.get(key));
    intents.current.delete(key);
    setPrefs((previous) => {
      const next = { ...previous };
      if (value === 0) delete next[movie.id];
      else next[movie.id] = value;
      return next;
    });
    setNotice(
      "Preference saved. New recommendations will use your updated choices.",
    );
  }

  async function sendEvent(movie: Movie, kind: EventBody["kind"]) {
    if (!picks) return;
    const key = `${picks.request_id}:${movie.id}:${kind}`;
    if (!intents.current.has(key))
      intents.current.set(key, {
        event_id: crypto.randomUUID(),
        request_id: picks.request_id,
        item_id: movie.id,
        kind,
        event_time_ms: Date.now(),
      } satisfies EventBody);
    await api("/api/v2/events", "POST", intents.current.get(key));
    intents.current.delete(key);
  }

  async function ensureShown(movie: Movie) {
    if (!picks) return;
    const key = `${picks.request_id}:${movie.id}`;
    if (shown.current.has(key)) return;
    const current = showing.current.get(key);
    if (current) return current;
    const promise = sendEvent(movie, "shown")
      .then(() => {
        shown.current.add(key);
      })
      .finally(() => {
        showing.current.delete(key);
      });
    showing.current.set(key, promise);
    return promise;
  }

  async function outcome(movie: Movie, kind: "save" | "dismiss" | "watched") {
    await ensureShown(movie);
    await sendEvent(movie, kind);
    setPicks(
      (previous) =>
        previous && {
          ...previous,
          items: previous.items.filter((item) => item.id !== movie.id),
        },
    );
    setNotice(
      kind === "save"
        ? "Saved to your watchlist."
        : kind === "watched"
          ? "Added to your watched history."
          : "Dismissed. It will not appear in fresh recommendations.",
    );
  }

  async function exportData() {
    const result = await api<unknown>("/api/v2/me/export");
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "recserve-account.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Your account export has been downloaded.");
  }

  const consented = me?.consent_version === me?.required_consent && !!me;
  const titles: Record<Page, string> = {
    discover: "Your next good film.",
    onboarding: "Start with what you love.",
    search: "A little more your taste.",
    watchlist: "For another evening.",
    account: "Your account. Your choices.",
  };

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="header">
        <a className="wordmark" href="/" aria-label="RecServe home">
          rec<span>serve</span>
          <i> / films</i>
        </a>
        {consented && (
          <nav aria-label="Main navigation">
            {(["discover", "search", "watchlist", "account"] as Page[]).map(
              (next) => (
                <button
                  key={next}
                  aria-current={page === next ? "page" : undefined}
                  disabled={busy}
                  onClick={() => void navigate(next)}
                >
                  {
                    {
                      discover: "Discover",
                      search: "Your taste",
                      watchlist: "Watchlist",
                      account: "Account",
                      onboarding: "Onboarding",
                    }[next]
                  }
                </button>
              ),
            )}
          </nav>
        )}
        <span className="pilot-label">RESEARCH PILOT</span>
      </header>
      <main id="main">
        {error && (
          <div className="message error" role="alert">
            {error}{" "}
            {!me && (
              <button onClick={() => void loadMe()}>Retry connection</button>
            )}
          </div>
        )}
        {notice && (
          <div className="message" role="status">
            {notice}
          </div>
        )}
        {loading ? (
          <div className="loading" role="status">
            Opening your film shelf…
          </div>
        ) : !me ? (
          <section className="welcome">
            <p className="eyebrow">A SMALL CATALOG. A NEW PERSPECTIVE.</p>
            <h1>
              A good film
              <br />
              is worth finding.
            </h1>
            <p className="intro">
              Rediscover films you missed. Tell us what you like, keep a
              watchlist, and help us learn what makes a recommendation useful.
            </p>
            <a className="button primary" href="/auth/login">
              Continue with Google <span aria-hidden="true">↗</span>
            </a>
            <p className="fine">
              Invitation required · Adults only · No ads or payments
            </p>
            <div className="editorial-note">
              <span>01 / THE COLLECTION</span>
              <p>
                A historical MovieLens catalog, not a guide to current releases
                or where to stream them. This is a noncommercial research pilot.
              </p>
            </div>
          </section>
        ) : !consented ? (
          <section className="narrow">
            <p className="eyebrow">BEFORE WE BEGIN</p>
            <h1>Join the research pilot.</h1>
            <p>
              We store your Google identity and email, your movie choices, and
              interactions with recommendations. We use these to operate and
              evaluate this service. Saving a film does not automatically mean
              you like it.
            </p>
            <p>
              You can export or delete your account. Interaction records are
              retained for 90 days; backups for seven days. Contact the person
              who invited you with questions. No recommendation-quality
              improvement is promised.
            </p>
            <label className="check">
              <input
                type="checkbox"
                checked={adult}
                onChange={(e) => setAdult(e.target.checked)}
              />{" "}
              I am at least 18 years old.
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={research}
                onChange={(e) => setResearch(e.target.checked)}
              />{" "}
              I understand and consent to this noncommercial research pilot.
            </label>
            <button
              className="primary"
              disabled={!adult || !research || busy}
              onClick={() =>
                void perform(async () => {
                  await api("/api/v2/me/consent", "PUT", {
                    adult: true,
                    research: true,
                  });
                  setMe({ ...me, consent_version: me.required_consent });
                  setPage("onboarding");
                  await loadPreferences();
                  await search();
                })
              }
            >
              Agree and choose films
            </button>
            <button
              className="text-button"
              disabled={busy}
              onClick={() =>
                void perform(async () => {
                  await api("/auth/logout", "POST");
                  setMe(null);
                })
              }
            >
              Sign out
            </button>
          </section>
        ) : (
          <>
            <section className="section-heading">
              <div>
                <p className="eyebrow">
                  {page === "discover"
                    ? "CURATED BY YOUR CHOICES"
                    : "YOUR FILM SHELF"}
                </p>
                <h1 ref={heading} tabIndex={-1}>
                  {titles[page]}
                </h1>
              </div>
              {page === "discover" && (
                <button
                  className="primary"
                  disabled={busy}
                  onClick={() => void perform(() => discover())}
                >
                  {picks ? "Refresh recommendations" : "Find films"}
                </button>
              )}
            </section>
            {busy && (
              <p className="loading" role="status">
                Working on your request…
              </p>
            )}
            {(page === "onboarding" || page === "search") && (
              <>
                <p className="intro small">
                  {page === "onboarding"
                    ? `Choose five films you like to introduce your taste. ${liked} selected. You can also skip this step.`
                    : "Likes and dislikes are explicit preferences. Clear a choice whenever your taste changes."}
                </p>
                <form
                  className="search"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void perform(() => search());
                  }}
                >
                  <label htmlFor="movie-query">Find a film by title</label>
                  <div>
                    <input
                      id="movie-query"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      maxLength={100}
                      placeholder="Try a film you remember…"
                    />
                    <button className="primary" disabled={busy}>
                      Search
                    </button>
                  </div>
                </form>
                {page === "onboarding" && (
                  <div className="onboard-progress">
                    <span>{Math.min(liked, 5)} / 5 likes</span>
                    <button
                      disabled={busy}
                      onClick={() => void navigate("discover")}
                    >
                      {liked >= 5 ? "Discover films" : "Skip for now"}
                    </button>
                  </div>
                )}
                {movies?.items.length === 0 && (
                  <p className="empty">
                    No titles found in this historical catalog. Try another
                    title.
                  </p>
                )}
                {!movies && !busy && (
                  <button onClick={() => void perform(() => search())}>
                    Load titles
                  </button>
                )}
                <div className="movie-grid">
                  {movies?.items.map((movie) => (
                    <Card key={movie.id} movie={movie}>
                      <button
                        aria-pressed={prefs[movie.id] === 1}
                        disabled={busy}
                        onClick={() => void perform(() => preference(movie, 1))}
                      >
                        Like
                      </button>
                      <button
                        aria-pressed={prefs[movie.id] === -1}
                        disabled={busy}
                        onClick={() =>
                          void perform(() => preference(movie, -1))
                        }
                      >
                        Dislike
                      </button>
                      {prefs[movie.id] && (
                        <button
                          disabled={busy}
                          onClick={() =>
                            void perform(() => preference(movie, 0))
                          }
                        >
                          Clear
                        </button>
                      )}
                    </Card>
                  ))}
                </div>
                {movies?.next_cursor != null && (
                  <button
                    disabled={busy}
                    onClick={() =>
                      void perform(() => search(movies.next_cursor!))
                    }
                  >
                    More titles
                  </button>
                )}
              </>
            )}
            {page === "discover" && (
              <>
                <p className="intro small">
                  A starting point for your next evening in. Your liked,
                  disliked, saved, watched, and dismissed films stay out of
                  fresh recommendations.
                </p>
                {picks && (
                  <p className="policy-note">
                    {picks.candidate_source === "popularity"
                      ? "Training-popularity picks, filtered by your choices. Personalized ranking is still under evaluation."
                      : "Ranked using your explicit preferences."}{" "}
                    {picks.degraded &&
                      "Personalized retrieval is temporarily unavailable; these are popularity fallback picks."}
                  </p>
                )}
                {!picks && !busy && (
                  <div className="empty">
                    Your next film starts here. Select “Find films” to open your
                    first set.
                  </div>
                )}
                {picks?.items.length === 0 && (
                  <div className="empty">
                    No more eligible films in this set. Refresh for a new set,
                    or edit your preferences.
                  </div>
                )}
                <div className="movie-grid">
                  {picks?.items.map((movie) => (
                    <Card
                      key={`${picks.request_id}:${movie.id}`}
                      movie={movie}
                      shown={() => ensureShown(movie)}
                    >
                      <button
                        disabled={busy}
                        onClick={() =>
                          void perform(() => outcome(movie, "save"))
                        }
                      >
                        Save
                      </button>
                      <button
                        disabled={busy}
                        onClick={() =>
                          void perform(() => outcome(movie, "watched"))
                        }
                      >
                        Watched
                      </button>
                      <button
                        disabled={busy}
                        onClick={() =>
                          void perform(() => outcome(movie, "dismiss"))
                        }
                      >
                        Dismiss
                      </button>
                    </Card>
                  ))}
                </div>
                {picks?.exhausted && (
                  <p>We found fewer eligible titles than requested.</p>
                )}
                <button
                  className="text-button"
                  disabled={busy}
                  onClick={() => void navigate("onboarding")}
                >
                  Update your starting preferences →
                </button>
              </>
            )}
            {page === "watchlist" && (
              <>
                <p className="intro small">
                  Saved films and your viewing history. A save is an
                  intention—not a confirmed watch.
                </p>
                {!watchlist && !busy && (
                  <button
                    onClick={() =>
                      void perform(async () =>
                        setWatchlist(await api<Watchlist>("/api/v2/watchlist")),
                      )
                    }
                  >
                    Load watchlist
                  </button>
                )}
                {watchlist?.items.length === 0 && (
                  <p className="empty">
                    Your shelf is empty. Save a film from Discover to keep it
                    here.
                  </p>
                )}
                <div className="movie-grid">
                  {watchlist?.items
                    .filter((movie) => movie.saved || movie.watched)
                    .map((movie) => (
                      <Card key={movie.id} movie={movie}>
                        <span className="status-tag">
                          {movie.watched ? "Watched" : "Saved for later"}
                        </span>
                      </Card>
                    ))}
                </div>
              </>
            )}
            {page === "account" && (
              <section className="account-panel">
                <h2>Your data stays yours.</h2>
                <p>
                  This adults-only pilot stores your identity, preferences, and
                  interactions. Events are retained for 90 days and backups for
                  seven days. Deletion revokes your sessions immediately and
                  removes primary account records; expired backups are removed
                  under the retention policy.
                </p>
                <p>
                  Research consent: {me.consent_version}. For questions, contact
                  the person who invited you.
                </p>
                <div className="actions">
                  <button
                    disabled={busy}
                    onClick={() => void perform(exportData)}
                  >
                    Export my data
                  </button>
                  <button
                    disabled={busy}
                    onClick={() =>
                      void perform(async () => {
                        await api("/auth/logout", "POST");
                        setMe(null);
                        setPicks(null);
                      })
                    }
                  >
                    Sign out
                  </button>
                </div>
                <hr />
                <h2>Leave the pilot</h2>
                <p>
                  Account deletion cannot be undone. You will need a new
                  invitation to return.
                </p>
                {!deleting ? (
                  <button className="danger" onClick={() => setDeleting(true)}>
                    Delete my account
                  </button>
                ) : (
                  <div className="confirmation" role="alert">
                    <p>
                      Delete your account and all primary preferences and
                      history?
                    </p>
                    <button
                      className="danger"
                      disabled={busy}
                      onClick={() =>
                        void perform(async () => {
                          await api("/api/v2/me", "DELETE");
                          setMe(null);
                          setPicks(null);
                          setPrefs({});
                          setWatchlist(null);
                          setDeleting(false);
                          setNotice("Your account was deleted.");
                        })
                      }
                    >
                      Confirm permanent deletion
                    </button>
                    <button disabled={busy} onClick={() => setDeleting(false)}>
                      Cancel
                    </button>
                  </div>
                )}
              </section>
            )}
          </>
        )}
      </main>
      <footer>
        <span>recserve / a recommendation systems research pilot</span>
        <span>
          Historical data: GroupLens MovieLens · No endorsement implied
        </span>
      </footer>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
