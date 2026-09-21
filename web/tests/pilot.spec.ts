import { test, expect, type BrowserContext } from "@playwright/test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
const output = process.env.BROWSER_OUTPUT_DIR || "../.cache/production/browser";

const fixture = JSON.parse(readFileSync(process.env.BROWSER_FIXTURE!, "utf8"));
async function authenticate(context: BrowserContext, number: number) {
  const account = fixture.accounts[number];
  await context.addCookies([
    {
      name: "__Host-recserve",
      value: account.token,
      url: fixture.origin,
      secure: true,
      httpOnly: true,
      sameSite: "Lax",
    },
    {
      name: "__Host-recserve-csrf",
      value: account.csrf,
      url: fixture.origin,
      secure: true,
      httpOnly: false,
      sameSite: "Lax",
    },
  ]);
}

test("invitation-only landing is keyboard accessible", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /A good film/ }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /Continue with Google/ }),
  ).toHaveAttribute("href", "/auth/login");
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("link", { name: "Skip to content" }),
  ).toBeFocused();
  await page.screenshot({
    path: join(output, "landing-desktop.png"),
    fullPage: true,
  });
});

test("real API onboarding, recommendation, save, retry, export and deletion", async ({
  page,
  context,
}) => {
  await authenticate(context, 0);
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Join the research pilot." }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Agree and choose films" }),
  ).toBeDisabled();
  await page.getByRole("checkbox").nth(0).check();
  await page.getByRole("checkbox").nth(1).check();
  await page.getByRole("button", { name: "Agree and choose films" }).click();
  await expect(
    page.getByRole("heading", { name: "Start with what you love." }),
  ).toBeVisible();
  for (let i = 0; i < 5; i++) {
    await page
      .getByRole("button", { name: "Like", exact: true })
      .nth(i)
      .click();
    await expect(
      page.getByRole("button", { name: "Like", exact: true }).nth(i),
    ).toHaveAttribute("aria-pressed", "true");
  }
  await page
    .getByRole("button", { name: "Discover films", exact: true })
    .click();
  await expect(page.locator(".movie")).toHaveCount(10);
  const title = await page.locator(".movie h3").first().textContent();
  await page.getByRole("button", { name: "Save", exact: true }).first().click();
  await expect(
    page.getByRole("status").filter({ hasText: "Saved to your watchlist." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Watchlist", exact: true }).click();
  await expect(page.getByRole("heading", { name: title! })).toBeVisible();
  await page.getByRole("button", { name: "Discover", exact: true }).click();
  await expect(page.getByRole("heading", { name: title! })).toHaveCount(0);
  let failed = false;
  const ids: string[] = [];
  await page.route("**/api/v2/recommendations", async (route) => {
    ids.push(route.request().postDataJSON().request_id);
    if (!failed) {
      failed = true;
      await route.fulfill({
        status: 503,
        json: { detail: "Fixture temporary failure" },
      });
    } else await route.continue();
  });
  await page.getByRole("button", { name: "Refresh recommendations" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Fixture temporary failure",
  );
  await page.getByRole("button", { name: "Refresh recommendations" }).click();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(ids[0]).toBe(ids[1]);
  await page.screenshot({
    path: join(output, "discovery-desktop.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Account", exact: true }).click();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export my data" }).click();
  expect((await download).suggestedFilename()).toBe("recserve-account.json");
  await page.getByRole("button", { name: "Delete my account" }).click();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.getByRole("button", { name: "Delete my account" }).click();
  await page
    .getByRole("button", { name: "Confirm permanent deletion" })
    .click();
  await expect(
    page.getByRole("heading", { name: /A good film/ }),
  ).toBeVisible();
  expect((await page.request.get("/api/v2/me")).status()).toBe(401);
});

test("mobile search empty state and logout", async ({ page, context }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await authenticate(context, 1);
  await page.goto("/");
  await page.getByRole("checkbox").nth(0).check();
  await page.getByRole("checkbox").nth(1).check();
  await page.getByRole("button", { name: "Agree and choose films" }).click();
  await page.getByLabel("Find a film by title").fill("no such fixture movie");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(
    page.getByText(
      "No titles found in this historical catalog. Try another title.",
    ),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: join(output, "search-mobile.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Account", exact: true }).click();
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(
    page.getByRole("link", { name: /Continue with Google/ }),
  ).toBeVisible();
});
