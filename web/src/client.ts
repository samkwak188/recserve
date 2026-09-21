import type { components } from "./generated/api";

export type Me = components["schemas"]["Me"];
export type Movie = components["schemas"]["Movie"];
export type Picks = components["schemas"]["RecommendationResult"];
export type PreferencePage = components["schemas"]["PreferencePage"];
export type Watchlist = components["schemas"]["Watchlist"];
export type MoviePage = components["schemas"]["MoviePage"];
export type EventBody = components["schemas"]["Event"];

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const csrf =
    document.cookie
      .split("; ")
      .find((value) => value.startsWith("__Host-recserve-csrf="))
      ?.split("=")[1] ?? "";
  const response = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": decodeURIComponent(csrf),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = "The request could not be completed. Please try again.";
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* A proxy error need not be JSON. */
    }
    throw new ApiError(response.status, detail);
  }
  return response.status === 204
    ? (undefined as T)
    : ((await response.json()) as T);
}
