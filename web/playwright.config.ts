import { defineConfig } from "@playwright/test";
import { join } from "node:path";
const output = process.env.BROWSER_OUTPUT_DIR || "../.cache/production/browser";

export default defineConfig({
  testDir: "./tests",
  workers: 1,
  retries: 0,
  timeout: 30000,
  outputDir: join(output, "artifacts"),
  reporter: [
    ["line"],
    ["json", { outputFile: join(output, "report.json") }],
  ],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL,
    browserName: "chromium",
    ignoreHTTPSErrors: true,
    viewport: { width: 1360, height: 900 },
    screenshot: "only-on-failure",
  },
});
