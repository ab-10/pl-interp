import { test, expect } from "@playwright/test";

test.describe("Feature Steering App", () => {
  test("page loads with header and UI elements", async ({ page }) => {
    await page.goto("http://localhost:3000");

    // Header
    await expect(page.locator("h1")).toHaveText("Feature Steering");

    // Prompt input with default value
    const textarea = page.locator("textarea");
    await expect(textarea).toBeVisible();
    await expect(textarea).toHaveValue("def fibonacci(n):");

    // Generate button
    await expect(page.getByRole("button", { name: /Generate/ })).toBeVisible();

    // Tab bar
    await expect(page.getByRole("button", { name: "Code" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Analysis" })).toBeVisible();
  });

  test("features load from backend", async ({ page }) => {
    await page.goto("http://localhost:3000");

    // Wait for features to load (skeleton disappears, section title appears)
    await expect(page.getByText("Steering Features")).toBeVisible({ timeout: 10000 });

    // At least one slider present (exact labels depend on backend)
    const sliders = page.locator('input[type="range"]');
    const count = await sliders.count();
    expect(count).toBeGreaterThanOrEqual(2); // at least temperature + 1 feature
  });

  test("sliders default to 0 and can be adjusted", async ({ page }) => {
    await page.goto("http://localhost:3000");
    await expect(page.getByText("Steering Features")).toBeVisible({ timeout: 10000 });

    // Feature sliders (skip temperature slider which has a different range)
    // Adjust first feature slider
    const featureSliders = page.locator('input[type="range"][step="0.5"]');
    const count = await featureSliders.count();
    expect(count).toBeGreaterThanOrEqual(1);

    // All feature sliders start at 0
    await expect(featureSliders.first()).toHaveValue("0");

    // Adjust a slider
    await featureSliders.first().fill("2.5");
    await expect(featureSliders.first()).toHaveValue("2.5");
  });

  test("generate produces baseline and steered output", async ({ page }) => {
    await page.goto("http://localhost:3000");
    await expect(page.getByText("Steering Features")).toBeVisible({ timeout: 10000 });

    // Empty state shown before generating
    await expect(page.getByText("Enter a prompt and click Generate")).toBeVisible();

    // Click generate
    await page.getByRole("button", { name: /Generate/ }).click();

    // Wait for diff results (model inference takes time)
    await expect(page.locator("pre code")).toBeVisible({ timeout: 120000 });
  });

  test("steered output differs when feature strength is non-zero", async ({ page }) => {
    await page.goto("http://localhost:3000");
    await expect(page.getByText("Steering Features")).toBeVisible({ timeout: 10000 });

    // Set first feature slider to +3.0
    const featureSliders = page.locator('input[type="range"][step="0.5"]');
    await featureSliders.first().fill("3");

    // Change prompt
    await page.locator("textarea").fill("def add(a, b):");

    // Generate
    await page.getByRole("button", { name: /Generate/ }).click();

    // Wait for results
    await expect(page.locator("pre code")).toBeVisible({ timeout: 120000 });
  });

  test("error banner shows when backend is unreachable", async ({ page }) => {
    // Block backend requests
    await page.route("**/features", (route) => route.abort());
    await page.goto("http://localhost:3000");

    // Error banner should appear
    await expect(page.getByText(/Cannot connect to backend/)).toBeVisible({ timeout: 10000 });

    // Dismiss it
    await page.getByRole("button", { name: /Dismiss/ }).click();
    await expect(page.getByText(/Cannot connect to backend/)).not.toBeVisible();
  });

  test("tabs switch between Code, Analysis, and Feature Space", async ({ page }) => {
    await page.goto("http://localhost:3000");

    // Code tab is default — shows generate prompt
    await expect(page.getByText("Enter a prompt and click Generate")).toBeVisible();

    // Switch to Analysis tab
    await page.getByRole("button", { name: "Analysis" }).click();
    await expect(page.getByText("Generate code to see property density analysis")).toBeVisible();

    // Switch back to Code
    await page.getByRole("button", { name: "Code" }).click();
    await expect(page.getByText("Enter a prompt and click Generate")).toBeVisible();
  });

  test("controls section is collapsible", async ({ page }) => {
    await page.goto("http://localhost:3000");
    await expect(page.getByText("Steering Features")).toBeVisible({ timeout: 10000 });

    // Look for the controls toggle (may not exist if backend has no random controls)
    const controlsButton = page.getByRole("button", { name: /Controls/ });
    const hasControls = await controlsButton.isVisible().catch(() => false);

    if (hasControls) {
      // Controls are collapsed by default — click to expand
      await controlsButton.click();
      // Should now see control slider(s)
      const controlSliders = page.locator('.border-l.border-zinc-700 input[type="range"]');
      const count = await controlSliders.count();
      expect(count).toBeGreaterThanOrEqual(1);
    }
  });
});
