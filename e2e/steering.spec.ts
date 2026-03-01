import { test, expect } from "@playwright/test";

test.describe("Feature Steering App", () => {
  test("page loads with header and UI elements", async ({ page }) => {
    await page.goto("http://localhost:3000");

    // Header
    await expect(page.locator("h1")).toHaveText("Feature Steering");
    await expect(page.getByText("Steer Mistral 7B code generation")).toBeVisible();

    // Prompt input with default value
    const textarea = page.locator("textarea");
    await expect(textarea).toBeVisible();
    await expect(textarea).toHaveValue("def fibonacci(n):");

    // Generate button
    await expect(page.getByRole("button", { name: /Generate/ })).toBeVisible();
  });

  test("features load from backend", async ({ page }) => {
    await page.goto("http://localhost:3000");

    // Wait for features to load (skeleton disappears, labels appear)
    await expect(page.getByText("recursion / recursive calls")).toBeVisible({ timeout: 10000 });
    await expect(page.getByText("list comprehension / generator expressions")).toBeVisible();
    await expect(page.getByText("error handling / try-except blocks")).toBeVisible();
    await expect(page.getByText("type annotations / type hints")).toBeVisible();
    await expect(page.getByText("object-oriented / class definitions")).toBeVisible();

    // All 5 sliders present
    const sliders = page.locator('input[type="range"]');
    await expect(sliders).toHaveCount(5);
  });

  test("sliders default to 0 and can be adjusted", async ({ page }) => {
    await page.goto("http://localhost:3000");
    await expect(page.getByText("recursion / recursive calls")).toBeVisible({ timeout: 10000 });

    // All sliders start at 0
    const sliders = page.locator('input[type="range"]');
    for (let i = 0; i < 5; i++) {
      await expect(sliders.nth(i)).toHaveValue("0");
    }

    // Adjust first slider
    await sliders.first().fill("5");
    await expect(sliders.first()).toHaveValue("5");
    await expect(page.getByText("+5.0")).toBeVisible();
  });

  test("generate produces baseline and steered output", async ({ page }) => {
    await page.goto("http://localhost:3000");
    await expect(page.getByText("recursion / recursive calls")).toBeVisible({ timeout: 10000 });

    // Empty state shown before generating
    await expect(page.getByText("Enter a prompt and click Generate")).toBeVisible();

    // Click generate
    await page.getByRole("button", { name: /Generate/ }).click();

    // Wait for results (model inference takes time)
    await expect(page.getByText("Baseline")).toBeVisible({ timeout: 120000 });
    await expect(page.getByText("Steered")).toBeVisible();

    // Code blocks should contain the prompt text
    const codeBlocks = page.locator("pre code");
    await expect(codeBlocks).toHaveCount(2);
    await expect(codeBlocks.first()).toContainText("def fibonacci");
    await expect(codeBlocks.last()).toContainText("def fibonacci");
  });

  test("steered output differs when feature strength is non-zero", async ({ page }) => {
    await page.goto("http://localhost:3000");
    await expect(page.getByText("recursion / recursive calls")).toBeVisible({ timeout: 10000 });

    // Set type hints feature to +8
    const sliders = page.locator('input[type="range"]');
    await sliders.nth(3).fill("8"); // type annotations slider

    // Change prompt to something where steering has visible effect
    await page.locator("textarea").fill("def add(a, b):");

    // Generate
    await page.getByRole("button", { name: /Generate/ }).click();

    // Wait for results
    await expect(page.getByText("Baseline")).toBeVisible({ timeout: 120000 });

    const codeBlocks = page.locator("pre code");
    await expect(codeBlocks).toHaveCount(2);

    const baseline = await codeBlocks.first().textContent();
    const steered = await codeBlocks.last().textContent();

    // Both should contain the prompt
    expect(baseline).toContain("def add");
    expect(steered).toContain("def add");

    // Steered should differ from baseline
    expect(steered).not.toEqual(baseline);
  });

  test("error banner shows when backend is unreachable", async ({ page }) => {
    // Block backend requests by navigating with a broken API
    await page.route("**/features", (route) => route.abort());
    await page.goto("http://localhost:3000");

    // Error banner should appear
    await expect(page.getByText(/Cannot connect to backend/)).toBeVisible({ timeout: 10000 });

    // Dismiss it
    await page.getByRole("button", { name: /Dismiss/ }).click();
    await expect(page.getByText(/Cannot connect to backend/)).not.toBeVisible();
  });
});
