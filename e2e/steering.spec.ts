import { test, expect } from "@playwright/test";

/** Mock /features response — flat dict of feature labels. */
const MOCK_FEATURES = {
  "13176": "Typing",
  "16290": "Recursion",
};

/** Mock /info response. */
const MOCK_INFO = {
  model: "mistralai/Ministral-8B-Instruct-2410",
  sae: "layer_18_sae_checkpoint.pt",
  layer: 18,
};

/** Mock /generate response. */
const MOCK_GENERATE = {
  baseline: "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  steered: "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
};

/** Install mock API routes for all backend endpoints. */
async function mockBackend(page: import("@playwright/test").Page) {
  await page.route("**/api/backend/features", (route) =>
    route.fulfill({ json: MOCK_FEATURES })
  );
  await page.route("**/api/backend/info", (route) =>
    route.fulfill({ json: MOCK_INFO })
  );
  await page.route("**/api/backend/generate", (route) =>
    route.fulfill({ json: MOCK_GENERATE })
  );
}

test.describe("Feature Steering App", () => {
  test("page loads with header and UI elements", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");

    // Header
    await expect(page.locator("h1")).toHaveText("Feature Steering");
    await expect(
      page.getByText("Steer Ministral 8B code generation")
    ).toBeVisible();

    // Prompt input with default value
    const textarea = page.locator("textarea");
    await expect(textarea).toBeVisible();
    await expect(textarea).toHaveValue("Write a Python function that merges two sorted lists.");

    // Generate button
    await expect(
      page.getByRole("button", { name: /Generate/ })
    ).toBeVisible();
  });

  test("model info displays correctly", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");

    await expect(
      page.getByText("Model: mistralai/Ministral-8B-Instruct-2410")
    ).toBeVisible({ timeout: 5000 });
    await expect(page.getByText("SAE: layer_18_sae_checkpoint.pt (layer 18)")).toBeVisible();
  });

  test("features load as toggle buttons", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");

    // Feature toggle buttons
    await expect(
      page.getByRole("button", { name: "Typing" })
    ).toBeVisible({ timeout: 5000 });
    await expect(page.getByRole("button", { name: "Recursion" })).toBeVisible();

    // No feature slider visible initially (only temperature slider)
    const sliders = page.locator('input[type="range"]');
    await expect(sliders).toHaveCount(1);
  });

  test("toggling a feature shows slider at default strength 3", async ({
    page,
  }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByRole("button", { name: "Typing" })).toBeVisible({ timeout: 5000 });

    // Click Typing toggle
    await page.getByRole("button", { name: "Typing" }).click();

    // Slider appears with default value of 3
    const featureSlider = page.locator('input[type="range"][min="-10"][max="10"]');
    await expect(featureSlider).toHaveCount(1);
    await expect(featureSlider).toHaveValue("3");
    await expect(page.getByText("+3.0")).toBeVisible();

    // Adjust slider
    await featureSlider.fill("5");
    await expect(featureSlider).toHaveValue("5");
    await expect(page.getByText("+5.0")).toBeVisible();
  });

  test("features are mutually exclusive", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByRole("button", { name: "Typing" })).toBeVisible({ timeout: 5000 });

    // Activate Typing
    await page.getByRole("button", { name: "Typing" }).click();
    const featureSliders = page.locator('input[type="range"][min="-10"][max="10"]');
    await expect(featureSliders).toHaveCount(1);

    // Switch to Recursion — replaces the slider
    await page.getByRole("button", { name: "Recursion" }).click();
    await expect(featureSliders).toHaveCount(1);

    // Only one slider present
    await expect(featureSliders).toHaveCount(1);
  });

  test("generate produces diff output", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByRole("button", { name: "Typing" })).toBeVisible({ timeout: 5000 });

    // Empty state shown before generating
    await expect(
      page.getByText("Enter a prompt and click Generate")
    ).toBeVisible();

    // Click generate
    await page.getByRole("button", { name: /Generate/ }).click();

    // Diff view should appear
    await expect(page.getByText("Diff")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("pre")).toBeVisible();
  });

  test("custom feature input appears inline", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByRole("button", { name: "Typing" })).toBeVisible({ timeout: 5000 });

    // Feature ID input alongside existing features
    const featureInput = page.locator('input[placeholder="Feature ID"]');
    await expect(featureInput).toBeVisible();

    // Add button present
    await expect(page.getByRole("button", { name: "Add" })).toBeVisible();
  });

  test("can add and remove custom features as toggle buttons", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByRole("button", { name: "Typing" })).toBeVisible({ timeout: 5000 });

    // Add a custom feature
    await page.locator('input[placeholder="Feature ID"]').fill("42");
    await page.getByRole("button", { name: "Add" }).click();

    // Custom feature appears as a toggle button and is auto-activated with slider
    await expect(page.getByRole("button", { name: /Feature #42/ })).toBeVisible();
    const featureSlider = page.locator('input[type="range"][min="-10"][max="10"]');
    await expect(featureSlider).toHaveCount(1);
    await expect(featureSlider).toHaveValue("3");

    // Remove it
    await page.getByTitle("Remove").click();
    await expect(page.getByRole("button", { name: /Feature #42/ })).not.toBeVisible();
  });

  test("error banner shows when backend is unreachable", async ({ page }) => {
    // Block backend requests
    await page.route("**/api/backend/features", (route) => route.abort());
    await page.route("**/api/backend/info", (route) => route.abort());
    await page.goto("/");

    // Error banner should appear
    await expect(
      page.getByText(/Cannot connect to backend/)
    ).toBeVisible({ timeout: 10000 });

    // Dismiss it
    await page.getByRole("button", { name: /Dismiss/ }).click();
    await expect(
      page.getByText(/Cannot connect to backend/)
    ).not.toBeVisible();
  });

  test("no labeled features message shows with empty registry", async ({
    page,
  }) => {
    // Return empty feature labels
    await page.route("**/api/backend/features", (route) =>
      route.fulfill({ json: {} })
    );
    await page.route("**/api/backend/info", (route) =>
      route.fulfill({ json: MOCK_INFO })
    );
    await page.goto("/");

    await expect(
      page.getByText("No features available")
    ).toBeVisible({ timeout: 5000 });
  });
});
