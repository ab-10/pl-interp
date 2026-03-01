import { test, expect } from "@playwright/test";

/** Mock /features response — flat dict of feature labels. */
const MOCK_FEATURES = {
  "100": "Type annotations (Python/TypeScript)",
  "200": "Error handling patterns",
  "300": "Recursive patterns",
};

/** Mock /info response. */
const MOCK_INFO = {
  model: "mistralai/Mistral-7B-Instruct-v0.3",
  sae: "sae_checkpoint.pt",
  layer: 16,
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
      page.getByText("Steer Mistral 7B code generation")
    ).toBeVisible();

    // Prompt input with default value
    const textarea = page.locator("textarea");
    await expect(textarea).toBeVisible();
    await expect(textarea).toHaveValue("def fibonacci(n):");

    // Generate button
    await expect(
      page.getByRole("button", { name: /Generate/ })
    ).toBeVisible();
  });

  test("model info displays correctly", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");

    await expect(
      page.getByText("Model: mistralai/Mistral-7B-Instruct-v0.3")
    ).toBeVisible({ timeout: 5000 });
    await expect(page.getByText("SAE: sae_checkpoint.pt (layer 16)")).toBeVisible();
  });

  test("features load as flat list", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");

    // Feature labels
    await expect(
      page.getByText("Type annotations (Python/TypeScript)")
    ).toBeVisible({ timeout: 5000 });
    await expect(page.getByText("Error handling patterns")).toBeVisible();
    await expect(page.getByText("Recursive patterns")).toBeVisible();

    // Temperature slider + 3 feature sliders = 4 range inputs
    const sliders = page.locator('input[type="range"]');
    await expect(sliders).toHaveCount(4);
  });

  test("feature sliders default to 0 and can be adjusted", async ({
    page,
  }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByText("Type annotations")).toBeVisible({ timeout: 5000 });

    // Feature sliders (skip the temperature slider which is first)
    const featureSliders = page.locator(
      'input[type="range"][min="-10"][max="10"]'
    );
    const count = await featureSliders.count();
    expect(count).toBe(3);

    for (let i = 0; i < count; i++) {
      await expect(featureSliders.nth(i)).toHaveValue("0");
    }

    // Adjust first feature slider
    await featureSliders.first().fill("5");
    await expect(featureSliders.first()).toHaveValue("5");
    await expect(page.getByText("+5.0")).toBeVisible();
  });

  test("generate produces diff output", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByText("Type annotations")).toBeVisible({ timeout: 5000 });

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

  test("custom feature input works without layer selector", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByText("Custom Features")).toBeVisible({
      timeout: 5000,
    });

    // No layer dropdown
    const selects = page.locator("select");
    await expect(selects).toHaveCount(0);

    // Feature ID input
    const featureInput = page.locator('input[placeholder="e.g. 1234"]');
    await expect(featureInput).toBeVisible();
  });

  test("can add and remove custom features", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/");
    await expect(page.getByText("Custom Features")).toBeVisible({
      timeout: 5000,
    });

    // Add a custom feature
    await page.locator('input[placeholder="e.g. 1234"]').fill("42");
    await page.getByRole("button", { name: "Add" }).click();

    // Custom feature slider appears
    await expect(page.getByText("#42")).toBeVisible();

    // Remove it
    await page.getByTitle("Remove").click();
    await expect(page.getByText("#42")).not.toBeVisible();
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
      page.getByText("No labeled features yet")
    ).toBeVisible({ timeout: 5000 });
  });
});
