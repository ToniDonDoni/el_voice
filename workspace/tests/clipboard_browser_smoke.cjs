/* Live Chromium/clipboard/provider smoke. Run with Playwright available in NODE_PATH.
 * Uses a synthetic screenshot only. Remote Voice must already be running.
 * Set EL_VOICE_TEST_ARTIFACTS to an existing ignored work directory for screenshots.
 */
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ permissions: ["clipboard-read", "clipboard-write"],
    viewport: { width: 1100, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    window.microphoneRequests = 0;
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = (...args) => {
      window.microphoneRequests += 1;
      return original(...args);
    };
  });
  try {
    await page.goto(process.env.EL_VOICE_TEST_URL || "http://localhost:8765/");
    await page.getByText("Remote Voice ready", { exact: true }).waitFor({ timeout: 60000 });
    await page.evaluate(async () => {
      const canvas = document.createElement("canvas");
      canvas.width = 500; canvas.height = 220;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "white"; ctx.fillRect(0, 0, 500, 220);
      ctx.fillStyle = "red"; ctx.fillRect(40, 50, 120, 120);
      ctx.fillStyle = "blue"; ctx.beginPath(); ctx.arc(350, 110, 65, 0, Math.PI * 2); ctx.fill();
      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    });
    await page.getByRole("button", { name: "Paste image", exact: true }).click();
    await page.locator("#composer-status").filter({ hasText: "Screenshot ready" }).waitFor();
    await page.locator("#question-text").fill("Опиши фигуры и их цвета на этой картинке. Коротко.");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await page.locator(".message-status").filter({ hasText: "Response generated" }).last().waitFor({ timeout: 90000 });
    const answer = await page.locator(".message.assistant .message-body").last().innerText();
    assert.match(answer, /красн/i); assert.match(answer, /квадрат/i);
    assert.match(answer, /син/i); assert.match(answer, /круг/i);
    assert.equal(await page.locator("#image-preview").isVisible(), false);
    assert.equal(await page.locator(".message.user .message-image").count(), 1);
    assert.equal(await page.evaluate(() => window.microphoneRequests), 0);
    if (process.env.EL_VOICE_TEST_ARTIFACTS) {
      await page.locator("#question-form").scrollIntoViewIfNeeded();
      await page.screenshot({ path: `${process.env.EL_VOICE_TEST_ARTIFACTS}/clipboard-image-answer.png`, fullPage: true });
    }
    // A follow-up sends text only and still uses the established session.
    await page.locator("#question-text").fill("Сколько будет два плюс два? Ответь полным предложением, число напиши словом.");
    await page.locator("#question-text").press("Enter");
    await page.waitForFunction(() => document.querySelectorAll(".message.assistant").length === 2);
    await page.locator(".message.assistant").last().locator(".message-status").filter({ hasText: "Response generated" }).waitFor({ timeout: 90000 });
    const followup = await page.locator(".message.assistant .message-body").last().innerText();
    assert.match(followup, /4|четыр[её]/i);
    await page.locator("#question-text").focus();
    await page.keyboard.press(process.platform === "darwin" ? "Meta+V" : "Control+V");
    await page.locator("#composer-status").filter({ hasText: "Screenshot ready" }).waitFor();
    await page.getByRole("button", { name: "Remove image", exact: true }).click();
    assert.equal(await page.locator("#image-preview").isVisible(), false);
    await page.setViewportSize({ width: 390, height: 900 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    if (process.env.EL_VOICE_TEST_ARTIFACTS) {
      await page.screenshot({ path: `${process.env.EL_VOICE_TEST_ARTIFACTS}/clipboard-mobile.png`, fullPage: true });
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ image_answer: answer, text_answer: followup,
      microphone_requests: 0, screenshot_consumed: true, keyboard_paste_and_remove: true,
      mobile_overflow: false, browser_errors: errors }));
    await page.getByRole("button", { name: "End session", exact: true }).click();
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
