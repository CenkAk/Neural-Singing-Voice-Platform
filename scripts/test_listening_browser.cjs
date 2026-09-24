// Run against scripts/browser_fixture.py, never against an operator's dataset.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require(process.env.NSVP_PLAYWRIGHT_MODULE || "playwright");

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
    const screenshotDirectory = process.env.NSVP_SCREENSHOT_DIRECTORY;
    if (screenshotDirectory) await fs.mkdir(screenshotDirectory, { recursive: true });
    const fixture = await page.request.get("http://127.0.0.1:8873/__fixture__");
    assert.deepEqual(await fixture.json(), { fixture: "nsvp-disposable-browser-v03" });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto("http://127.0.0.1:8873");
    await page.getByRole("heading", { name: "Blind listening comparison" }).waitFor();
    await page.waitForFunction(() => document.querySelector('select[name="first"]')?.options.length === 3);
    if (screenshotDirectory) await page.screenshot({ path: path.join(screenshotDirectory, "01-workspace.png") });
    await page.locator('select[name="first"]').selectOption({ index: 1 });
    await page.locator('select[name="second"]').selectOption({ index: 2 });
    await page.getByRole("button", { name: "Start blind comparison" }).click();
    const dialog = page.getByRole("dialog");
    await dialog.waitFor();
    assert.equal(await dialog.locator("select").count(), 9);
    for (const select of await dialog.locator("select").all()) assert.equal(await select.inputValue(), "");
    assert.equal((await dialog.innerText()).includes("synthetic-a"), false);
    if (screenshotDirectory) {
      await page.setViewportSize({ width: 1440, height: 1800 });
      await dialog.screenshot({ path: path.join(screenshotDirectory, "02-blind-listening.png") });
      await page.setViewportSize({ width: 1440, height: 1000 });
    }
    const a = dialog.getByLabel("Listen to A", { exact: true });
    const b = dialog.getByLabel("Listen to B", { exact: true });
    await a.evaluate(async audio => { await audio.play(); audio.currentTime = 0.8; audio.pause(); });
    await b.evaluate(audio => audio.play());
    assert.ok(await b.evaluate(audio => audio.currentTime >= 0.8));
    await b.evaluate(audio => audio.pause());
    assert.equal(await a.evaluate(audio => audio.paused), true);
    for (const label of ["A", "B"]) for (const dimension of ["naturalness", "singer_similarity", "content_preservation", "artifact_severity"]) {
      await dialog.locator(`select[name="${label}-${dimension}"]`).selectOption(label === "A" ? "2" : "4");
    }
    await dialog.locator('select[name="preference"]').selectOption("tie");
    await dialog.getByRole("button", { name: "Save ratings and reveal runs" }).click();
    await dialog.getByText("Ratings saved. Run identities are now visible.").waitFor();
    assert.ok((await dialog.innerText()).includes("synthetic-a"));
    await page.keyboard.press("Escape");
    await page.getByText("Rated sessions: 1", { exact: true }).waitFor();
    assert.equal(await page.getByText("Ratings: 1. Preferred: 0. Ties: 1.", { exact: true }).count(), 2);
    assert.equal(await page.getByText("2.00 / 5", { exact: true }).count(), 4);
    assert.equal(await page.getByText("4.00 / 5", { exact: true }).count(), 4);
    await page.reload();
    await page.getByText("Rated sessions: 1", { exact: true }).waitFor();
    await page.waitForFunction(() => document.querySelector('select[name="first"]')?.options.length === 3);
    await page.getByRole("button", { name: /: Rated$/ }).click();
    await page.getByText("Ratings saved. Run identities are now visible.").waitFor();
    await dialog.getByText("Overall preference: Tie", { exact: true }).waitFor();
    const scores = dialog.getByRole("table", { name: "Saved listening scores" });
    assert.deepEqual(await scores.getByRole("cell").allTextContents(), ["2", "4", "2", "4", "2", "4", "2", "4"]);
    const sourceUrl = await dialog.getByLabel("Listen to source", { exact: true }).getAttribute("src");
    const sourceAudio = await page.request.get(new URL(sourceUrl, page.url()).href);
    assert.equal(sourceAudio.ok(), true);
    const upload = { name: "synthetic-browser.wav", mimeType: "audio/wav", buffer: await sourceAudio.body() };
    await page.keyboard.press("Escape");
    await page.getByLabel("Source audio", { exact: true }).setInputFiles(upload);
    await page.getByLabel("Voice reference", { exact: true }).setInputFiles(upload);
    await page.getByLabel(/^Source type/).selectOption("vocal");
    await page.getByLabel(/^Backend/).selectOption("cpu");
    await page.getByLabel(/^Comparison converter/).selectOption("seed_vc");
    const queued = page.waitForResponse(response => response.url().endsWith("/api/benchmark-runs") && response.request().method() === "POST");
    await page.getByRole("button", { name: "Run comparison", exact: true }).click();
    assert.equal((await queued).ok(), true);
    await page.getByText("benchmark: SUCCEEDED", { exact: true }).waitFor({ timeout: 30000 });
    const selected = page.locator("details").filter({ hasText: "uploaded-audio / selected:" });
    await selected.locator("summary").click();
    await selected.getByRole("table", { name: /Evaluation:/ }).waitFor();
    await selected.getByRole("region", { name: "Evaluation limitations" }).getByText("Spectral differences include intended timbre changes; they do not establish perceptual quality.", { exact: true }).waitFor();
    const pitch = selected.getByRole("img", { name: "F0 comparison", exact: true });
    await pitch.waitFor();
    assert.equal(await pitch.locator("path").count(), 2);
    for (const path of await pitch.locator("path").all()) {
      const drawing = await path.getAttribute("d");
      assert.ok(drawing.startsWith("M") && !drawing.includes("NaN"));
    }
    const outputAudio = selected.getByLabel("converted_vocal_raw.wav", { exact: true });
    await outputAudio.evaluate(async audio => { await audio.play(); audio.pause(); });
    assert.ok(await outputAudio.evaluate(audio => audio.duration > 0));
    await selected.getByRole("button", { name: "Show waveform: converted_vocal_raw.wav", exact: true }).click();
    const waveform = selected.getByRole("img", { name: "Waveform: converted_vocal_raw.wav", exact: true });
    await waveform.waitFor();
    const drawing = await waveform.locator("path").getAttribute("d");
    assert.equal(drawing.includes("NaN"), false);
    assert.equal((drawing.match(/M/g) || []).length, 512);
    assert.deepEqual(errors, []);
    if (screenshotDirectory) {
      await pitch.screenshot({ path: path.join(screenshotDirectory, "03-pitch-comparison.png") });
      await waveform.screenshot({ path: path.join(screenshotDirectory, "04-waveform.png") });
    }
    console.log("PASS: uploads, benchmark queue/poll/results, artifact playback, blind ratings, persistence, no page errors");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
