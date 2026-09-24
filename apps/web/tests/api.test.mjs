import assert from "node:assert/strict";
import test from "node:test";
import { artifactUrl, parseCatalog, parseJob, request, terminal } from "../src/api.ts";

const job = { id: "fixture", kind: "conversion", state: "RUNNING", stage: "converting", progress: 0.5 };

test("job parsing rejects invalid progress and preserves terminal failures", () => {
  for (const progress of [NaN, Infinity, -0.1, 1.1, "0.5", null]) {
    assert.throws(() => parseJob({ ...job, progress }), /Invalid job progress/);
  }
  assert.equal(terminal(parseJob(job)), false);
  for (const state of ["SUCCEEDED", "FAILED", "CANCELLED"]) {
    assert.equal(terminal(parseJob({ ...job, state })), true);
  }
  assert.equal(parseJob({ ...job, state: "FAILED", error: { message: "Model unavailable" } }).error, "Model unavailable");
  assert.throws(() => parseJob({ ...job, result: [] }), /Invalid API object/);
});

test("provider availability requires booleans, not truthy API strings", () => {
  const provider = { name: "seed_vc", task: "svc", stability: "experimental", installed: false,
    configured: false, backends: { rocm: "not_tested" }, warnings: [] };
  const catalog = { providers: [provider], profiles: {}, vocal_processing_profiles: [] };
  assert.equal(parseCatalog(catalog).providers[0].installed, false);
  assert.throws(() => parseCatalog({ ...catalog, providers: [{ ...provider, installed: "false" }] }), /availability/);
  assert.throws(() => parseCatalog({ ...catalog, providers: [{ ...provider, backends: { rocm: true } }] }), /API text/);
  assert.equal(artifactUrl("run/hash/audio.wav?download=1#part"), "/api/artifacts/run%2Fhash%2Faudio.wav%3Fdownload%3D1%23part");
});

test("request handles structured errors, non-JSON failures and cancellation", async context => {
  const fetch = context.mock.method(globalThis, "fetch");
  fetch.mock.mockImplementation(async () => new Response(JSON.stringify({ detail: "Missing model" }), { status: 400 }));
  await assert.rejects(request("/jobs"), /Missing model/);
  fetch.mock.mockImplementation(async () => new Response("Bad gateway", { status: 502 }));
  await assert.rejects(request("/jobs"), /Request failed \(502\)/);
  fetch.mock.mockImplementation(async () => { throw new DOMException("Cancelled", "AbortError"); });
  await assert.rejects(request("/jobs"), { name: "AbortError" });
  fetch.mock.mockImplementation(async () => Response.json({ state: "SUCCEEDED" }));
  assert.deepEqual(await request("/jobs"), { state: "SUCCEEDED" });
});
