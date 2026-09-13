export function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid API object");
  return Object.fromEntries(Object.entries(value));
}
export function items(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new Error("Invalid API list");
  return value;
}
export function text(value: unknown): string {
  if (typeof value !== "string") throw new Error("Invalid API text");
  return value;
}
export function strings(value: unknown): Record<string, string> {
  return Object.fromEntries(Object.entries(object(value)).map(([key, entry]) => [key, text(entry)]));
}
export async function request(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    let detail: unknown;
    try { detail = object(await response.json()).detail; } catch { /* Non-JSON errors use the HTTP status. */ }
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  return response.json();
}
export const post = (path: string, payload: unknown) => request(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
});
export async function upload(file: File): Promise<string> {
  const body = new FormData(); body.append("file", file);
  return text(object(await request("/uploads", { method: "POST", body })).artifact_id);
}
export type Job = { id: string; kind: string; state: string; stage: string; progress: number; error?: string; result: Record<string, unknown> };
export function parseJob(value: unknown): Job {
  const job = object(value);
  if (typeof job.progress !== "number" || !Number.isFinite(job.progress) || job.progress < 0 || job.progress > 1) throw new Error("Invalid job progress");
  return { id: text(job.id), kind: text(job.kind), state: text(job.state), stage: text(job.stage), progress: job.progress,
    error: job.error ? text(object(job.error).message) : undefined, result: job.result ? object(job.result) : {} };
}
export const terminal = (job?: Job) => !job || ["SUCCEEDED", "FAILED", "CANCELLED"].includes(job.state);
export const artifactUrl = (id: string) => `/api/artifacts/${encodeURIComponent(id)}`;
export type Provider = { name: string; task: string; stability: string; installed: boolean; configured: boolean; backends: Record<string, string>; warnings: string[] };
export type Catalog = { providers: Provider[]; profiles: Record<string, string>; vocalProfiles: string[] };
export function parseCatalog(value: unknown): Catalog {
  const catalog = object(value);
  return {
    providers: items(catalog.providers).map(entry => {
      const provider = object(entry);
      if (typeof provider.installed !== "boolean" || typeof provider.configured !== "boolean") throw new Error("Invalid provider availability");
      return { name: text(provider.name), task: text(provider.task), stability: text(provider.stability),
        installed: provider.installed, configured: provider.configured, backends: strings(provider.backends), warnings: items(provider.warnings).map(text) };
    }),
    profiles: Object.fromEntries(Object.entries(object(catalog.profiles)).map(([name, profile]) => [name, text(object(profile).provider)])),
    vocalProfiles: items(catalog.vocal_processing_profiles).map(text),
  };
}
