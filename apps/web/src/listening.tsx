import { useEffect, useRef, useState } from "react";
import { items, object, parseJob, post, request, strings, text, type Job } from "./api";

type Rating = { a: Record<string, number>; b: Record<string, number>; preference: string };
type Session = { id: string; status: "pending" | "rated"; media: Record<string, string>; mapping?: Record<string, unknown>; rating?: Rating };
function parseRating(value: unknown): Rating {
  const data = object(value), preference = text(data.preference);
  if (!["A", "B", "tie"].includes(preference)) throw new Error("Invalid listening preference");
  const scores = (value: unknown) => {
    const data = object(value);
    return Object.fromEntries(Object.keys(dimensions).map(name => {
      const score = data[name];
      if (typeof score !== "number" || !Number.isInteger(score) || score < 1 || score > 5) throw new Error("Invalid listening score");
      return [name, score];
    }));
  };
  return { a: scores(data.a), b: scores(data.b), preference };
}
function parseSession(value: unknown): Session {
  const data = object(value), status = text(data.status), media = strings(data.media), id = text(data.id);
  if (!/^[a-f0-9]{32}$/.test(id) || !["pending", "rated"].includes(status)) throw new Error("Invalid listening session");
  for (const label of ["A", "B", "source", "reference"]) {
    if (!new RegExp(`^/listening-sessions/${id}/audio/${label}\\?listener_id=[a-f0-9]{32}$`).test(media[label] ?? "")) throw new Error("Invalid listening media");
  }
  return { id, status: status === "rated" ? "rated" : "pending", media,
    mapping: status === "rated" ? object(data.mapping) : undefined,
    rating: status === "rated" ? parseRating(data.rating) : undefined };
}
const dimensions = { naturalness: "Naturalness", singer_similarity: "Singer similarity", content_preservation: "Content preservation", artifact_severity: "Artifact severity (higher is worse)" };
const message = (error: unknown) => error instanceof Error ? error.message : "Listening request failed";
function parseSummary(value: unknown) {
  const data = object(value);
  const count = (value: unknown) => {
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) throw new Error("Invalid listening count");
    return value;
  };
  return { count: count(data.rated_session_count), limitations: items(data.limitations).map(text), runs: items(data.runs).map(value => {
    const run = object(value), means = object(run.means);
    return { id: text(run.run_id), count: count(run.rating_count), preferred: count(run.preferred_count), ties: count(run.tie_count),
      means: Object.keys(dimensions).map(name => {
        const mean = means[name];
        if (typeof mean !== "number" || !Number.isFinite(mean) || mean < 1 || mean > 5) throw new Error("Invalid listening mean");
        return mean;
      }) };
  }) };
}

export function Listening({ jobs }: { jobs: Job[] }) {
  const [listener] = useState(() => {
    try {
      const stored = localStorage.getItem("nsvp-listener");
      if (stored && /^[a-f0-9]{32}$/.test(stored)) return { id: stored, persistent: true };
    } catch { /* Storage can be disabled by browser privacy settings. */ }
    const id = crypto.randomUUID().replaceAll("-", "");
    try { localStorage.setItem("nsvp-listener", id); return { id, persistent: true }; }
    catch { return { id, persistent: false }; }
  });
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [savedRuns, setSavedRuns] = useState<Job[]>([]);
  const [hasMoreRuns, setHasMoreRuns] = useState(true), [loadingRuns, setLoadingRuns] = useState(false);
  const [runOffset, setRunOffset] = useState(0), [runError, setRunError] = useState("");
  const [history, setHistory] = useState<Session[]>([]), [session, setSession] = useState<Session>();
  const [error, setError] = useState(""), [busy, setBusy] = useState(false);
  const [summary, setSummary] = useState<ReturnType<typeof parseSummary>>();
  const [summaryError, setSummaryError] = useState(""), [summaryRevision, setSummaryRevision] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null), players = useRef<Record<string, HTMLAudioElement | null>>({});
  const current = useRef<string>("");
  useEffect(() => {
    const found: Record<string, string> = {};
    for (const job of [...savedRuns, ...jobs]) {
      const results = job.result.results ? items(job.result.results).map(object) : [{ conversion: job.result }];
      for (const row of results) {
        if (!row.conversion) continue;
        const conversion = object(row.conversion);
        if (!conversion.artifacts) continue;
        const id = strings(conversion.artifacts)["run_manifest.json"];
        if (id) found[id] = `${typeof row.case_id === "string" ? row.case_id + " / " : ""}${typeof row.configuration_id === "string" ? row.configuration_id + " / " : ""}${text(conversion.conversion_id).slice(0, 12)}`;
      }
    }
    setChoices(previous => ({ ...previous, ...found }));
  }, [jobs, savedRuns]);
  useEffect(() => {
    const controller = new AbortController();
    setLoadingRuns(true);
    request("/conversion-runs?limit=50", { signal: controller.signal }).then(value => {
      const page = items(value).map(parseJob);
      setSavedRuns(page); setRunOffset(page.length); setHasMoreRuns(page.length === 50);
    }).catch(error => { if (!controller.signal.aborted) setRunError(message(error)); })
      .finally(() => { if (!controller.signal.aborted) setLoadingRuns(false); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    request("/listening-sessions?listener_id=" + listener.id, { signal: controller.signal }).then(value => setHistory(items(value).map(parseSession)))
      .catch(error => { if (!controller.signal.aborted) setError(message(error)); });
    return () => controller.abort();
  }, [listener]);
  useEffect(() => {
    const controller = new AbortController();
    setSummary(undefined); setSummaryError("");
    request("/listening-summary?listener_id=" + listener.id, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setSummary(parseSummary(value)); })
      .catch(error => { if (!controller.signal.aborted) setSummaryError(message(error)); });
    return () => controller.abort();
  }, [listener.id, summaryRevision]);
  useEffect(() => {
    if (session && !dialog.current?.open) {
      document.querySelectorAll("audio").forEach(player => player.pause());
      dialog.current?.showModal();
    }
  }, [session]);
  function close() {
    Object.values(players.current).forEach(player => player?.pause()); current.current = ""; setSession(undefined);
  }
  function playing(label: string) {
    const previous = players.current[current.current], next = players.current[label];
    if (["A", "B"].includes(label) && ["A", "B"].includes(current.current) && current.current !== label && previous && next) {
      next.currentTime = Math.min(previous.currentTime, Number.isFinite(next.duration) ? next.duration : previous.currentTime);
    }
    Object.entries(players.current).forEach(([name, player]) => { if (name !== label) player?.pause(); });
    current.current = label;
  }
  function remember(next: Session) {
    setSession(next); setHistory(previous => [next, ...previous.filter(item => item.id !== next.id)]);
    if (next.status === "rated") setSummaryRevision(value => value + 1);
  }
  return <section className="panel results"><h2>Blind listening comparison</h2>
    <p>Compare two runs with the same source and reference. A/B order is randomized. Your anonymous listener ID stays in this browser.</p>
    {!listener.persistent && <p role="status">Browser storage is unavailable. Listening history will not be recoverable after reloading this page.</p>}
    {loadingRuns && <p role="status">Loading saved conversions...</p>}
    {runError && <p role="alert">{runError}</p>}
    {(hasMoreRuns || runError) && <button className="secondary" disabled={loadingRuns || busy} onClick={async () => {
      setLoadingRuns(true); setRunError("");
      try {
        const page = items(await request(`/conversion-runs?limit=50&offset=${runOffset}`)).map(parseJob);
        setSavedRuns(previous => [...previous, ...page]); setRunOffset(previous => previous + page.length);
        setHasMoreRuns(page.length === 50);
      } catch (error) { setRunError(message(error)); } finally { setLoadingRuns(false); }
    }}>{runError ? "Retry loading conversions" : "Load older conversions"}</button>}
    <form onSubmit={async event => {
      event.preventDefault(); const data = new FormData(event.currentTarget); setBusy(true); setError("");
      try { remember(parseSession(await post("/listening-sessions", { listener_id: listener.id, first_manifest_id: data.get("first"), second_manifest_id: data.get("second") }))); }
      catch (error) { setError(message(error)); } finally { setBusy(false); }
    }}><fieldset disabled={busy}>
      {["first", "second"].map((name, index) => <label key={name}>Run {index + 1}<select name={name} required defaultValue=""><option value="">Choose a run</option>{Object.entries(choices).map(([id, title]) => <option key={id} value={id}>{title}</option>)}</select></label>)}
      <button disabled={Object.keys(choices).length < 2}>Start blind comparison</button>
    </fieldset></form>
    {Object.keys(choices).length < 2 && <p>Complete a benchmark comparison or two conversions to choose runs.</p>}
    {!session && error && <p role="alert">{error}</p>}
    <h3>Listening history</h3>{history.length === 0 && <p>No listening sessions yet.</p>}
    {history.map(item => <button className="secondary" disabled={busy} key={item.id} onClick={async () => {
      setError(""); setBusy(true);
      try { remember(parseSession(await request(`/listening-sessions/${item.id}?listener_id=${listener.id}`))); }
      catch (error) { setError(message(error)); } finally { setBusy(false); }
    }}>{item.id.slice(0, 8)}: {item.status === "rated" ? "Rated" : "Resume rating"}</button>)}
    <h3>Your listening results</h3>
    {summaryError ? <><p role="alert">{summaryError}</p><button className="secondary" onClick={() => setSummaryRevision(value => value + 1)}>Retry listening results</button></>
      : !summary ? <p role="status">Loading listening results...</p> : <>
        <p>Rated sessions: {summary.count}</p>
        {summary.limitations.map(note => <p key={note}>{note}</p>)}
        {summary.runs.map(run => <article key={run.id} aria-label={`Listening results for ${run.id}`}>
          <h4>Run {run.id.slice(0, 12)}</h4>
          <p>Ratings: {run.count}. Preferred: {run.preferred}. Ties: {run.ties}.</p>
          <dl>{Object.values(dimensions).map((title, index) => <div key={title}><dt>{title}</dt><dd>{run.means[index].toFixed(2)} / 5</dd></div>)}</dl>
        </article>)}
      </>}
    <dialog ref={dialog} className="listening-dialog" aria-labelledby="listening-title" onClose={close} onCancel={event => { if (busy) event.preventDefault(); }}>
      {session && <><h2 id="listening-title">{session.status === "pending" ? "Blind A/B listening" : "Completed listening"}</h2>
        <p>Play A or B to switch at the same position. Source and reference play independently. Volume is unchanged.</p>
        {["A", "B", "source", "reference"].map(label => <label key={session.id + label}>{label}<audio ref={node => { players.current[label] = node; }} controls preload="metadata" aria-label={`Listen to ${label}`} src={"/api" + session.media[label]} onPlay={() => playing(label)}/></label>)}
        {error && <p role="alert">{error}</p>}
        {session.status === "pending" ? <form key={session.id} onSubmit={async event => {
          event.preventDefault(); const data = new FormData(event.currentTarget); setBusy(true); setError("");
          const scores = (label: string) => Object.fromEntries(Object.keys(dimensions).map(name => [name, Number(data.get(`${label}-${name}`))]));
          try { remember(parseSession(await post(`/listening-sessions/${session.id}/ratings?listener_id=${listener.id}`, { a: scores("A"), b: scores("B"), preference: data.get("preference") }))); }
          catch (error) { setError(message(error)); } finally { setBusy(false); }
        }}><p>Score 1 to 5. Higher is better, except artifact severity. Submitted ratings cannot be edited.</p>
          <fieldset disabled={busy}>{["A", "B"].map(label => <div key={label}><h3>{label}</h3>{Object.entries(dimensions).map(([name, title]) => <label key={name}>{title}<select required name={`${label}-${name}`} defaultValue=""><option value="">Choose a score</option>{[1, 2, 3, 4, 5].map(score => <option key={score}>{score}</option>)}</select></label>)}</div>)}
            <label>Overall preference<select required name="preference" defaultValue=""><option value="">Choose preference</option><option>A</option><option>B</option><option value="tie">Tie</option></select></label><button>Save ratings and reveal runs</button>
          </fieldset></form> : <><p>Ratings saved. Run identities are now visible.</p>{Object.entries(session.mapping ?? {}).map(([label, value]) => {
            const entry = object(value), executions = object(entry.executions);
            return <p key={label}>{label}: {text(entry.run_id).slice(0, 12)}. {Object.values(executions).map(value => text(object(value).provider)).join(", ")}</p>;
          })}{session.rating && <>
            <p>Overall preference: {session.rating.preference === "tie" ? "Tie" : session.rating.preference}</p>
            <table><caption>Saved listening scores</caption><thead><tr><th scope="col">Dimension</th><th scope="col">A</th><th scope="col">B</th></tr></thead>
              <tbody>{Object.entries(dimensions).map(([name, title]) => <tr key={name}><th scope="row">{title}</th><td>{session.rating?.a[name]}</td><td>{session.rating?.b[name]}</td></tr>)}</tbody>
            </table>
          </>}</>}
        <button className="secondary" disabled={busy} onClick={() => dialog.current?.close()}>Close listening session</button>
      </>}
    </dialog>
  </section>;
}
