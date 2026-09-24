import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { artifactUrl, items, object, parseCatalog, parseJob, post, request, strings, terminal, text, upload, type Catalog, type Job } from "./api";
import "./styles.css";
import { Listening } from "./listening";

const errorText = (error: unknown) => error instanceof Error ? error.message : "Request failed";
type Source = { id: string; artifact: string; label: string };
type Model = { id: string; mode: string; status: string };

function Waveform({ url, name }: { url: string; name: string }) {
  const [requested, setRequested] = useState(false), [path, setPath] = useState("");
  const [duration, setDuration] = useState(0), [error, setError] = useState("");
  useEffect(() => {
    if (!requested) return;
    const controller = new AbortController();
    let context: AudioContext | undefined;
    async function load() {
      try {
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok) throw new Error(`Audio preview failed (${response.status})`);
        const size = Number(response.headers.get("content-length"));
        if (!Number.isFinite(size) || size <= 0 || size > 64 * 1024 * 1024) throw new Error("Waveform preview requires an audio file of 64 MB or less with a known size.");
        const bytes = await response.arrayBuffer();
        if (controller.signal.aborted) return;
        context = new AudioContext();
        const audio = await context.decodeAudioData(bytes);
        if (controller.signal.aborted) return;
        const channels = Array.from({ length: audio.numberOfChannels }, (_, index) => audio.getChannelData(index));
        const points: string[] = [];
        const bins = Math.min(512, audio.length);
        for (let bin = 0; bin < bins; bin++) {
          let low = 0, high = 0;
          const start = Math.floor(bin * audio.length / bins), end = Math.floor((bin + 1) * audio.length / bins);
          for (const channel of channels) for (let index = start; index < end; index++) {
            low = Math.min(low, channel[index]); high = Math.max(high, channel[index]);
          }
          const x = bin * 512 / bins;
          points.push(`M${x.toFixed(2)},${(50 - Math.min(1, high) * 48).toFixed(2)}V${(50 - Math.max(-1, low) * 48).toFixed(2)}`);
        }
        setPath(points.join(" ")); setDuration(audio.duration);
      } catch (error) { if (!controller.signal.aborted) setError(errorText(error)); }
      finally { if (context && context.state !== "closed") await context.close(); }
    }
    void load();
    return () => { controller.abort(); };
  }, [requested, url]);
  if (!requested) return <button className="secondary" onClick={() => setRequested(true)}>Show waveform: {name}</button>;
  if (error) return <p role="alert">{error}</p>;
  if (!path) return <p role="status">Loading waveform...</p>;
  return <figure><svg viewBox="0 0 512 100" width="100%" role="img" aria-label={`Waveform: ${name}`}>
    <path d={path} fill="none" stroke="currentColor" strokeWidth="1"/>
  </svg><figcaption>0 to {duration.toFixed(2)} seconds. Combined channel peaks; display range -1 to 1.</figcaption></figure>;
}

function Artifacts({ value }: { value: unknown }) {
  return <div className="artifacts">{Object.entries(strings(value)).map(([name, id]) => <div className="artifact" key={name}>
    <a href={artifactUrl(id)} download>{name} <span>Download</span></a>
    {/\.(wav|flac|mp3|ogg|m4a|opus)$/i.test(name) && <><audio aria-label={name} controls preload="none" src={artifactUrl(id)}/><Waveform url={artifactUrl(id)} name={name}/></>}
  </div>)}</div>;
}
function PitchComparison({ value }: { value: unknown }) {
  const preview = object(value);
  const points = (value: unknown): Array<[number, number | null]> => items(value).map(value => {
    const pair = items(value), time = pair[0], hz = pair[1];
    if (pair.length !== 2 || typeof time !== "number" || !Number.isFinite(time) || time < 0 ||
      (hz !== null && (typeof hz !== "number" || !Number.isFinite(hz) || hz <= 0))) throw new Error("Invalid F0 preview");
    return [time, hz];
  });
  const source = points(preview.source), output = points(preview.output);
  const frequencies = [...source, ...output].map(point => point[1]).filter((hz): hz is number => hz !== null);
  if (!frequencies.length) return <p>No voiced frames available for an F0 comparison.</p>;
  const low = Math.max(0, Math.floor(Math.min(...frequencies) / 10) * 10 - 10), high = Math.ceil(Math.max(...frequencies) / 10) * 10 + 10;
  const end = Math.max(...source.map(point => point[0]), ...output.map(point => point[0]), 0.001);
  const drawing = (track: Array<[number, number | null]>, marker: string) => track.filter(point => point[1] !== null)
    .map(([time, hz]) => `M${(40 + time / end * 590).toFixed(2)},${(120 - ((hz ?? low) - low) / (high - low) * 100).toFixed(2)}${marker}`).join(" ");
  return <figure><svg viewBox="0 0 640 155" width="100%" role="img" aria-label="F0 comparison">
    <text x="0" y="18" fontSize="10">{high} Hz</text><text x="0" y="124" fontSize="10">{low} Hz</text>
    <text x="40" y="148" fontSize="10">0 s</text><text x="630" y="148" textAnchor="end" fontSize="10">{end.toFixed(2)} s</text>
    <path d={drawing(source, "v3")} stroke="#2563eb" fill="none" strokeWidth="2"/>
    <path d={drawing(output, "h3")} stroke="#c2410c" fill="none" strokeWidth="2"/>
  </svg><figcaption>F0 estimates: blue vertical marks show source ({text(preview.source_extractor)}); orange horizontal marks show output ({text(preview.output_extractor)}). Source transposition: {String(preview.source_transpose_semitones)} semitones. Diagnostic estimates, not ground truth. {text(preview.sampling)}.</figcaption></figure>;
}

function Evaluation({ value }: { value: unknown }) {
  const report = object(value);
  const limitations = [...new Set(items(report.limitations ?? []).map(text))];
  return <>{report.pitch_preview != null && <PitchComparison value={report.pitch_preview}/>}<div className="table-scroll"><table><caption>Evaluation: {text(report.input_condition)}</caption>
    <thead><tr><th>Family / metric</th><th>Value</th><th>Status / limitation</th></tr></thead>
    <tbody>{Object.entries(object(report.families)).flatMap(([family, metrics]) => Object.entries(object(metrics)).map(([name, value]) => {
      const metric = object(value), status = text(metric.status);
      const evaluator = metric.evaluator ? object(metric.evaluator) : undefined;
      return <tr key={family + name}><th>{family} / {name}</th><td>{status === "measured" && typeof metric.value === "number" ? metric.value.toPrecision(5) : "Not measured"} {typeof metric.unit === "string" ? metric.unit : ""}</td>
        <td>{status}{typeof metric.reason === "string" && <small>{metric.reason}</small>}{evaluator && <>
          <small>{text(evaluator.name)} {text(evaluator.version)}</small>
          {typeof evaluator.model === "string" && <small>Model: {evaluator.model}</small>}
          {typeof evaluator.domain === "string" && <small>Domain: {evaluator.domain}</small>}
          {typeof evaluator.reference_kind === "string" && <small>Reference: {evaluator.reference_kind}</small>}
          {typeof evaluator.language === "string" && <small>Language: {evaluator.language}</small>}
        </>}</td></tr>;
    }))}</tbody></table></div>{limitations.length > 0 && <section aria-label="Evaluation limitations"><h3>Evaluation limitations</h3><ul>{limitations.map(note => <li key={note}>{note}</li>)}</ul></section>}</>;
}
function Results({ job }: { job: Job }) {
  const [report, setReport] = useState<unknown>(), [error, setError] = useState("");
  const id = job.result.artifacts ? strings(job.result.artifacts)["evaluation_report.json"] : undefined;
  useEffect(() => {
    setReport(undefined); setError("");
    if (!id) return;
    const controller = new AbortController();
    request("/artifacts/" + encodeURIComponent(id), { signal: controller.signal }).then(setReport).catch(error => {
      if (!controller.signal.aborted) setError(errorText(error));
    });
    return () => controller.abort();
  }, [id]);
  return <section className="panel results"><h2>Results</h2>
    {job.result.artifacts != null && <Artifacts value={job.result.artifacts}/>}
    {error && <p role="alert">{error}</p>}{report != null && <Evaluation value={report}/>}
    {job.result.evaluation != null && !id && <Evaluation value={job.result.evaluation}/>}
    {job.result.results != null && items(job.result.results).map((value, index) => {
      const row = object(value), conversion = row.conversion ? object(row.conversion) : undefined;
      return <details key={index}><summary>{text(row.case_id)} / {text(row.configuration_id)}: {text(row.status)}</summary>
        {items(row.warnings).map((warning, i) => <p key={i}>{text(warning)}</p>)}
        {row.evaluation != null && <Evaluation value={row.evaluation}/>}
        {conversion?.artifacts != null && <Artifacts value={conversion.artifacts}/>}
      </details>;
    })}
  </section>;
}
function App() {
  const [catalog, setCatalog] = useState<Catalog>({ providers: [], profiles: {}, vocalProfiles: [] });
  const [models, setModels] = useState<Model[]>([]), [runs, setRuns] = useState<Job[]>([]);
  const [song, setSong] = useState<File>(), [reference, setReference] = useState<File>(), [job, setJob] = useState<Job>();
  const [message, setMessage] = useState("Loading providers"), [error, setError] = useState(""), [busy, setBusy] = useState(false);
  const [converter, setConverter] = useState(""), [separator, setSeparator] = useState("");
  const [converterProfile, setConverterProfile] = useState(""), [separatorProfile, setSeparatorProfile] = useState("");
  const [vocalProfile, setVocalProfile] = useState(""), [modelProfile, setModelProfile] = useState("");
  const [backend, setBackend] = useState("auto"), [precision, setPrecision] = useState("auto");
  const [transpose, setTranspose] = useState(0), [seed, setSeed] = useState(42), [keep, setKeep] = useState(true);
  const [inputKind, setInputKind] = useState("song"), [condition, setCondition] = useState("unknown");
  const [prepared, setPrepared] = useState<{ manifest: string; sources: Source[] }>(), [selected, setSelected] = useState("");
  const [compare, setCompare] = useState("soulx_singer");
  const active = busy || !terminal(job);
  const resetPreparation = () => { setPrepared(undefined); setSelected(""); };

  useEffect(() => {
    const controller = new AbortController(), init = { signal: controller.signal };
    Promise.all([request("/providers", init), request("/models", init), request("/benchmark-runs", init)]).then(([providers, modelList, benchmarks]) => {
      setCatalog(parseCatalog(providers));
      setModels(items(modelList).map(value => { const model = object(value); return {
        id: text(model.model_name) + "/" + text(model.version), mode: text(model.mode), status: text(model.evaluation_status),
      }; }));
      setRuns(items(benchmarks).map(parseJob)); setMessage("Ready for authorized audio");
    }).catch(error => { if (!controller.signal.aborted) setError(errorText(error)); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!job || terminal(job)) return;
    const id = job.id, controller = new AbortController();
    const timer = window.setTimeout(() => {
      request("/jobs/" + encodeURIComponent(id), { signal: controller.signal }).then(parseJob).then(next => {
        setError("");
        setJob(next);
        if (!terminal(next)) return;
        setMessage(next.kind + ": " + next.state);
        if (next.kind === "benchmark") setRuns(previous => [next, ...previous.filter(run => run.id !== next.id)]);
        if (next.kind === "vocal_preparation" && next.state === "SUCCEEDED") {
          const manifest = strings(next.result.artifacts)["vocal_preparation.json"];
          if (!manifest) throw new Error("Preparation manifest is missing");
          setPrepared({ manifest, sources: items(next.result.sources).map(value => {
            const source = object(value); return { id: text(source.source_id), artifact: text(source.artifact_id), label: typeof source.label === "string" ? source.label : text(source.source_id) };
          }) }); setSelected("");
        }
      }).catch(error => { if (!controller.signal.aborted) { setError(errorText(error)); setJob(current => current ? { ...current } : current); } });
    }, 1000);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [job]);

  async function queue(action: "conversion" | "preparation" | "benchmark") {
    if (!song || (action !== "preparation" && !reference)) return;
    setBusy(true); setError(""); setMessage("Uploading audio");
    try {
      const source = prepared && action === "conversion" ? undefined : await upload(song);
      const target = action === "preparation" || !reference ? undefined : await upload(reference);
      const selections = { voice_converter: modelProfile ? null : converter || null, converter_profile: modelProfile ? null : converterProfile || null,
        model_profile: modelProfile || null, backend, precision, transpose_semitones: transpose };
      const processing = { separator: separator || null, separator_profile: separatorProfile || null, vocal_processing_profile: vocalProfile || null };
      let response: unknown;
      if (action === "preparation") {
        resetPreparation();
        response = await post("/vocal-preparation-jobs", { source_artifact_id: source, input_kind: inputKind, input_condition: condition, backend, ...processing });
      } else if (action === "benchmark") {
        response = await post("/benchmark-runs", { name: "Comparison: " + song.name, seeds: [seed],
          cases: [{ case_id: "uploaded-audio", source_artifact_id: source, reference_artifact_id: target, input_kind: inputKind, input_condition: condition }],
          configurations: [{ configuration_id: "selected", ...selections, ...processing },
            { configuration_id: "comparison", ...selections, ...processing, voice_converter: compare, converter_profile: null, model_profile: null }] });
      } else {
        response = await post("/conversion-jobs", { reference_artifact_id: target, output_name: song.name.replace(/\.[^.]+$/, ""),
          ...selections, keep_intermediates: keep, random_seed: seed, input_kind: inputKind, input_condition: condition,
          ...(prepared ? { preparation_manifest_artifact_id: prepared.manifest, selected_source_id: selected } : { song_artifact_id: source, ...processing }) });
      }
      setJob(parseJob(response)); setMessage(action + " queued");
    } catch (error) { setError(errorText(error)); } finally { setBusy(false); }
  }
  const providers = (task: string) => catalog.providers.filter(provider => provider.task === task && provider.stability !== "research_reference")
    .map(provider => <option key={provider.name} value={provider.name}>{provider.name} ({provider.stability}{provider.configured ? "" : ", setup needed"})</option>);
  const profiles = (provider: string, task: string) => Object.entries(catalog.profiles).filter(([, owner]) => provider ? owner === provider : catalog.providers.some(entry => entry.name === owner && entry.task === task))
    .map(([name]) => <option key={name}>{name}</option>);
  const select = (label: string, value: string, change: (value: string) => void, options: React.ReactNode, disabled = false) =>
    <label>{label}<select value={value} onChange={event => change(event.target.value)} disabled={disabled}>{options}</select></label>;
  const defaults = <option value="">Configured default</option>;
  return <main>
    <header><div><span className="eyebrow">AUTHORIZED AUDIO ML WORKSPACE</span><h1>Neural Singing<br/><em>Voice Platform</em></h1></div><div className="status" role="status">{message}</div></header>
    {error && <p className="error" role="alert">{error}</p>}
    <section className="grid"><article className="panel convert"><h2>Convert a voice</h2><p>Preserve an existing singing performance using authorized source and reference audio.</p>
      <form onSubmit={event => { event.preventDefault(); void queue("conversion"); }}><fieldset disabled={active}>
        <label>Source audio<input required type="file" accept="audio/*" onChange={event => { setSong(event.target.files?.[0]); resetPreparation(); }}/></label>
        <label>Voice reference<input required type="file" accept="audio/*" onChange={event => setReference(event.target.files?.[0])}/></label>
        {select("Source type", inputKind, value => { setInputKind(value); resetPreparation(); }, <><option value="song">Song, separate stems first</option><option value="vocal">Vocal stem</option></>)}
        {select("Input condition", condition, value => { setCondition(value); resetPreparation(); }, ["unknown", "clean_lead", "mixed_vocal", "separated_lead"].map(value => <option key={value}>{value}</option>))}
        {select("Converter", converter, value => { setConverter(value); setConverterProfile(""); }, <>{defaults}{providers("svc")}</>, !!modelProfile)}
        {select("Converter profile", converterProfile, setConverterProfile, <>{defaults}{profiles(converter, "svc")}</>, !!modelProfile)}
        {select("Registered model", modelProfile, setModelProfile, <><option value="">Use provider configuration</option>{models.map(model => <option value={model.id} key={model.id}>{model.id} ({model.mode})</option>)}</>)}
        {select("Separator", separator, value => { setSeparator(value); setSeparatorProfile(""); }, <>{defaults}{providers("separation")}</>, inputKind === "vocal" || !!prepared)}
        {select("Separator profile", separatorProfile, setSeparatorProfile, <>{defaults}{profiles(separator, "separation")}</>, inputKind === "vocal" || !!prepared)}
        {select("Vocal processing", vocalProfile, setVocalProfile, <>{defaults}{catalog.vocalProfiles.map(name => <option key={name}>{name}</option>)}</>, !!prepared)}
        {select("Backend", backend, setBackend, ["auto", "cpu", "cuda", "rocm", "mps", "directml"].map(name => <option key={name}>{name}</option>))}
        {select("Precision", precision, setPrecision, ["auto", "fp32", "fp16"].map(name => <option key={name}>{name}</option>))}
        <label>Transpose (semitones)<input type="number" min={-12} max={12} step={1} required value={transpose} onChange={event => setTranspose(event.target.valueAsNumber)}/></label>
        <label>Random seed<input type="number" min={0} max={4294967295} step={1} required value={seed} onChange={event => setSeed(event.target.valueAsNumber)}/></label>
        <label className="checkbox"><input type="checkbox" checked={keep} onChange={event => setKeep(event.target.checked)}/>Keep intermediates</label>
        <button type="button" disabled={!song} onClick={() => void queue("preparation")}>Prepare and listen to vocal sources</button>
        {prepared && <div className="source-list"><p>Select a source explicitly before conversion.</p>{prepared.sources.map(source => <label className="source" key={source.id}><span><input type="radio" name="source" checked={selected === source.id} onChange={() => setSelected(source.id)}/>{source.label}</span><audio aria-label={source.label} controls preload="none" src={artifactUrl(source.artifact)}/></label>)}<button type="button" onClick={resetPreparation}>Use original source</button></div>}
        <button disabled={!song || !reference || (!!prepared && !selected)}>Queue conversion</button>
      </fieldset></form></article>
      <aside className="panel"><h2>Voice models</h2>{models.length ? models.map(model => <div className="model" key={model.id}><strong>{model.id}</strong><span>{model.mode}</span><small>{model.status}</small></div>) : <p>No model registered. Provider defaults remain available after local asset setup.</p>}
        <h2>Compare converters</h2><p>Both configurations use the original uploaded audio and seed. Missing assets produce a not-tested result.</p>
        {select("Comparison converter", compare, setCompare, providers("svc"), active)}
        <button disabled={active || !song || !reference || !Number.isInteger(seed) || !Number.isInteger(transpose)} onClick={() => void queue("benchmark")}>Run comparison</button>
        <h3>Benchmark runs</h3>{runs.length ? runs.map(run => <button className="secondary" key={run.id} disabled={active} onClick={() => setJob(run)}>{run.id.slice(0, 12)}: {run.state}</button>) : <p>No benchmark runs yet.</p>}
      </aside></section>
    <section className="panel pipeline"><h2>Job progress</h2><p role="status">{job ? job.kind + ": " + job.state + " / " + job.stage : "No job selected"}</p><progress aria-label="Job progress" max={1} value={job?.progress ?? 0}/>
      {job && !terminal(job) && <button onClick={async () => { try { setJob(parseJob(await post("/jobs/" + encodeURIComponent(job.id) + "/cancel", {}))); } catch (error) { setError(errorText(error)); } }}>Cancel job</button>}
      {job?.error && <p className="error" role="alert">{job.error}</p>}</section>
    {job && Object.keys(job.result).length > 0 && <Results key={job.id} job={job}/>}
    <Listening jobs={job ? [...runs, job] : runs}/>
    <section className="panel results"><h2>Component capabilities</h2><p>Installed and configured do not mean model inference was verified. Compatibility is reported per provider.</p><div className="table-scroll"><table><thead><tr><th>Provider</th><th>State</th><th>Backend compatibility</th></tr></thead><tbody>{catalog.providers.map(provider => <tr key={provider.name}><th>{provider.name}<small>{provider.task} / {provider.stability}</small></th><td>{provider.installed ? "Installed" : "Not installed"}<br/>{provider.configured ? "Configured" : "Not configured"}</td><td>{Object.entries(provider.backends).map(([name, status]) => <div key={name}>{name}: {status}</div>)}{provider.warnings.map((warning, index) => <small key={index}>{warning}</small>)}</td></tr>)}</tbody></table></div></section>
    <footer>Use only voices and music you own or are authorized to process. Unmeasured metrics are never displayed as scores.</footer>
  </main>;
}
const root = document.getElementById("root");
if (!root) throw new Error("Application root is missing");
createRoot(root).render(<React.StrictMode><App/></React.StrictMode>);
