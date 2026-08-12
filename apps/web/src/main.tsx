import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Job = { id: string; state: string; stage: string; progress: number; error?: { message: string }; result?: { artifacts?: Record<string, string> } };
type Model = { model_name: string; version: string; architecture: string; evaluation_status: string };

const stages = ["Uploading", "Separating stems", "Analyzing vocal", "Extracting pitch", "Converting voice", "Post-processing", "Mixing", "Evaluating", "Complete"];

function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [song, setSong] = useState<File>();
  const [reference, setReference] = useState<File>();
  const [job, setJob] = useState<Job>();
  const [message, setMessage] = useState("Ready for authorized audio");

  useEffect(() => { fetch("/api/models").then(r => r.ok ? r.json() : []).then(setModels).catch(() => setMessage("API is offline")); }, []);
  useEffect(() => {
    if (!job || ["SUCCEEDED", "FAILED", "CANCELLED"].includes(job.state)) return;
    const timer = window.setInterval(async () => setJob(await fetch(`/api/jobs/${job.id}`).then(r => r.json())), 1000);
    return () => window.clearInterval(timer);
  }, [job]);

  async function upload(file: File) {
    const body = new FormData(); body.append("file", file);
    const response = await fetch("/api/uploads", { method: "POST", body });
    if (!response.ok) throw new Error((await response.json()).detail || "Upload failed");
    return (await response.json()).artifact_id as string;
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!song || !reference) return;
    try {
      setMessage("Uploading authorized audio…");
      const [songId, referenceId] = await Promise.all([upload(song), upload(reference)]);
      const response = await fetch("/api/conversion-jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ song_artifact_id: songId, reference_artifact_id: referenceId, output_name: song.name.replace(/\.[^.]+$/, "") }) });
      if (!response.ok) throw new Error((await response.json()).detail || "Could not queue job");
      setJob(await response.json()); setMessage("Conversion queued");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unexpected error"); }
  }

  return <main>
    <header><div><span className="eyebrow">AUTHORIZED AUDIO ML WORKSPACE</span><h1>Neural Singing<br/><em>Voice Platform</em></h1></div><div className="status"><i></i>{message}</div></header>
    <section className="grid">
      <article className="panel convert"><div className="number">01</div><h2>Convert a song</h2><p>Preserve melody and timing while conditioning the output on your registered voice.</p>
        <form onSubmit={submit}>
          <label>AUTHORIZED SONG<input type="file" accept="audio/*" onChange={e => setSong(e.target.files?.[0])}/><span>{song?.name || "Choose song"}</span></label>
          <label>YOUR VOICE REFERENCE<input type="file" accept="audio/*" onChange={e => setReference(e.target.files?.[0])}/><span>{reference?.name || "Choose dry vocal"}</span></label>
          <button disabled={!song || !reference || (!!job && !["SUCCEEDED", "FAILED", "CANCELLED"].includes(job.state))}>QUEUE CONVERSION <b>→</b></button>
        </form>
      </article>
      <article className="panel models"><div className="number">02</div><h2>Voice models</h2>{models.length ? models.map(model => <div className="model" key={`${model.model_name}-${model.version}`}><strong>{model.model_name}</strong><span>{model.version} · {model.architecture}</span><small>{model.evaluation_status}</small></div>) : <div className="empty">No verified model registered yet.<br/><code>nsvp models list</code></div>}</article>
    </section>
    <section className="panel pipeline"><div className="number">03</div><div><h2>Pipeline</h2><p>{job ? `${job.state} · ${job.stage}` : "A conversion job will expose every processing stage."}</p></div><div className="progress"><div style={{width: `${(job?.progress || 0) * 100}%`}}/></div><ol>{stages.map((stage, index) => <li className={job && index / stages.length <= job.progress ? "active" : ""} key={stage}><span>{String(index + 1).padStart(2, "0")}</span>{stage}</li>)}</ol>{job?.error && <div className="error">{job.error.message}</div>}</section>
    {job?.result?.artifacts && <section className="panel results"><div className="number">04</div><h2>Results</h2><div className="artifacts">{Object.entries(job.result.artifacts).map(([name, id]) => <a href={`/api/artifacts/${id}`} key={id}>{name}<span>DOWNLOAD</span></a>)}</div></section>}
    <footer>Use only voices and music you own or are authorized to process. Metrics are reported only when measured.</footer>
  </main>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App/></React.StrictMode>);

