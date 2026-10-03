import { useEffect, useRef, useState } from "react";

const emptyMetrics = {
  totalVehicles: 0, cars: 0, buses: 0, trucks: 0, motorcycles: 0,
  averageSpeed: null, occupancy: 0, slowVehicles: 0, trafficStatus: "WAITING",
};

function Metric({ label, value, unit = "" }) {
  return <div className="metric"><span className="metric-label">{label}</span><strong>{value ?? "N/A"}<small>{value == null ? "" : unit}</small></strong></div>;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [metrics, setMetrics] = useState(emptyMetrics);
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    const events = new EventSource("/api/events");
    events.addEventListener("metrics", (event) => setMetrics(JSON.parse(event.data)));
    events.addEventListener("error", (event) => {
      if (event.data) setError(JSON.parse(event.data).error);
    });
    events.addEventListener("complete", (event) => {
      setRunning(false);
      setBusy(false);
      const data = JSON.parse(event.data);
      if (data.error) setError(data.error);
    });
    return () => events.close();
  }, []);

  async function startAnalysis(event) {
    event.preventDefault();
    if (!file) return setError("Choose a video file to start.");
    setError("");
    setBusy(true);
    setMetrics(emptyMetrics);
    const body = new FormData();
    body.append("video", file);
    try {
      const response = await fetch("/api/upload", { method: "POST", body });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Upload failed.");
      setRunning(true);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  async function stopAnalysis() {
    await fetch("/api/stop", { method: "POST" });
    setRunning(false);
    setBusy(false);
  }

  const statusClass = metrics.trafficStatus.toLowerCase();

  return <main className="shell">
    <header className="topbar">
      <a className="brand" href="#top"><span className="brand-mark">T</span><span>TRAFFIC<span className="brand-light">LENS</span></span></a>
      <div className="top-meta"><span className={`live-dot ${running ? "active" : ""}`} /> LOCAL ANALYSIS <span className="meta-divider">/</span> YOLO TRACKING</div>
      <span className="version">ACADEMY PROJECT <b>01</b></span>
    </header>

    <section className="intro" id="top">
      <div><div className="eyebrow"><span>01</span> INTELLIGENT TRANSPORT SYSTEMS</div>
        <h1>See the flow.<br /><em>Understand</em> the road.</h1>
        <p className="intro-copy">Upload a traffic video to detect and track vehicles, estimate speed, and read congestion conditions as the footage plays.</p>
      </div>
      <div className="intro-index"><span>VIDEO ANALYSIS</span><b>01 <i>/</i> 04</b><div className="index-line"><span /></div><small>CAR · BUS · TRUCK · MOTORCYCLE</small></div>
    </section>

    <section className="workspace">
      <div className="workspace-head"><div><span className="section-no">01</span><h2>Live monitoring</h2></div><span className="processing-state"><i className={running ? "on" : ""} />{running ? "ANALYSIS RUNNING" : "AWAITING VIDEO"}</span></div>
      <div className="monitor-grid">
        <div className="video-panel">
          <div className="panel-top"><span><i className="tiny-square" /> CAMERA FEED <span className="panel-sep">/</span> PROCESSED OUTPUT</span><span className="panel-time">{running ? "LIVE" : "LOCAL"}</span></div>
          <div className="video-stage">
            {running ? <img className="stream-image" src="/stream" alt="Live YOLO processed video" /> : <div className="empty-state"><div className="target-icon"><span /><i /><b /></div><strong>Ready when you are.</strong><p>Select a video file to begin detection and tracking.</p><span className="empty-resolution">PROCESSED VIDEO WILL APPEAR HERE</span></div>}
            <div className="corner corner-tl"/><div className="corner corner-tr"/><div className="corner corner-bl"/><div className="corner corner-br"/>
            {running && <div className="feed-badge"><i /> LIVE DETECTION</div>}
          </div>
          <div className="video-bottom"><span><i className="green-dot" /> YOLO · BYTETRACK</span><span>ANNOTATED FRAME STREAM</span></div>
        </div>

        <aside className="metrics-panel">
          <div className="metrics-heading"><div><span className="eyebrow">CURRENT FRAME</span><h3>Traffic metrics</h3></div><span className="metrics-symbol">↗</span></div>
          <div className="status-card"><span>TRAFFIC CONDITION</span><strong className={`status-${statusClass}`}>{metrics.trafficStatus}</strong><div className="status-track"><i className={statusClass}/></div></div>
          <div className="metric-grid">
            <Metric label="UNIQUE VEHICLES" value={metrics.totalVehicles} />
            <Metric label="EST. AVG. SPEED" value={metrics.averageSpeed} unit={metrics.averageSpeed == null ? "" : " km/h"} />
            <Metric label="ROAD OCCUPANCY" value={metrics.occupancy} unit="%" />
            <Metric label="SLOW VEHICLES" value={metrics.slowVehicles} />
          </div>
          <div className="class-title"><span>VEHICLES BY CLASS</span><span>UNIQUE IDS</span></div>
          <div className="class-list"><div><span><i className="class-dot car"/>Cars</span><b>{metrics.cars}</b></div><div><span><i className="class-dot bus"/>Buses</span><b>{metrics.buses}</b></div><div><span><i className="class-dot truck"/>Trucks</span><b>{metrics.trucks}</b></div><div><span><i className="class-dot moto"/>Motorcycles</span><b>{metrics.motorcycles}</b></div></div>
          <div className="metrics-foot"><span>VALUES UPDATE WITH EACH PROCESSED FRAME</span><span>ESTIMATES ONLY</span></div>
        </aside>
      </div>
    </section>

    <section className="upload-section">
      <div className="upload-copy"><span className="eyebrow"><span>02</span> INPUT SOURCE</span><h2>Bring the footage.</h2><p>Choose a video file from your device. Detection begins after upload and frames stream back as they are processed.</p><span className="file-note">VIDEO FILES · UP TO 800 MB</span></div>
      <form className="upload-card" onSubmit={startAnalysis}>
        <input ref={inputRef} type="file" accept="video/*" hidden onChange={(event) => { setFile(event.target.files?.[0] || null); setError(""); }} />
        <button type="button" className="file-pick" onClick={() => inputRef.current?.click()}><span className="upload-icon">↑</span><span className="pick-text"><b>{file ? file.name : "Choose a video"}</b><small>{file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB · READY TO PROCESS` : "MP4, MOV, AVI · BROWSE FILES"}</small></span><span className="browse">BROWSE <i>↗</i></span></button>
        <div className="upload-actions"><span>{error || (busy ? "Starting detector…" : "Files are processed locally on this machine.")}</span>{running ? <button type="button" className="start-button stop-button" onClick={stopAnalysis}>STOP ANALYSIS <i>■</i></button> : <button type="submit" className="start-button" disabled={!file || busy}>START ANALYSIS <i>↗</i></button>}</div>
      </form>
    </section>
    <footer><span>TRAFFIC LENS <i>·</i> ACADEMY PROJECT</span><span>SPEED AND OCCUPANCY ARE APPROXIMATE ESTIMATES</span><a href="#top">BACK TO TOP ↑</a></footer>
  </main>;
}
