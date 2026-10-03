import express from "express";
import multer from "multer";
import { spawn } from "node:child_process";
import { createReadStream, mkdirSync, unlinkSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const uploadDir = path.join(here, "uploads");
mkdirSync(uploadDir, { recursive: true });

const app = express();
const upload = multer({
  dest: uploadDir,
  limits: { fileSize: 800 * 1024 * 1024 },
  fileFilter: (_req, file, done) => {
    if (file.mimetype.startsWith("video/")) done(null, true);
    else done(new Error("Choose a video file."));
  },
});

let activeJob = null;
let latestFrame = null;
let latestMetrics = null;
let latestError = null;
const eventClients = new Set();
const streamClients = new Set();

function sendEvent(event, data) {
  const message = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const client of eventClients) client.write(message);
}

function cleanupJob(job) {
  if (job?.uploadPath) {
    try { unlinkSync(job.uploadPath); } catch {}
  }
  if (activeJob === job) activeJob = null;
}

app.get("/api/status", (_req, res) => {
  res.json({ running: Boolean(activeJob), metrics: latestMetrics, error: latestError });
});

app.get("/api/events", (req, res) => {
  res.set({ "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
  res.flushHeaders();
  eventClients.add(res);
  if (latestMetrics) res.write(`event: metrics\ndata: ${JSON.stringify(latestMetrics)}\n\n`);
  req.on("close", () => eventClients.delete(res));
});

app.get("/stream", (req, res) => {
  res.writeHead(200, {
    "Content-Type": "multipart/x-mixed-replace; boundary=frame",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    Connection: "keep-alive",
    Pragma: "no-cache",
  });
  const client = { res, closed: false };
  const push = (jpeg) => {
    if (!client.closed && !res.destroyed) {
      res.write(`--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.length}\r\n\r\n`);
      res.write(jpeg);
      res.write("\r\n");
    }
  };
  client.push = push;
  if (latestFrame) push(latestFrame);
  streamClients.add(client);
  req.on("close", () => { client.closed = true; streamClients.delete(client); });
});

app.post("/api/upload", upload.single("video"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Select a video first." });
  if (activeJob) {
    unlinkSync(req.file.path);
    return res.status(409).json({ error: "A video is already processing." });
  }

  latestFrame = null;
  latestMetrics = null;
  latestError = null;
  const job = { uploadPath: req.file.path };
  activeJob = job;
  const pythonCommand = process.env.PYTHON_COMMAND || (process.platform === "win32" ? "python" : "python3");
  const importCheck = spawn(pythonCommand, ["-c", "import cv2, numpy, torch; import ultralytics"], { windowsHide: true });
  let importError = "";
  importCheck.stderr.setEncoding("utf8");
  importCheck.stderr.on("data", (text) => { importError += text; });
  importCheck.on("error", (error) => {
    latestError = `Could not start Python (${pythonCommand}): ${error.message}. Set PYTHON_COMMAND to your Python executable.`;
    sendEvent("error", { error: latestError });
    if (!res.headersSent) res.status(500).json({ error: latestError });
    cleanupJob(job);
  });
  importCheck.on("close", (checkCode) => {
    if (checkCode !== 0) {
      latestError = `Python dependencies are missing for ${pythonCommand}. Install them with: python -m pip install -r requirements.txt. On Windows, if pip reports WinError 206, use the short-path steps in README.md. Details: ${importError.trim().split("\n").slice(-1)[0] || "import failed"}`;
      sendEvent("error", { error: latestError });
      if (!res.headersSent) res.status(500).json({ error: latestError });
      sendEvent("complete", { error: latestError });
      cleanupJob(job);
      return;
    }

  const worker = spawn(pythonCommand, [path.join(here, "traffic_congestion.py"), "--video", req.file.path, "--stream"], {
    cwd: here,
    windowsHide: true,
  });
  job.worker = worker;
  let lineBuffer = "";

  worker.stdout.setEncoding("utf8");
  worker.stdout.on("data", (text) => {
    lineBuffer += text;
    const lines = lineBuffer.split(/\r?\n/);
    lineBuffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("FRAME ")) continue;
      try {
        const packet = JSON.parse(line.slice(6));
        latestFrame = Buffer.from(packet.image, "base64");
        latestMetrics = packet.metrics;
        for (const client of streamClients) client.push(latestFrame);
        sendEvent("metrics", latestMetrics);
      } catch (error) {
        console.error("Could not read a video frame:", error);
      }
    }
  });
  worker.stderr.setEncoding("utf8");
  worker.stderr.on("data", (text) => console.error("Python detector:", text.trim()));
  worker.on("error", (error) => {
    latestError = `Could not start Python: ${error.message}`;
    sendEvent("error", { error: latestError });
    cleanupJob(job);
  });
  worker.on("close", (code) => {
    if (code && !latestError) latestError = `Detector exited with code ${code}. Check Python dependencies and model file.`;
    sendEvent("complete", { error: latestError });
    cleanupJob(job);
  });

  res.json({ message: "Video uploaded. Processing has started." });
  });
});

app.post("/api/stop", (_req, res) => {
  if (activeJob?.worker) activeJob.worker.kill();
  latestFrame = null;
  res.json({ message: "Processing stopped." });
});

app.use((error, _req, res, _next) => {
  const status = error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE" ? 413 : 400;
  res.status(status).json({ error: error.message || "Upload failed." });
});

const port = Number(process.env.PORT || 3001);
app.listen(port, () => console.log(`Traffic analysis server listening at http://localhost:${port}`));
