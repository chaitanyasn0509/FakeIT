"use client";

import { Activity, Cloud, Download, History, UploadCloud, Wand2 } from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useState } from "react";

import { BeforeAfterSlider } from "@/components/BeforeAfterSlider";
import { MetricGrid } from "@/components/MetricGrid";
import { absoluteApiUrl, fetchHistory, JobResponse, predictJob, PredictResponse, uploadImage } from "@/lib/api";
import { renderGeoTiffFile, renderGeoTiffUrl } from "@/lib/geotiffPreview";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [beforePreview, setBeforePreview] = useState<string | null>(null);
  const [afterPreview, setAfterPreview] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [history, setHistory] = useState<JobResponse[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const downloadHref = useMemo(() => {
    return prediction?.download_url ? absoluteApiUrl(prediction.download_url) : null;
  }, [prediction]);

  useEffect(() => {
    fetchHistory().then(setHistory).catch(() => setHistory([]));
  }, []);

  async function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setPrediction(null);
    setAfterPreview(null);
    setJobId(null);
    setError(null);
    if (!selected) {
      setBeforePreview(null);
      return;
    }
    try {
      setBeforePreview(await renderGeoTiffFile(selected));
    } catch {
      setBeforePreview(URL.createObjectURL(selected));
    }
  }

  async function onUpload() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const uploaded = await uploadImage(file);
      setJobId(uploaded.job_id);
      setHistory(await fetchHistory());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function onPredict() {
    if (!jobId) return;
    setBusy(true);
    setError(null);
    try {
      const result = await predictJob(jobId);
      setPrediction(result);
      if (result.download_url) {
        setAfterPreview(await renderGeoTiffUrl(absoluteApiUrl(result.download_url)));
      }
      setHistory(await fetchHistory());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="workspace">
      <aside className="rail">
        <div className="brand">
          <Cloud size={22} />
          <span>UNCLOUD IT</span>
        </div>
        <label className="upload-target">
          <UploadCloud size={20} />
          <span>{file ? file.name : "GeoTIFF"}</span>
          <input accept=".tif,.tiff,image/tiff" type="file" onChange={onFileChange} />
        </label>
        <button className="primary" disabled={!file || busy} onClick={onUpload}>
          <UploadCloud size={18} />
          <span>Upload</span>
        </button>
        <button className="primary accent" disabled={!jobId || busy} onClick={onPredict}>
          <Wand2 size={18} />
          <span>Reconstruct</span>
        </button>
        {downloadHref && (
          <a className="download" href={downloadHref}>
            <Download size={18} />
            <span>GeoTIFF</span>
          </a>
        )}
        {error && <p className="error">{error}</p>}
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">LISS-IV cloud removal</p>
            <h1>Surface Reconstruction Console</h1>
          </div>
          <div className="status">
            <Activity size={17} />
            <span>{busy ? "Running" : prediction?.status ?? "Ready"}</span>
          </div>
        </header>

        <BeforeAfterSlider beforeSrc={beforePreview} afterSrc={afterPreview} />
        {prediction && <MetricGrid metrics={prediction.metrics} confidence={prediction.confidence_score} />}
      </section>

      <aside className="history">
        <div className="history-title">
          <History size={18} />
          <span>History</span>
        </div>
        <div className="history-list">
          {history.slice(0, 8).map((job) => (
            <button className="history-row" key={job.id} onClick={() => setJobId(job.id)}>
              <span>{job.id.slice(0, 8)}</span>
              <strong>{job.status}</strong>
            </button>
          ))}
        </div>
      </aside>
    </main>
  );
}
