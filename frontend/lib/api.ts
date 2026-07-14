export type UploadResponse = {
  job_id: string;
  status: string;
};

export type PredictResponse = {
  job_id: string;
  status: string;
  cloud_mask_url: string | null;
  download_url: string | null;
  metrics: Record<string, number>;
  confidence_score: number | null;
};

export type JobResponse = {
  id: string;
  status: string;
  input_uri: string;
  mask_uri: string | null;
  output_uri: string | null;
  metrics: Record<string, number>;
  confidence_score: number | null;
  created_at: string;
  updated_at: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function absoluteApiUrl(path: string) {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function predictJob(jobId: string): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function fetchHistory(): Promise<JobResponse[]> {
  const response = await fetch(`${API_BASE}/history`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
