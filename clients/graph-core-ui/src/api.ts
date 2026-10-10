export interface Namespace {
  id: string;
  name: string;
  created_at?: string | null;
}
export interface Profile {
  profile_id: string;
  kind: "embedding" | "llm";
  provider: string;
  model: string;
  label: string | null;
  dimensions: number | null;
}
export interface Collection {
  id: string;
  name: string;
  strategy: string;
  embedding_profile_id: string | null;
  llm_profile_id: string | null;
  default_query_mode: string | null;
  gleaning_passes: number;
}
export interface Capabilities {
  embedding_profiles: Profile[];
  llm_profiles: Profile[];
  retrieval_strategies: string[];
  max_chunk_size: number;
}
export interface QueryResult {
  response: string;
  entities_used: string[];
  relationships_used: string[];
  mode?: string | null;
  retrieval_context?: string;
}
export interface Job {
  id: string;
  type: string;
  status: string;
  progress_percent?: number;
  chunks_total?: number;
  chunks_completed?: number;
  error?: string | null;
  collection_id?: string;
  document_path?: string | null;
  payload?: { question?: string; result?: QueryResult };
}

export class Api {
  constructor(private readonly token: string) {}

  async request<T>(
    path: string,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<T> {
    const response = await fetch(`/api${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        Authorization: `Bearer ${this.token}`,
        "Content-Type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
    const text = await response.text();
    let data: unknown;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = null;
    }
    if (!response.ok) {
      const detail = (data as { detail?: unknown } | null)?.detail;
      throw new Error(
        typeof detail === "string"
          ? detail
          : detail
            ? JSON.stringify(detail)
            : `API request failed (${response.status}). Check that Graph Core is running.`,
      );
    }
    return data as T;
  }
}

export const terminalStatus = (status: string) =>
  ["completed", "failed", "cancelled"].includes(status);

export async function waitForJob(
  api: Api,
  id: string,
  signal: AbortSignal,
  onProgress: (job: Job) => void,
): Promise<Job> {
  while (!signal.aborted) {
    const job = await api.request<Job>(`/jobs/${id}`, undefined, signal);
    onProgress(job);
    if (terminalStatus(job.status)) {
      if (job.status !== "completed")
        throw new Error(job.error || `Job ${job.status}`);
      return job;
    }
    await new Promise<void>((resolve, reject) => {
      const abort = () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      };
      const timer = setTimeout(() => {
        signal.removeEventListener("abort", abort);
        resolve();
      }, 1500);
      signal.addEventListener("abort", abort, { once: true });
      if (signal.aborted) abort();
    });
  }
  throw new DOMException("Aborted", "AbortError");
}
