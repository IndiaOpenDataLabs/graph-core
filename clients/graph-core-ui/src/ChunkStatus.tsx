import { useEffect, useRef, useState } from "react";
import { Api } from "./api";
import "./ChunkStatus.css";

interface Chunk {
  id: string;
  chunk_index: number;
  status: string;
  error: string | null;
  processing_started_at: string | null;
  completed_at: string | null;
}
interface Snapshot {
  job_status: string;
  counts: Record<string, number>;
  total: number;
  filtered_total: number;
  chunks: Chunk[];
}

export function ChunkStatus({
  api,
  jobId,
  onClose,
  onRetried,
}: {
  api: Api;
  jobId: string;
  onClose: () => void;
  onRetried: () => Promise<void>;
}) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [offset, setOffset] = useState(0);
  const [failedOnly, setFailedOnly] = useState(false);
  const [revision, setRevision] = useState(0);
  const [retrying, setRetrying] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const action = useRef<AbortController | null>(null);
  const pageSize = 50;

  useEffect(() => () => action.current?.abort(), []);
  useEffect(() => {
    const controller = new AbortController();
    let inFlight = false;
    async function load() {
      if (inFlight) return;
      inFlight = true;
      try {
        const state = await api.request<Snapshot>(
          `/jobs/${jobId}/chunks?offset=${offset}&limit=${pageSize}${failedOnly ? "&status=failed" : ""}`,
          undefined,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setSnapshot(state);
          setError("");
        }
      } catch (e) {
        if (!controller.signal.aborted)
          setError(e instanceof Error ? e.message : String(e));
      } finally {
        inFlight = false;
      }
    }
    void load();
    const timer = setInterval(() => void load(), 3000);
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, [api, jobId, offset, failedOnly, revision]);

  async function retry(index?: number) {
    if (retrying) return;
    const controller = new AbortController();
    action.current = controller;
    setRetrying(true);
    setError("");
    setNotice("");
    try {
      const result = await api.request<{ retried_chunks: number[] }>(
        `/jobs/${jobId}/retry-failed-chunks`,
        index === undefined ? {} : { chunk_indices: [index] },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setNotice(
        result.retried_chunks.length
          ? `Queued ${result.retried_chunks.length} failed chunk(s) for retry. Successful chunks are unchanged.`
          : "Scheduled the pending chunks.",
      );
      await onRetried();
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (!controller.signal.aborted) {
        setRetrying(false);
        setRevision((value) => value + 1);
      }
    }
  }

  const canRetry =
    snapshot?.job_status === "failed" &&
    snapshot.counts.failed > 0 &&
    !snapshot.counts.processing &&
    !snapshot.counts.pending;
  const canResume =
    snapshot?.job_status === "running" &&
    snapshot.counts.pending > 0 &&
    !snapshot.counts.processing;
  return (
    <section className="chunk-status" aria-label="Ingestion chunk status">
      <div className="section-heading">
        <h3>Ingestion chunks</h3>
        <button className="secondary" onClick={onClose}>
          Close
        </button>
      </div>
      <small>{jobId}</small>
      <p className="muted">
        Refreshes every 3 seconds. Retry is enabled after ingestion finishes. It
        reuses saved chunk text; successful chunks are not reprocessed.
      </p>
      {error && (
        <p role="alert" className="error-text">
          {error}
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
      {snapshot ? (
        <>
          <div className="chunk-counts">
            {Object.entries(snapshot.counts).map(([status, count]) => (
              <span
                key={status}
                className={`badge ${status === "failed" ? "failed" : ""}`}
              >
                {status}: {count}
              </span>
            ))}
          </div>
          <div className="chunk-controls">
            <label>
              <input
                type="checkbox"
                checked={failedOnly}
                disabled={retrying}
                onChange={(event) => {
                  setFailedOnly(event.target.checked);
                  setOffset(0);
                }}
              />{" "}
              Failed only
            </label>
            <button
              disabled={retrying || !canRetry}
              onClick={() => void retry()}
            >
              {retrying ? "Scheduling…" : "Retry all failed chunks"}
            </button>
            {canResume && (
              <button
                className="secondary"
                disabled={retrying}
                onClick={() => void retry()}
              >
                Resume pending chunks
              </button>
            )}
          </div>
          {snapshot.total === 0 ? (
            <p className="muted">
              No per-chunk records. Chunk inspection is available for graph
              document ingestion jobs.
            </p>
          ) : (
            <>
              <div className="chunk-table">
                <table>
                  <thead>
                    <tr>
                      <th>Chunk</th>
                      <th>Status</th>
                      <th>Details</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshot.chunks.map((chunk) => (
                      <tr key={chunk.id}>
                        <td>
                          {chunk.chunk_index + 1}
                          <small>index {chunk.chunk_index}</small>
                        </td>
                        <td>
                          <span
                            className={`badge ${chunk.status === "failed" ? "failed" : ""}`}
                          >
                            {chunk.status}
                          </span>
                        </td>
                        <td>
                          {chunk.error ? (
                            <details>
                              <summary>Error</summary>
                              <pre className="chunk-error">{chunk.error}</pre>
                            </details>
                          ) : chunk.completed_at ? (
                            <small>
                              Finished{" "}
                              {new Date(chunk.completed_at).toLocaleString()}
                            </small>
                          ) : chunk.processing_started_at ? (
                            <small>
                              Started{" "}
                              {new Date(
                                chunk.processing_started_at,
                              ).toLocaleString()}
                            </small>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td>
                          {chunk.status === "failed" && (
                            <button
                              className="secondary"
                              disabled={retrying || !canRetry}
                              onClick={() => void retry(chunk.chunk_index)}
                            >
                              Retry chunk
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {!snapshot.chunks.length && (
                <p className="muted">No chunks match this filter.</p>
              )}
              <div className="chunk-controls">
                <button
                  className="secondary"
                  disabled={retrying || offset === 0}
                  onClick={() =>
                    setOffset((value) => Math.max(0, value - pageSize))
                  }
                >
                  Previous
                </button>
                <span>
                  {snapshot.filtered_total ? offset + 1 : 0}–
                  {Math.min(
                    offset + snapshot.chunks.length,
                    snapshot.filtered_total,
                  )}{" "}
                  of {snapshot.filtered_total}
                </span>
                <button
                  className="secondary"
                  disabled={
                    retrying || offset + pageSize >= snapshot.filtered_total
                  }
                  onClick={() => setOffset((value) => value + pageSize)}
                >
                  Next
                </button>
              </div>
            </>
          )}
        </>
      ) : (
        <p className="muted">Loading chunks…</p>
      )}
    </section>
  );
}
