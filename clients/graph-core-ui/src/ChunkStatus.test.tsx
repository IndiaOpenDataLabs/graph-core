import { afterEach, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Api } from "./api";
import { ChunkStatus } from "./ChunkStatus";

const snapshot = {
  job_status: "failed",
  counts: { completed: 1, failed: 1, pending: 0, processing: 0, cancelled: 0 },
  total: 2,
  filtered_total: 2,
  chunks: [
    {
      id: "success",
      chunk_index: 0,
      status: "completed",
      error: null,
      completed_at: "2026-01-01T00:00:00Z",
      processing_started_at: null,
    },
    {
      id: "failure",
      chunk_index: 1,
      status: "failed",
      error: "Extraction failed",
      completed_at: null,
      processing_started_at: null,
    },
  ],
};

afterEach(() => vi.unstubAllGlobals());

function setup(state = snapshot, failRetry = false) {
  const requests: { path: string; body: unknown; token: string }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init: RequestInit) => {
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      requests.push({
        path,
        body,
        token: (init.headers as Record<string, string>).Authorization,
      });
      if (init.method === "POST") {
        return new Response(
          JSON.stringify(
            failRetry
              ? { detail: "Wait for ingestion to finish" }
              : { retried_chunks: [1] },
          ),
          { status: failRetry ? 409 : 202 },
        );
      }
      return new Response(JSON.stringify(state), { status: 200 });
    }),
  );
  const onRetried = vi.fn(async () => {});
  const onClose = vi.fn();
  render(
    <ChunkStatus
      api={new Api("user-token")}
      jobId="job-one"
      onRetried={onRetried}
      onClose={onClose}
    />,
  );
  return { requests, onRetried, onClose };
}

it.each(["one", "all"])(
  "inspects errors and schedules %s failed-chunk retry without reprocessing successes",
  async (mode) => {
    const { requests, onRetried } = setup();
    await screen.findByText("Extraction failed");
    fireEvent.click(
      screen.getByRole("button", {
        name: mode === "one" ? "Retry chunk" : "Retry all failed chunks",
      }),
    );
    await screen.findByText(/Queued 1 failed chunk/);
    expect(requests.find((r) => r.body !== undefined)).toEqual({
      path: "/api/jobs/job-one/retry-failed-chunks",
      body: mode === "one" ? { chunk_indices: [1] } : {},
      token: "Bearer user-token",
    });
    await waitFor(() => expect(onRetried).toHaveBeenCalledOnce());
    expect(screen.getByText("completed: 1")).toBeInTheDocument();
  },
);

it("disables retries while chunks are processing", async () => {
  setup({
    ...snapshot,
    job_status: "running",
    counts: { ...snapshot.counts, processing: 1 },
  });
  await screen.findByText("Extraction failed");
  expect(
    screen.getByRole("button", { name: "Retry all failed chunks" }),
  ).toBeDisabled();
  expect(screen.getByRole("button", { name: "Retry chunk" })).toBeDisabled();
});

it("surfaces retry conflicts without claiming a retry was queued", async () => {
  const { onRetried } = setup(snapshot, true);
  await screen.findByText("Extraction failed");
  fireEvent.click(screen.getByRole("button", { name: "Retry chunk" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Wait for ingestion to finish",
  );
  expect(onRetried).not.toHaveBeenCalled();
  expect(screen.queryByText(/Queued/)).not.toBeInTheDocument();
});

it("filters failed chunks and can resume saved pending retries", async () => {
  const { requests } = setup({
    ...snapshot,
    job_status: "running",
    counts: { ...snapshot.counts, failed: 0, pending: 1 },
  });
  await screen.findByText("Extraction failed");
  fireEvent.click(screen.getByRole("checkbox", { name: "Failed only" }));
  await waitFor(() =>
    expect(requests.some((r) => r.path.includes("status=failed"))).toBe(true),
  );
  fireEvent.click(
    screen.getByRole("button", { name: "Resume pending chunks" }),
  );
  await waitFor(() =>
    expect(
      requests.some(
        (r) =>
          r.path.endsWith("retry-failed-chunks") &&
          JSON.stringify(r.body) === "{}",
      ),
    ).toBe(true),
  );
});
