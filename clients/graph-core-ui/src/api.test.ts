import { afterEach, describe, expect, it, vi } from "vitest";
import { Api, waitForJob } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});
const response = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status });

describe("REST client", () => {
  it("uses the proxy and bearer token, and serializes request bodies", async () => {
    const fetch = vi.fn().mockResolvedValue(response({ id: "created" }));
    vi.stubGlobal("fetch", fetch);
    await expect(
      new Api("scoped-token").request("/collections/", { name: "notes" }),
    ).resolves.toEqual({ id: "created" });
    expect(fetch).toHaveBeenCalledWith(
      "/api/collections/",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Authorization: "Bearer scoped-token",
        }),
        body: '{"name":"notes"}',
      }),
    );
  });
  it("surfaces server validation and proxy failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(response({ detail: "Token expired" }, 401))
        .mockResolvedValueOnce(new Response("Bad Gateway", { status: 502 })),
    );
    await expect(new Api("token").request("/collections/")).rejects.toThrow(
      "Token expired",
    );
    await expect(new Api("token").request("/collections/")).rejects.toThrow(
      "502",
    );
  });
  it("polls a job through completion", async () => {
    vi.useFakeTimers();
    const request = vi
      .spyOn(Api.prototype, "request")
      .mockResolvedValueOnce({ id: "job", status: "running" })
      .mockResolvedValueOnce({ id: "job", status: "completed" });
    const progress = vi.fn();
    const pending = waitForJob(
      new Api("token"),
      "job",
      new AbortController().signal,
      progress,
    );
    await vi.advanceTimersByTimeAsync(1500);
    await expect(pending).resolves.toMatchObject({ status: "completed" });
    expect(progress).toHaveBeenCalledTimes(2);
    request.mockRestore();
  });
  it("reports worker failures and stops polling when aborted", async () => {
    const request = vi
      .spyOn(Api.prototype, "request")
      .mockResolvedValueOnce({
        id: "job",
        status: "failed",
        error: "Provider unavailable",
      })
      .mockResolvedValue({ id: "job", status: "running" });
    await expect(
      waitForJob(
        new Api("token"),
        "job",
        new AbortController().signal,
        vi.fn(),
      ),
    ).rejects.toThrow("Provider unavailable");
    const controller = new AbortController();
    const pending = waitForJob(
      new Api("token"),
      "job",
      controller.signal,
      vi.fn(),
    );
    const assertion = expect(pending).rejects.toMatchObject({
      name: "AbortError",
    });
    await Promise.resolve();
    controller.abort();
    await assertion;
    request.mockRestore();
  });
});
