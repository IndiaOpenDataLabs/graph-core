import { afterEach, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

const namespace = {
  id: "12345678-1234-1234-1234-123456789abc",
  name: "Research",
};
const collection = {
  id: "collection-one",
  name: "Notes",
  strategy: "custom_graph_rag",
  default_query_mode: "mix",
  gleaning_passes: 1,
};
const result = {
  response: "Graph-backed answer",
  retrieval_context: "Entities:\nAda (person): a mathematician",
  entities_used: ["Ada"],
  relationships_used: ["Ada -> wrote -> Notes"],
  mode: "mix",
};
const calls: {
  path: string;
  token: string | undefined;
  body: Record<string, unknown> | undefined;
}[] = [];
function mockApi() {
  calls.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init: RequestInit) => {
      calls.push({
        path,
        token: (init.headers as Record<string, string>).Authorization,
        body: init.body ? JSON.parse(String(init.body)) : undefined,
      });
      let data: unknown;
      if (path === "/api/platform/namespaces/") data = [namespace];
      else if (path === "/api/platform/namespaces/me") data = namespace;
      else if (path.endsWith("/issue-user-token"))
        data = {
          token: "namespace-token",
          namespace_id: namespace.id,
          namespace_name: namespace.name,
        };
      else if (path === "/api/collections/")
        data =
          init.method === "POST"
            ? { ...collection, name: JSON.parse(String(init.body)).name }
            : [collection];
      else if (path === "/api/platform/capabilities")
        data = {
          embedding_profiles: [],
          llm_profiles: [],
          retrieval_strategies: ["vector", "light_rag", "custom_graph_rag"],
          max_chunk_size: 5000,
        };
      else if (path === "/api/jobs/?limit=30") data = [];
      else if (path.endsWith("/query") || path.endsWith("/ingest/doc"))
        data = { job_id: "job-one", status: "queued" };
      else if (path === "/api/jobs/job-one")
        data = { id: "job-one", status: "completed", progress_percent: 100 };
      else if (path.endsWith("/result")) data = { result };
      else if (path === "/api/platform/profiles")
        data = { profile_id: "profile-one" };
      else throw new Error(`Unexpected request: ${path}`);
      return new Response(JSON.stringify(data), { status: 200 });
    }),
  );
}
afterEach(() => vi.unstubAllGlobals());
async function connectUser() {
  fireEvent.change(screen.getByLabelText("Token type"), {
    target: { value: "user" },
  });
  fireEvent.change(screen.getByLabelText("JWT bearer token"), {
    target: { value: "user-token" },
  });
  fireEvent.change(screen.getByLabelText("Namespace ID"), {
    target: { value: namespace.id },
  });
  fireEvent.click(screen.getByRole("button", { name: "Connect" }));
  await screen.findByLabelText("Your question");
}

it("selects a namespace as admin, queries with its scoped token, and inspects retrieved content", async () => {
  mockApi();
  render(<App />);
  fireEvent.change(screen.getByLabelText("JWT bearer token"), {
    target: { value: "admin-token" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Connect" }));
  fireEvent.click(
    await screen.findByRole("button", { name: "Open workspace →" }),
  );
  fireEvent.change(await screen.findByLabelText("Your question"), {
    target: { value: "Who is Ada?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Run query →" }));
  expect(await screen.findByText("Graph-backed answer")).toBeInTheDocument();
  expect(
    screen.getByText("Entities: Ada (person): a mathematician"),
  ).toBeInTheDocument();
  expect(calls.find((c) => c.path.endsWith("/query"))).toMatchObject({
    token: "Bearer namespace-token",
    body: { question: "Who is Ada?", mode: null, llm_profile_id: null },
  });
  expect(localStorage.length).toBe(0);
  fireEvent.click(screen.getByRole("button", { name: "Switch namespace" }));
  expect(screen.queryByText("Graph-backed answer")).not.toBeInTheDocument();
  expect(
    await screen.findByRole("button", { name: "Open workspace →" }),
  ).toBeInTheDocument();
});

it("creates model profiles, collections, and ingests documents using the user token", async () => {
  mockApi();
  render(<App />);
  await connectUser();
  fireEvent.click(screen.getByRole("button", { name: "Model profiles" }));
  fireEvent.change(screen.getAllByLabelText("Provider")[0], {
    target: { value: "openai" },
  });
  fireEvent.change(screen.getByLabelText("Model"), {
    target: { value: "text-embedding-3-small" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Create profile" }));
  expect(await screen.findByText("Profile created.")).toBeInTheDocument();
  expect(calls.find((c) => c.path === "/api/platform/profiles")).toMatchObject({
    token: "Bearer user-token",
    body: { kind: "embedding", dimensions: 1536, provider: "openai" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Collections" }));
  fireEvent.change(screen.getByLabelText("Name"), {
    target: { value: "New notes" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Create collection" }));
  await screen.findByText("Created New notes. Add data next.");
  fireEvent.click(screen.getByRole("button", { name: "Add data" }));
  fireEvent.change(screen.getByLabelText("Content"), {
    target: { value: "Ada wrote notes." },
  });
  fireEvent.click(screen.getByRole("button", { name: "Add to collection →" }));
  await screen.findByText("Document ingestion completed.");
  expect(calls.find((c) => c.path.endsWith("/ingest/doc"))).toMatchObject({
    body: { text: "Ada wrote notes.", domain: null, document_path: null },
  });
});

it("asks only for an admin JWT and ignores namespace input left over from user mode", async () => {
  mockApi();
  render(<App />);
  expect(screen.queryByLabelText("Namespace ID")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Token type"), {
    target: { value: "user" },
  });
  fireEvent.change(screen.getByLabelText("Namespace ID"), {
    target: { value: "invalid-id" },
  });
  fireEvent.change(screen.getByLabelText("Token type"), {
    target: { value: "admin" },
  });
  expect(screen.queryByLabelText("Namespace ID")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("JWT bearer token"), {
    target: { value: "admin-token" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Connect" }));
  await screen.findByRole("button", { name: "Open workspace →" });
  expect(
    calls.find((c) => c.path === "/api/platform/namespaces/"),
  ).toMatchObject({ token: "Bearer admin-token" });
  expect(calls.some((c) => c.path.endsWith("/issue-user-token"))).toBe(false);
});

it("requires a namespace ID for user connections", () => {
  mockApi();
  render(<App />);
  fireEvent.change(screen.getByLabelText("Token type"), {
    target: { value: "user" },
  });
  fireEvent.change(screen.getByLabelText("JWT bearer token"), {
    target: { value: "user-token" },
  });
  expect(screen.getByLabelText("Namespace ID")).toBeRequired();
  expect(screen.getByRole("button", { name: "Connect" })).toBeDisabled();
  expect(calls).toHaveLength(0);
});

it("rejects a user token belonging to another namespace", async () => {
  mockApi();
  render(<App />);
  fireEvent.change(screen.getByLabelText("Token type"), {
    target: { value: "user" },
  });
  fireEvent.change(screen.getByLabelText("JWT bearer token"), {
    target: { value: "user-token" },
  });
  fireEvent.change(screen.getByLabelText("Namespace ID"), {
    target: { value: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Connect" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Namespace ID does not match this user token's namespace.",
  );
  expect(screen.queryByLabelText("Your question")).not.toBeInTheDocument();
  expect(calls.some((c) => c.path === "/api/collections/")).toBe(false);
});

it("renders API errors without connecting to a namespace", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Invalid JWT bearer token" }), {
        status: 401,
      }),
    ),
  );
  render(<App />);
  fireEvent.change(screen.getByLabelText("JWT bearer token"), {
    target: { value: "invalid" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Connect" }));
  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Invalid JWT bearer token",
    ),
  );
  expect(screen.getByLabelText("JWT bearer token")).toBeInTheDocument();
});
