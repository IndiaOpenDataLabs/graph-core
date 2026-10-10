import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import {
  Api,
  waitForJob,
  terminalStatus,
  type Namespace,
  type Collection,
  type Capabilities,
  type Profile,
  type Job,
  type QueryResult,
} from "./api";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { loadConnection, saveConnection, type Session } from "./connection";
import { ChunkStatus } from "./ChunkStatus";
const namespaceIdPattern =
  "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";
type Tab = "query" | "data" | "collections" | "profiles" | "jobs";
const titles: Record<Tab, string> = {
  query: "Query explorer",
  data: "Add data",
  collections: "Collections",
  profiles: "Model profiles",
  jobs: "Activity",
};
const value = (form: FormData, key: string) =>
  String(form.get(key) || "").trim();
const optional = (form: FormData, key: string) => value(form, key) || null;

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
function Result({ result }: { result: QueryResult }) {
  return (
    <div className="result-grid">
      <section className="panel">
        <div className="section-heading">
          <h2>Answer</h2>
          <span className="badge">{result.mode || "default"}</span>
        </div>
        <div className="prose">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            skipHtml
            components={{
              a: ({ children, href }) => (
                <a href={href} target="_blank" rel="noopener noreferrer">
                  {children}
                </a>
              ),
            }}
          >
            {result.response ||
              "No answer returned. Try adding data or a different question."}
          </ReactMarkdown>
        </div>
      </section>
      <section className="panel">
        <h2>Retrieved content</h2>
        <p className="muted">
          Content retrieved for this query, separate from the generated answer.
          Combined modes include each retrieval path.
        </p>
        {result.retrieval_context ? (
          <pre className="context">{result.retrieval_context}</pre>
        ) : (
          <Empty>
            No retrieval content was recorded. Older jobs may only contain
            identifiers.
          </Empty>
        )}
        <details>
          <summary>Entities used ({result.entities_used?.length || 0})</summary>
          <ul>
            {result.entities_used?.map((entry, i) => (
              <li key={i}>{entry}</li>
            ))}
          </ul>
        </details>
        <details>
          <summary>
            Relationships used ({result.relationships_used?.length || 0})
          </summary>
          <ul>
            {result.relationships_used?.map((entry, i) => (
              <li key={i}>{entry}</li>
            ))}
          </ul>
        </details>
        <details>
          <summary>Raw result JSON</summary>
          <pre className="context">{JSON.stringify(result, null, 2)}</pre>
        </details>
      </section>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<"admin" | "user">("admin");
  const [token, setToken] = useState("");
  const [namespaceId, setNamespaceId] = useState("");
  const [saved] = useState(loadConnection);
  const [adminToken, setAdminToken] = useState<string | null>(saved.adminToken);
  const adminApi = useMemo(
    () => (adminToken ? new Api(adminToken) : null),
    [adminToken],
  );
  const [namespaces, setNamespaces] = useState<Namespace[]>([]);
  const [session, setSession] = useState<Session | null>(saved.session);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [storageAvailable, setStorageAvailable] = useState(true);

  useEffect(() => {
    setStorageAvailable(saveConnection({ adminToken, session }));
  }, [adminToken, session]);

  useEffect(() => {
    if (!adminApi || session) return;
    const controller = new AbortController();
    void adminApi
      .request<Namespace[]>(
        "/platform/namespaces/",
        undefined,
        controller.signal,
      )
      .then(setNamespaces)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [adminApi, session]);

  async function perform(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  async function authenticate(e: FormEvent) {
    e.preventDefault();
    await perform(async () => {
      const api = new Api(token.trim());
      const requestedId = namespaceId.trim().toLowerCase();
      if (mode === "admin") {
        const list = await api.request<Namespace[]>("/platform/namespaces/");
        setNamespaces(list);
        setAdminToken(token.trim());
      } else {
        if (!new RegExp(`^${namespaceIdPattern}$`).test(requestedId)) {
          throw new Error("Enter a valid namespace ID (UUID).");
        }
        const namespace = await api.request<Namespace>(
          "/platform/namespaces/me",
        );
        if (namespace.id.toLowerCase() !== requestedId) {
          throw new Error(
            "Namespace ID does not match this user token's namespace.",
          );
        }
        setSession({ namespace, token: token.trim() });
      }
      setToken("");
      setNamespaceId("");
    });
  }
  async function openNamespace(api: Api, id: string) {
    const minted = await api.request<{
      token: string;
      namespace_id: string;
      namespace_name: string;
    }>(`/platform/namespaces/${id}/issue-user-token`, {
      subject: "graph-core-ui",
      expires_in_days: 1,
    });
    setSession({
      namespace: { id: minted.namespace_id, name: minted.namespace_name },
      token: minted.token,
    });
  }
  async function connect(namespace: Namespace) {
    if (!adminApi) return;
    await perform(() => openNamespace(adminApi, namespace.id));
  }
  function disconnect() {
    setSession(null);
    setError("");
  }
  function signOut() {
    disconnect();
    setAdminToken(null);
    setNamespaces([]);
    setToken("");
    setNamespaceId("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">G</span>
          <div>
            Graph Core<small>KNOWLEDGE WORKSPACE</small>
          </div>
        </div>
        <div className="sidebar-context">
          <span className="eyebrow">NAMESPACE</span>
          <strong>{session?.namespace.name || "Not connected"}</strong>
          {session && <small>{session.namespace.id}</small>}
        </div>
        <div className="sidebar-bottom">
          <span className="connection-dot" /> REST API · via local proxy
          <p>
            {storageAvailable ? (
              <>
                Connection saved for this tab.
                <br />
                Refresh keeps you signed in.
              </>
            ) : (
              <>
                Browser storage unavailable.
                <br />
                Refresh will require signing in.
              </>
            )}
          </p>
          {(session || adminApi) && (
            <button className="secondary" onClick={signOut}>
              Sign out
            </button>
          )}
        </div>
      </aside>
      <main>
        {session ? (
          <Workspace
            key={session.namespace.id}
            session={session}
            onDisconnect={disconnect}
            canSwitch={!!adminApi}
          />
        ) : (
          <>
            <header>
              <div>
                <span className="eyebrow">CONTROL PLANE</span>
                <h1>Choose your workspace</h1>
                <p>
                  Connect a namespace to manage models, build collections, and
                  explore retrieval.
                </p>
              </div>
            </header>
            {error && (
              <div role="alert" className="alert">
                {error}
              </div>
            )}
            {!adminApi ? (
              <section className="panel connection-panel">
                <h2>Connect to Graph Core</h2>
                <p className="muted">
                  Start the API and worker, then provide an admin or namespace
                  user JWT.
                </p>
                <form onSubmit={authenticate}>
                  <fieldset disabled={busy}>
                    <Field label="Token type">
                      <select
                        value={mode}
                        onChange={(e) =>
                          setMode(e.target.value as "admin" | "user")
                        }
                      >
                        <option value="admin">
                          Admin — select or create a namespace
                        </option>
                        <option value="user">
                          User — connect directly to your namespace
                        </option>
                      </select>
                    </Field>
                    {mode === "user" && (
                      <>
                        <Field label="Namespace ID">
                          <input
                            value={namespaceId}
                            onChange={(e) => setNamespaceId(e.target.value)}
                            required
                            pattern={namespaceIdPattern}
                            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                            autoComplete="off"
                          />
                        </Field>
                        <p className="hint">
                          Required. Must match the namespace assigned to your
                          user JWT.
                        </p>
                      </>
                    )}
                    <Field label="JWT bearer token">
                      <input
                        type="password"
                        autoComplete="off"
                        value={token}
                        onChange={(e) => setToken(e.target.value)}
                        required
                        placeholder="Paste token"
                      />
                    </Field>
                    <button
                      disabled={
                        !token.trim() ||
                        (mode === "user" && !namespaceId.trim())
                      }
                    >
                      {busy ? "Connecting…" : "Connect"}
                    </button>
                  </fieldset>
                </form>
                <p className="hint">
                  {mode === "admin" ? (
                    <>
                      Generate an admin JWT with{" "}
                      <code>uv run graph-core-admin-jwt</code>. Select a
                      namespace after signing in.
                    </>
                  ) : (
                    <>
                      Use a namespace-scoped user JWT, not an admin JWT. Paste
                      only the token, without <code>Bearer</code>.
                    </>
                  )}
                </p>
              </section>
            ) : (
              <div className="two-column">
                <section className="panel">
                  <div className="section-heading">
                    <h2>Namespaces</h2>
                    <button
                      className="secondary"
                      disabled={busy}
                      onClick={() =>
                        perform(async () =>
                          setNamespaces(
                            await adminApi.request<Namespace[]>(
                              "/platform/namespaces/",
                            ),
                          ),
                        )
                      }
                    >
                      Refresh
                    </button>
                  </div>
                  {namespaces.length ? (
                    namespaces.map((ns) => (
                      <div className="list-row" key={ns.id}>
                        <div>
                          <strong>{ns.name}</strong>
                          <small>{ns.id}</small>
                        </div>
                        <button disabled={busy} onClick={() => connect(ns)}>
                          Open workspace →
                        </button>
                      </div>
                    ))
                  ) : (
                    <Empty>No namespaces yet. Create one to get started.</Empty>
                  )}
                </section>
                <section className="panel">
                  <h2>Create namespace</h2>
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      const form = e.currentTarget;
                      const data = new FormData(form);
                      void perform(async () => {
                        const created = await adminApi.request<
                          Namespace & { token: string }
                        >("/platform/namespaces/", {
                          name: value(data, "name"),
                        });
                        setNamespaces(
                          await adminApi.request<Namespace[]>(
                            "/platform/namespaces/",
                          ),
                        );
                        setSession({
                          namespace: created,
                          token: created.token,
                        });
                        form.reset();
                      });
                    }}
                  >
                    <fieldset disabled={busy}>
                      <Field label="Name">
                        <input
                          name="name"
                          required
                          placeholder="research-workspace"
                          pattern=".*\S.*"
                        />
                      </Field>
                      <button>{busy ? "Creating…" : "Create & open"}</button>
                    </fieldset>
                  </form>
                </section>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function Workspace({
  session,
  onDisconnect,
  canSwitch,
}: {
  session: Session;
  onDisconnect: () => void;
  canSwitch: boolean;
}) {
  const api = useMemo(() => new Api(session.token), [session.token]);
  const [tab, setTab] = useState<Tab>("query");
  const [collections, setCollections] = useState<Collection[]>([]);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [selected, setSelected] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [chunkJobId, setChunkJobId] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [progress, setProgress] = useState<Job | null>(null);
  const operation = useRef<AbortController | null>(null);
  const collection = collections.find((c) => c.id === selected);

  async function refresh(signal?: AbortSignal) {
    const [list, caps, recent] = await Promise.all([
      api.request<Collection[]>("/collections/", undefined, signal),
      api.request<Capabilities>("/platform/capabilities", undefined, signal),
      api.request<Job[]>("/jobs/?limit=30", undefined, signal),
    ]);
    setCollections(list);
    setCapabilities(caps);
    setJobs(recent);
    setSelected((current) =>
      list.some((c) => c.id === current) ? current : list[0]?.id || "",
    );
  }
  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => {
      controller.abort();
      operation.current?.abort();
    };
    // A workspace is remounted when its namespace changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  useEffect(() => {
    if (tab !== "jobs") return;
    const controller = new AbortController();
    const timer = setInterval(() => {
      void api
        .request<Job[]>("/jobs/?limit=30", undefined, controller.signal)
        .then(setJobs)
        .catch((e) => {
          if (!controller.signal.aborted) setError(e.message);
        });
    }, 3000);
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, [tab, api]);

  async function perform(task: (signal: AbortSignal) => Promise<void>) {
    operation.current?.abort();
    const controller = new AbortController();
    operation.current = controller;
    setBusy(true);
    setError("");
    setNotice("");
    setProgress(null);
    try {
      await task(controller.signal);
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }
  function switchCollection(id: string) {
    operation.current?.abort();
    setBusy(false);
    setSelected(id);
    setResult(null);
    setProgress(null);
    setNotice("");
    setError("");
  }
  function switchTab(next: Tab) {
    operation.current?.abort();
    setBusy(false);
    setTab(next);
    setProgress(null);
    setNotice("");
    setError("");
    setResult(null);
  }
  const profileOptions = (kind: "embedding" | "llm") =>
    (kind === "embedding"
      ? capabilities?.embedding_profiles
      : capabilities?.llm_profiles
    )?.map((p) => (
      <option key={p.profile_id} value={p.profile_id}>
        {p.label || p.model} · {p.provider}
      </option>
    ));

  return (
    <>
      <header>
        <div>
          <span className="eyebrow">{session.namespace.name} / WORKSPACE</span>
          <h1>{titles[tab]}</h1>
          <p>
            From raw text to graph-backed answers. See what your query actually
            retrieves.
          </p>
        </div>
        <button className="secondary" onClick={onDisconnect}>
          {canSwitch ? "Switch namespace" : "Disconnect"}
        </button>
      </header>
      <nav className="tabs" aria-label="Workspace sections">
        {(Object.keys(titles) as Tab[]).map((t) => (
          <button
            key={t}
            aria-current={tab === t ? "page" : undefined}
            onClick={() => switchTab(t)}
          >
            {titles[t]}
          </button>
        ))}
      </nav>
      {error && (
        <div role="alert" className="alert">
          {error}
        </div>
      )}
      {notice && (
        <div role="status" className="notice">
          {notice}
        </div>
      )}
      {loading ? (
        <Empty>Loading workspace…</Empty>
      ) : (
        <>
          {["query", "data"].includes(tab) && (
            <div className="collection-bar">
              <Field label="Active collection">
                <select
                  value={selected}
                  onChange={(e) => switchCollection(e.target.value)}
                  disabled={!collections.length}
                >
                  <option value="" disabled>
                    Select collection
                  </option>
                  {collections.map((c) => (
                    <option value={c.id} key={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </Field>
              {collection && (
                <span className="badge">{collection.strategy}</span>
              )}
              <button
                className="secondary"
                onClick={() => switchTab("collections")}
              >
                + New collection
              </button>
            </div>
          )}
          {tab === "query" && (
            <>
              {!collection ? (
                <Empty>
                  Create a collection, add some data, then ask your first
                  question.
                </Empty>
              ) : (
                <section className="panel">
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      const data = new FormData(e.currentTarget);
                      setResult(null);
                      void perform(async (signal) => {
                        const queued = await api.request<{ job_id: string }>(
                          `/collections/${selected}/query`,
                          {
                            question: value(data, "question"),
                            mode: optional(data, "mode"),
                            llm_profile_id: optional(data, "llm_profile_id"),
                          },
                          signal,
                        );
                        await waitForJob(
                          api,
                          queued.job_id,
                          signal,
                          setProgress,
                        );
                        const completed = await api.request<{
                          result: QueryResult;
                        }>(`/jobs/${queued.job_id}/result`, undefined, signal);
                        setResult(completed.result);
                        await refresh(signal);
                      });
                    }}
                  >
                    <fieldset disabled={busy}>
                      <Field label="Your question">
                        <textarea
                          name="question"
                          rows={3}
                          required
                          placeholder="What do we know about…?"
                        />
                      </Field>
                      <div className="form-row">
                        <Field label="Retrieval mode">
                          <select name="mode" key={selected}>
                            <option value="">
                              Collection default
                              {collection.default_query_mode
                                ? ` (${collection.default_query_mode})`
                                : ""}
                            </option>
                            {(collection.strategy === "vector"
                              ? []
                              : ["naive", "local", "global", "hybrid", "mix"]
                            ).map((m) => (
                              <option key={m}>{m}</option>
                            ))}
                          </select>
                        </Field>
                        <Field label="Answer model">
                          <select name="llm_profile_id">
                            <option value="">
                              Collection / server default
                            </option>
                            {profileOptions("llm")}
                          </select>
                        </Field>
                        <button>{busy ? "Querying…" : "Run query →"}</button>
                      </div>
                    </fieldset>
                  </form>
                </section>
              )}
              {progress && (
                <div role="status" className="job-progress">
                  <span className="badge">{progress.status}</span>
                  <span>Query job · {progress.id}</span>
                  {!terminalStatus(progress.status) && (
                    <span>Waiting for worker…</span>
                  )}
                </div>
              )}
              {result && <Result result={result} />}
            </>
          )}
          {tab === "data" &&
            (!collection ? (
              <Empty>Create a collection before adding data.</Empty>
            ) : (
              <section className="panel">
                <h2>Ingest into {collection.name}</h2>
                <p className="muted">
                  Paste text or load a UTF-8 text / Markdown file. Documents run
                  asynchronously; chunks ingest immediately.
                </p>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    const data = new FormData(e.currentTarget);
                    void perform(async (signal) => {
                      const endpoint = value(data, "ingest_type");
                      const text = value(data, "text");
                      if (
                        endpoint === "chunk" &&
                        capabilities &&
                        text.length > capabilities.max_chunk_size
                      )
                        throw new Error(
                          `Chunk exceeds ${capabilities.max_chunk_size.toLocaleString()} characters. Use document ingestion instead.`,
                        );
                      const response = await api.request<{
                        job_id?: string;
                        chunk_hash?: string;
                        entity_count?: number;
                        relationship_count?: number;
                      }>(
                        `/collections/${selected}/ingest/${endpoint}`,
                        {
                          text,
                          document_path: optional(data, "document_path"),
                          domain: optional(data, "domain"),
                        },
                        signal,
                      );
                      if (response.job_id) {
                        setNotice(
                          `Document queued · ${response.job_id}. You can leave this page; the worker will continue.`,
                        );
                        await waitForJob(
                          api,
                          response.job_id,
                          signal,
                          setProgress,
                        );
                        setNotice("Document ingestion completed.");
                      } else {
                        setNotice(
                          `Chunk added · ${response.entity_count || 0} entities, ${response.relationship_count || 0} relationships · ${response.chunk_hash}`,
                        );
                      }
                      await refresh(signal);
                    });
                  }}
                >
                  <fieldset disabled={busy}>
                    <Field label="Load text file">
                      <input
                        type="file"
                        accept=".txt,.md,.csv,.json,.jsonl,text/plain,text/markdown"
                        onChange={async (e) => {
                          const file = e.target.files?.[0];
                          const form = e.currentTarget.form;
                          if (!file || !form) return;
                          try {
                            if (file.size > 10 * 1024 * 1024)
                              throw new Error(
                                "Choose a text file smaller than 10 MB.",
                              );
                            const text = await file.text();
                            (
                              form.elements.namedItem(
                                "text",
                              ) as HTMLTextAreaElement
                            ).value = text;
                            (
                              form.elements.namedItem(
                                "document_path",
                              ) as HTMLInputElement
                            ).value = file.name;
                          } catch (err) {
                            setError(
                              err instanceof Error ? err.message : String(err),
                            );
                          }
                        }}
                      />
                    </Field>
                    <Field label="Content">
                      <textarea
                        name="text"
                        required
                        rows={12}
                        placeholder="Paste the source material you want to retrieve from…"
                      />
                    </Field>
                    <div className="form-row">
                      <Field label="Source path (optional)">
                        <input
                          name="document_path"
                          placeholder="notes/research.md"
                        />
                      </Field>
                      <Field label="Domain (optional)">
                        <input name="domain" placeholder="research" />
                      </Field>
                      <Field label="Ingestion type">
                        <select name="ingest_type">
                          <option value="doc">Document (background job)</option>
                          <option value="chunk">
                            Single chunk (synchronous)
                          </option>
                        </select>
                      </Field>
                    </div>
                    <button>
                      {busy ? "Ingesting…" : "Add to collection →"}
                    </button>
                  </fieldset>
                </form>
                {progress && (
                  <div className="job-progress" role="status">
                    <span>{progress.status}</span>
                    <progress
                      max={100}
                      value={progress.progress_percent || 0}
                    />
                    <span>{progress.progress_percent || 0}%</span>
                  </div>
                )}
              </section>
            ))}
          {tab === "collections" && (
            <div className="two-column">
              <section className="panel">
                <h2>
                  Your collections{" "}
                  <span className="count">{collections.length}</span>
                </h2>
                {collections.length ? (
                  collections.map((c) => (
                    <div className="list-row" key={c.id}>
                      <div>
                        <strong>{c.name}</strong>
                        <small>
                          {c.strategy} ·{" "}
                          {c.default_query_mode || "default mode"}
                        </small>
                        <small>{c.id}</small>
                      </div>
                      <button
                        className="secondary"
                        onClick={() => {
                          switchCollection(c.id);
                          switchTab("query");
                        }}
                      >
                        Explore →
                      </button>
                    </div>
                  ))
                ) : (
                  <Empty>No collections in this namespace.</Empty>
                )}
              </section>
              <section className="panel">
                <h2>Create collection</h2>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    const form = e.currentTarget;
                    const data = new FormData(form);
                    void perform(async (signal) => {
                      const created = await api.request<Collection>(
                        "/collections/",
                        {
                          name: value(data, "name"),
                          strategy: value(data, "strategy"),
                          embedding_profile_id: optional(
                            data,
                            "embedding_profile_id",
                          ),
                          llm_profile_id: optional(data, "llm_profile_id"),
                          default_query_mode: optional(
                            data,
                            "default_query_mode",
                          ),
                          gleaning_passes: Number(
                            value(data, "gleaning_passes"),
                          ),
                        },
                        signal,
                      );
                      await refresh(signal);
                      setSelected(created.id);
                      setNotice(`Created ${created.name}. Add data next.`);
                      form.reset();
                    });
                  }}
                >
                  <fieldset disabled={busy}>
                    <Field label="Name">
                      <input
                        name="name"
                        required
                        pattern=".*\S.*"
                        placeholder="research-notes"
                      />
                    </Field>
                    <Field label="Retrieval strategy">
                      <select name="strategy">
                        {(capabilities?.retrieval_strategies || []).map((s) => (
                          <option key={s}>{s}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Embedding profile">
                      <select name="embedding_profile_id">
                        <option value="">Server default</option>
                        {profileOptions("embedding")}
                      </select>
                    </Field>
                    <Field label="LLM profile">
                      <select name="llm_profile_id">
                        <option value="">Server default</option>
                        {profileOptions("llm")}
                      </select>
                    </Field>
                    <div className="form-row">
                      <Field label="Default query mode">
                        <select name="default_query_mode">
                          <option value="">Strategy default</option>
                          {["naive", "local", "global", "hybrid", "mix"].map(
                            (m) => (
                              <option key={m}>{m}</option>
                            ),
                          )}
                        </select>
                      </Field>
                      <Field label="Gleaning passes">
                        <input
                          name="gleaning_passes"
                          type="number"
                          min={0}
                          max={10}
                          defaultValue={1}
                          required
                        />
                      </Field>
                    </div>
                    <button>{busy ? "Creating…" : "Create collection"}</button>
                  </fieldset>
                </form>
              </section>
            </div>
          )}
          {tab === "profiles" && (
            <Profiles
              api={api}
              capabilities={capabilities}
              busy={busy}
              perform={perform}
              refresh={refresh}
              notify={setNotice}
            />
          )}
          {tab === "jobs" && (
            <section className="panel">
              <div className="section-heading">
                <h2>Recent jobs</h2>
                <button
                  className="secondary"
                  disabled={busy}
                  onClick={() =>
                    perform(async (signal) => {
                      await refresh(signal);
                    })
                  }
                >
                  Refresh
                </button>
              </div>
              <p className="muted">
                Refreshes every 3 seconds. Query results are stored durably and
                can be reopened here.
              </p>
              {jobs.length ? (
                jobs.map((job) => (
                  <div className="list-row" key={job.id}>
                    <div>
                      <strong>
                        {job.type === "query"
                          ? job.payload?.question ||
                            "Query (question unavailable)"
                          : job.type}{" "}
                        <span
                          className={`badge ${job.status === "failed" ? "failed" : ""}`}
                        >
                          {job.status}
                        </span>
                      </strong>
                      <small>
                        {job.type === "query" && "query · "}
                        {collections.find((c) => c.id === job.collection_id)
                          ?.name || job.collection_id}{" "}
                        {job.document_path && `· ${job.document_path}`}
                      </small>
                      <small>{job.id}</small>
                      {!!job.chunks_total && (
                        <small>
                          {job.chunks_completed || 0}/{job.chunks_total} chunks
                          processed · {job.progress_percent || 0}%
                        </small>
                      )}
                      {job.error && <p className="error-text">{job.error}</p>}
                    </div>
                    {job.type.startsWith("ingest") && (
                      <button
                        className="secondary"
                        onClick={() => setChunkJobId(job.id)}
                      >
                        Inspect chunks
                      </button>
                    )}
                    {job.type === "query" && job.status === "completed" && (
                      <button
                        className="secondary"
                        disabled={busy}
                        onClick={() =>
                          perform(async (signal) => {
                            const completed = await api.request<{
                              result: QueryResult;
                            }>(`/jobs/${job.id}/result`, undefined, signal);
                            setResult(completed.result);
                          })
                        }
                      >
                        Inspect result
                      </button>
                    )}
                  </div>
                ))
              ) : (
                <Empty>No jobs yet. Ingest a document or run a query.</Empty>
              )}
              {chunkJobId && (
                <ChunkStatus
                  key={chunkJobId}
                  api={api}
                  jobId={chunkJobId}
                  onClose={() => setChunkJobId("")}
                  onRetried={() => refresh()}
                />
              )}
              {result && <Result result={result} />}
            </section>
          )}
        </>
      )}
    </>
  );
}

function Profiles({
  api,
  capabilities,
  busy,
  perform,
  refresh,
  notify,
}: {
  api: Api;
  capabilities: Capabilities | null;
  busy: boolean;
  perform: (task: (signal: AbortSignal) => Promise<void>) => Promise<void>;
  refresh: (signal?: AbortSignal) => Promise<void>;
  notify: (message: string) => void;
}) {
  const [kind, setKind] = useState<"embedding" | "llm">("embedding");
  const [credentialId, setCredentialId] = useState("");
  const profiles: Profile[] = [
    ...(capabilities?.embedding_profiles || []),
    ...(capabilities?.llm_profiles || []),
  ];
  return (
    <div className="two-column">
      <section className="panel">
        <h2>Reusable model configurations</h2>
        <p className="muted">
          Profiles belong to this namespace. Bind them when creating a
          collection.
        </p>
        {profiles.length ? (
          profiles.map((p) => (
            <div className="profile-card" key={p.profile_id}>
              <span className="badge">{p.kind}</span>
              <h3>{p.label || p.model}</h3>
              <p>
                {p.provider} / {p.model}
                {p.dimensions && ` · ${p.dimensions} dimensions`}
              </p>
              <small>{p.profile_id}</small>
            </div>
          ))
        ) : (
          <Empty>
            No model profiles yet. Server defaults still work if configured.
          </Empty>
        )}
      </section>
      <div className="stack">
        <section className="panel">
          <h2>Create profile</h2>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const form = e.currentTarget;
              const data = new FormData(form);
              void perform(async (signal) => {
                await api.request(
                  "/platform/profiles",
                  {
                    kind,
                    provider: value(data, "provider"),
                    model: value(data, "model"),
                    label: optional(data, "label"),
                    credential_id: credentialId || null,
                    base_url: optional(data, "base_url"),
                    dimensions:
                      kind === "embedding"
                        ? Number(value(data, "dimensions"))
                        : null,
                    distance_metric:
                      kind === "embedding"
                        ? value(data, "distance_metric")
                        : null,
                    max_concurrent_calls: value(data, "max_concurrent_calls")
                      ? Number(value(data, "max_concurrent_calls"))
                      : null,
                  },
                  signal,
                );
                await refresh(signal);
                form.reset();
                setCredentialId("");
                notify("Profile created.");
              });
            }}
          >
            <fieldset disabled={busy}>
              <Field label="Kind">
                <select
                  value={kind}
                  onChange={(e) =>
                    setKind(e.target.value as "embedding" | "llm")
                  }
                >
                  <option value="embedding">Embedding</option>
                  <option value="llm">LLM</option>
                </select>
              </Field>
              <div className="form-row">
                <Field label="Provider">
                  <input name="provider" required placeholder="openai" />
                </Field>
                <Field label="Model">
                  <input
                    name="model"
                    required
                    placeholder={
                      kind === "embedding"
                        ? "text-embedding-3-small"
                        : "gpt-4o-mini"
                    }
                  />
                </Field>
              </div>
              <Field label="Label (optional)">
                <input name="label" placeholder="My model" />
              </Field>
              <Field label="Credential ID (optional)">
                <input
                  value={credentialId}
                  onChange={(e) => setCredentialId(e.target.value.trim())}
                  placeholder="Register below or paste an existing UUID"
                />
              </Field>
              <Field label="Base URL (optional)">
                <input
                  name="base_url"
                  type="url"
                  placeholder="https://api.openai.com/v1"
                />
              </Field>
              {kind === "embedding" && (
                <div className="form-row">
                  <Field label="Dimensions">
                    <input
                      name="dimensions"
                      type="number"
                      min={1}
                      required
                      defaultValue={1536}
                    />
                  </Field>
                  <Field label="Distance metric">
                    <select name="distance_metric">
                      <option>cosine</option>
                      <option>l2</option>
                      <option>ip</option>
                    </select>
                  </Field>
                </div>
              )}
              <Field label="Max concurrent calls (optional)">
                <input name="max_concurrent_calls" type="number" min={1} />
              </Field>
              <button>{busy ? "Saving…" : "Create profile"}</button>
            </fieldset>
          </form>
        </section>
        <section className="panel">
          <h2>Register provider credential</h2>
          <p className="muted">
            Secrets are encrypted by Graph Core. The returned ID is filled into
            the profile form above.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const form = e.currentTarget;
              const data = new FormData(form);
              void perform(async (signal) => {
                const result = await api.request<{ credential_id: string }>(
                  "/platform/credentials",
                  {
                    provider: value(data, "provider"),
                    secret: value(data, "secret"),
                    label: optional(data, "label"),
                    base_url: optional(data, "base_url"),
                  },
                  signal,
                );
                setCredentialId(result.credential_id);
                form.reset();
                notify(
                  "Credential registered. Complete the profile form above.",
                );
              });
            }}
          >
            <fieldset disabled={busy}>
              <Field label="Provider">
                <input name="provider" required placeholder="openai" />
              </Field>
              <Field label="API secret">
                <input
                  name="secret"
                  type="password"
                  autoComplete="off"
                  required
                />
              </Field>
              <Field label="Label (optional)">
                <input name="label" />
              </Field>
              <Field label="Provider base URL (optional)">
                <input name="base_url" type="url" />
              </Field>
              <button className="secondary">Register credential</button>
            </fieldset>
          </form>
        </section>
      </div>
    </div>
  );
}
