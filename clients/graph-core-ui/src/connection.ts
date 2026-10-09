import type { Namespace } from "./api";

export type Session = { namespace: Namespace; token: string };
export type Connection = { adminToken: string | null; session: Session | null };
export const connectionKey = "graph-core.connection.v1";

export function loadConnection(): Connection {
  const empty: Connection = { adminToken: null, session: null };
  try {
    const raw = sessionStorage.getItem(connectionKey);
    if (!raw) return empty;
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") return empty;
    const adminToken =
      typeof data.adminToken === "string" && data.adminToken.trim()
        ? data.adminToken
        : null;
    const candidate = data.session;
    const session =
      candidate &&
      typeof candidate.token === "string" &&
      candidate.token.trim() &&
      typeof candidate.namespace?.id === "string" &&
      candidate.namespace.id &&
      typeof candidate.namespace?.name === "string"
        ? {
            token: candidate.token,
            namespace: {
              id: candidate.namespace.id,
              name: candidate.namespace.name,
            },
          }
        : null;
    return { adminToken, session };
  } catch {
    return empty;
  }
}

export function saveConnection(connection: Connection): boolean {
  try {
    if (connection.adminToken || connection.session) {
      sessionStorage.setItem(connectionKey, JSON.stringify(connection));
    } else {
      sessionStorage.removeItem(connectionKey);
    }
    return true;
  } catch {
    // Storage can be disabled by browser settings; keep in-memory login working.
    return false;
  }
}
