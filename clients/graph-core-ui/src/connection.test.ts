import { afterEach, expect, it, vi } from "vitest";
import { connectionKey, loadConnection, saveConnection } from "./connection";

afterEach(() => {
  vi.restoreAllMocks();
  sessionStorage.clear();
});

it("round-trips both tokens and the active namespace", () => {
  const connection = {
    adminToken: "admin-token",
    session: {
      token: "user-token",
      namespace: { id: "namespace-id", name: "Research" },
    },
  };
  expect(saveConnection(connection)).toBe(true);
  expect(loadConnection()).toEqual(connection);
});

it("ignores malformed saved state", () => {
  for (const raw of [
    "not JSON",
    "null",
    "42",
    '{"adminToken":123,"session":{"token":true}}',
  ]) {
    sessionStorage.setItem(connectionKey, raw);
    expect(loadConnection()).toEqual({ adminToken: null, session: null });
  }
});

it("removes saved credentials when signed out", () => {
  saveConnection({ adminToken: "admin-token", session: null });
  expect(saveConnection({ adminToken: null, session: null })).toBe(true);
  expect(sessionStorage.getItem(connectionKey)).toBeNull();
});

it("does not crash when browser storage is unavailable", () => {
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("Storage blocked");
  });
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("Storage blocked");
  });
  expect(loadConnection()).toEqual({ adminToken: null, session: null });
  expect(saveConnection({ adminToken: "admin-token", session: null })).toBe(
    false,
  );
});
