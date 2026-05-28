# Authentication & Authorization Plan

> **Status:** Design — awaiting implementation
>
> **Date:** 2026-05-28
>
> **Motivation:** Namespaces are user-scoped workspaces. End users create namespaces
> via their consuming apps (TUI, web client). The platform must verify that every
> request comes from a registered app acting on behalf of a known user, without
> the platform becoming an identity provider.

---

## 1. Problem

The platform currently uses `X-Namespace-ID` header for namespace routing. This is
identification, not authentication — any client with a valid UUID gains full access.
We need:

1. **App registration** — consuming apps register with the platform, receive credentials
2. **Token issuance** — platform issues short-lived user-scoped tokens
3. **Token validation** — all API requests carry a valid token
4. **Namespace creation** — apps create namespaces for users via the API, not raw SQL
5. **Developer identity** — platform operators use Google/GitHub OAuth to manage registered apps

---

## 2. Identities

Three distinct identities, each with its own authentication mechanism:

| Identity | Who | Auth mechanism | Purpose |
|---|---|---|---|
| **Platform Admin** | You (operator) | `PLATFORM_ADMIN_KEY` env var | Register apps, manage platform |
| **Registered App** | TUI, web client, MCP server | `client_id` + `client_secret` | Exchange credentials for user tokens |
| **End User** | Person using the app | Short-lived JWT (issued by platform) | Query, ingest, manage collections |

### Key distinction

- **`client_id` / `client_secret`** identify the *app*. They are used server-to-server
  to request tokens. The secret never touches end users.
- **Access token (JWT)** identifies the *user + namespace*. It is used for all
  user-facing operations. The app holds it on behalf of the logged-in user.

---

## 3. Flows

### 3.1 App Registration (one-time, admin only)

```
Admin → POST /admin/apps
        Authorization: Bearer $PLATFORM_ADMIN_KEY
        {"name": "scripture-tui", "owner_email": "dev@example.com"}

        → {"client_id": "app_tui_abc123", "client_secret": "secret_xyz..."}
```

The `client_secret` is returned once. The admin stores it in the app's server config.

### 3.2 End User Login & Token Issuance

```
1. User opens TUI, logs in via Google OAuth
   → TUI backend gets Google user info (sub, email, picture)
   → Platform is NOT involved in this step

2. TUI requests platform token for the user:
   POST /oauth/token
   {
     "grant_type": "client_credentials",
     "client_id": "app_tui_abc123",
     "client_secret": "secret_xyz...",
     "user": {
       "sub": "google:12345",
       "email": "user@example.com"
     }
   }

   Platform validates client_id + secret, then:
   a) Looks up existing namespace for (app_id, user_sub)
   b) If none exists, creates one: "user-<google_sub>"
   c) Returns short-lived JWT:
      {
        "sub": "google:12345",
        "app_id": "app_tui_abc123",
        "namespace_id": "ns_abc",
        "iat": 1716892800,
        "exp": 1716896400   (1 hour)
      }

3. TUI uses the JWT for all requests:
   POST /collections/
   Authorization: Bearer <jwt>
```

### 3.3 Token Refresh

```
POST /oauth/token
{
  "grant_type": "client_credentials",
  "client_id": "app_tui_abc123",
  "client_secret": "secret_xyz...",
  "user": { "sub": "google:12345", "email": "user@example.com" }
}
→ New JWT (namespace already exists, no new namespace created)
```

### 3.4 Admin Developer Login (Google/GitHub OAuth)

For the admin portal (app management UI):

```
Admin visits /admin/login → redirected to Google/GitHub OAuth
→ OAuth callback validates identity
→ Platform issues admin session cookie
→ Admin can manage apps via /admin/apps
```

This is scoped to the admin portal only. Regular API requests still use JWT.

---

## 4. Data Model

### New tables

**`registered_apps`** — registered consuming applications

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `client_id` | String(64) | Unique, returned to developer |
| `client_secret_hash` | String(128) | bcrypt hash of the secret |
| `name` | String(128) | Human-readable app name |
| `owner_email` | String(256) | Contact email |
| `active` | Boolean | Soft-delete support |
| `created_at` | timestamptz | Creation time |

**`app_user_links`** — maps (app, user) → namespace, prevents duplicate namespaces

| Column | Type | Notes |
|---|---|---|
| `app_id` | UUID FK | → registered_apps.id |
| `user_sub` | String(256) | External user identifier (e.g., "google:12345") |
| `namespace_id` | UUID FK | → namespaces.id |
| `created_at` | timestamptz | First-link time |

PK: `(app_id, user_sub)`

**`namespaces`** — modified (add `owner_app_id`, `owner_user_sub`)

| New column | Type | Notes |
|---|---|---|
| `owner_app_id` | UUID FK (nullable) | App that created this namespace |
| `owner_user_sub` | String(256) (nullable) | User who owns this namespace |
| `metadata` | JSONB (nullable) | Extensible per-namespace data |

Existing columns (`id`, `name`, `created_at`) unchanged. Backfill nullable for
existing namespaces.

---

## 5. API Surface

### New endpoints

```
# Admin — app management
POST   /admin/apps                      Register new app (admin key)
GET    /admin/apps                      List all apps (admin key)
GET    /admin/apps/{client_id}          Get app details (admin key)
POST   /admin/apps/{client_id}/rotate   Rotate client_secret (admin key)
DELETE /admin/apps/{client_id}          Deactivate app (admin key)

# OAuth — token issuance
POST   /oauth/token                     Exchange client creds for user JWT

# Platform — namespace management (auth: user JWT)
POST   /platform/namespaces             Create namespace (auto-created by /oauth/token)
GET    /platform/namespaces/me          Get current namespace info
```

### Modified endpoints

All existing endpoints currently using `get_namespace_id` (X-Namespace-ID header)
will switch to `get_auth_context` (Authorization: Bearer JWT):

```
GET  /platform/capabilities
POST /platform/credentials
POST /platform/profiles
GET  /platform/embedding-profiles
GET  /platform/llm-profiles
POST /collections/
GET  /collections/
GET  /collections/{id}
POST /collections/{id}/ingest/chunk
POST /collections/{id}/ingest/document
POST /collections/{id}/query
GET  /jobs/{id}
GET  /jobs/{id}/stream
```

---

## 6. JWT Design

### Structure

```json
{
  "sub": "google:12345",
  "email": "user@example.com",
  "app_id": "app_tui_abc123",
  "namespace_id": "ns_uuid_here",
  "iat": 1716892800,
  "exp": 1716896400,
  "iss": "graph-core-platform"
}
```

### Signing

- Algorithm: HS256 (simple, no PKI needed)
- Key: `JWT_SECRET` env var on platform (32+ byte random string)
- TTL: 1 hour (configurable via `JWT_EXPIRATION_SECONDS`)
- The app doesn't sign JWTs — only the platform does

### Validation dependency

Replaces `get_namespace_id` with `get_auth_context`:

```python
async def get_auth_context(authorization: str = Header(...)) -> AuthContext:
    """Validate Bearer JWT, return (namespace_id, app_id, user_sub)."""
    scheme, token = authorization.split()
    if scheme != "Bearer":
        raise HTTPException(401)

    payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    current_namespace_id.set(uuid.UUID(payload["namespace_id"]))
    return AuthContext(
        namespace_id=uuid.UUID(payload["namespace_id"]),
        app_id=payload["app_id"],
        user_sub=payload["sub"],
    )
```

---

## 7. Dependencies (new packages)

| Package | Purpose |
|---|---|
| `PyJWT>=2.8.0` | JWT signing and validation |
| `passlib[bcrypt]>=1.7.4` | Hashing `client_secret` |
| `httpx>=0.28.0` | Already installed, used for Google/GitHub OAuth |

No new heavy dependencies. No OAuth library needed for the `/oauth/token`
endpoint — it's just client credential validation + JWT issuance.

---

## 8. Configuration (new env vars)

| Env var | Description |
|---|---|
| `JWT_SECRET` | 32+ byte secret for signing JWTs |
| `JWT_EXPIRATION_SECONDS` | Token TTL (default: 3600) |
| `PLATFORM_ADMIN_KEY` | Static key for admin endpoints (one-time operator secret) |
| `GOOGLE_OAUTH_CLIENT_ID` | For admin portal Google OAuth (optional) |
| `GOOGLE_OAUTH_CLIENT_SECRET` | For admin portal Google OAuth (optional) |
| `GITHUB_OAUTH_CLIENT_ID` | For admin portal GitHub OAuth (optional) |
| `GITHUB_OAUTH_CLIENT_SECRET` | For admin portal GitHub OAuth (optional) |

---

## 9. Implementation Phases

### Phase 1: Core auth infrastructure

New files:
- `src/graph_core/models/registered_app.py` — `RegisteredApp` + `AppUserLink` models
- `src/graph_core/services/oauth_service.py` — JWT issuance, client validation, namespace bootstrap
- `src/graph_core/api/oauth.py` — `POST /oauth/token` endpoint
- `src/graph_core/api/admin.py` — `POST /admin/apps` endpoint
- `src/graph_core/api/auth.py` — `get_auth_context` dependency, `AuthContext` dataclass
- `alembic/versions/0008_auth_tables.py` — migration for new tables + namespace columns

Changes:
- `src/graph_core/models/namespace.py` — add `owner_app_id`, `owner_user_sub`, `metadata` columns
- `src/graph_core/config.py` — add JWT/admin/OAuth settings
- `src/graph_core/api/dependencies.py` — deprecate `get_namespace_id`, keep for backward compat

### Phase 2: Migrate endpoints to JWT

Changes:
- `src/graph_core/api/platform.py` — replace `get_namespace_id` with `get_auth_context`
- `src/graph_core/api/collections.py` — same
- `src/graph_core/api/ingest.py` — same
- `src/graph_core/api/query.py` — same
- `src/graph_core/api/jobs.py` — same

Add `POST /platform/namespaces` endpoint (optional manual namespace creation).

### Phase 3: Admin portal auth (Google/GitHub OAuth)

New files:
- `src/graph_core/api/admin_auth.py` — Google/GitHub OAuth flow for admin portal

Changes:
- `src/graph_core/config.py` — OAuth client credentials
- `src/graph_core/api/admin.py` — add login/logout/session endpoints

### Phase 4: Cleanup

Changes:
- `src/graph_core/scripts/smoke_test.py` — update to use auth flow
- `tests/conftest.py` — update fixtures for authenticated requests
- Remove raw DB namespace creation from test helpers
- Remove `X-Namespace-ID` header support (deprecation period first)

---

## 10. Backward Compatibility

During Phase 2, `get_auth_context` will accept both:
- `Authorization: Bearer <jwt>` (new, preferred)
- `X-Namespace-ID: <uuid>` (legacy, deprecated, logs warning)

After deprecation period (1 sprint), legacy header is removed.

---

## 11. Security Considerations

1. **Client secret** stored as bcrypt hash — platform staff cannot reverse it
2. **JWT signing key** is separate from credential encryption key
3. **Short token TTL** — limits exposure window
4. **RLS policies** unchanged — they already enforce namespace isolation at DB level
5. **App-user-namespace mapping** stored in `app_user_links` — prevents namespace hijacking
6. **`client_secret` rotation** supported — old secret gracefully rejected
7. **Admin key** is env-only, never stored in DB

---

## 12. Out of Scope

- RBAC within a namespace (e.g., reader/writer roles) — owned by consuming app
- Token revocation before expiry — TTL is short enough; add if needed
- Refresh token flow — apps re-request tokens with client credentials
- SAML / enterprise SSO — future phase
- Rate limiting — future phase
- Multi-namespace users (one user, multiple namespaces per app) — future phase
