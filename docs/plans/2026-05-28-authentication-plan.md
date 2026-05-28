# Authentication & Authorization Plan

> **Status:** Design — awaiting implementation
>
> **Date:** 2026-05-28
>
> **Motivation:** Namespaces are user-scoped workspaces. The platform supports two
> deployment modes with different auth complexity:
>
> - **Self-hosted**: Single operator controls everything. Simple admin-key auth,
>   namespace-level tokens. No OAuth, no app registration overhead.
> - **Multi-tenant (hosted)**: Multiple independent consuming apps and users. Full
>   app registration, user-scoped JWTs, OAuth for admin portal.

---

## 1. Problem

The platform currently uses `X-Namespace-ID` header for namespace routing. This is
identification, not authentication — any client with a valid UUID gains full access.

Additionally, a self-hosted deployment shouldn't require OAuth infrastructure,
app registration, or JWT complexity. The operator should just set an admin key,
create namespaces, and use them.

---

## 2. Two Deployment Modes

`PLATFORM_MODE` env var selects the auth model. Default: `self_hosted`.

### Self-hosted (default)

One operator controls the entire platform. No separate "apps" or "users" concept —
the operator creates namespaces and issues namespace-scoped tokens directly.

```
1. Operator sets PLATFORM_ADMIN_KEY in .env

2. Operator creates a namespace via API:
   POST /platform/namespaces
   Authorization: Bearer $PLATFORM_ADMIN_KEY
   {"name": "my-workspace"}
   → {"id": "ns_abc", "name": "my-workspace", "api_key": "ns_key_xyz..."}

3. All requests use the namespace's api_key:
   POST /collections/
   Authorization: Bearer ns_key_xyz...
```

That's it. No app registration, no OAuth, no JWT complexity. The namespace `api_key`
is a long-lived token that grants access to that namespace only. RLS enforces isolation.

### Multi-tenant (hosted)

Multiple independent consuming apps, each with their own users. Platform acts as a
multi-tenant service with app registration, user-scoped tokens, and OAuth.

```
1. Platform admin registers apps:
   POST /admin/apps
   Authorization: Bearer $PLATFORM_ADMIN_KEY
   {"name": "scripture-tui", "owner_email": "dev@example.com"}
   → {"client_id": "app_tui_abc123", "client_secret": "secret_xyz..."}

2. End user logs in via Google OAuth in the TUI (platform not involved)

3. TUI requests user-scoped token from platform:
   POST /oauth/token
   {
     "grant_type": "client_credentials",
  "client_id": "app_tui_abc123",
      "client_secret": "secret_xyz...",
      "user": {"sub": "google:12345", "email": "user@example.com"}
    }
    → Short-lived JWT containing namespace_id

4. TUI uses JWT for all requests:
   POST /collections/
   Authorization: Bearer <jwt>
```

**Important:** This is not OAuth in the standard sense. The platform is not the
identity provider — users authenticate externally (e.g., Google in the TUI).
This endpoint performs **token exchange / identity bridging**: a trusted app
asserts a user identity, and the platform issues a scoped token. It is closer
to OIDC token exchange or JWT bearer assertion than to an OAuth authorization
code flow.

---

## 3. Identities per Mode

### Self-hosted

| Identity | Auth mechanism | Purpose |
|---|---|---|
| **Platform Admin** | `PLATFORM_ADMIN_KEY` env var | Create namespaces, manage platform |
| **Namespace** | `api_key` (returned on creation) | All queries, ingest, collections |

### Multi-tenant

| Identity | Auth mechanism | Purpose |
|---|---|---|
| **Platform Admin** | `PLATFORM_ADMIN_KEY` env var | Register apps, manage platform |
| **Registered App** | `client_id` + `client_secret` | Exchange credentials for user tokens |
| **End User** | Short-lived JWT (issued by platform) | Query, ingest, manage collections |

---

## 4. Flows

### 4.1 Self-hosted: Namespace Creation

```
POST /platform/namespaces
Authorization: Bearer $PLATFORM_ADMIN_KEY
{"name": "my-workspace"}

→ {
    "id": "ns_uuid",
    "name": "my-workspace",
    "api_key": "ns_key_<random_32_chars>"
  }
```

The `api_key` is returned once on creation. It can be regenerated later via
`POST /platform/namespaces/{id}/rotate-key`.

### 4.2 Self-hosted: Using a Namespace

```
# All API calls carry the namespace api_key
POST /collections/
Authorization: Bearer ns_key_<random_32_chars>
{"name": "docs", "strategy": "vector", ...}

POST /collections/{id}/query
Authorization: Bearer ns_key_<random_32_chars>
{"question": "What is dharma?"}
```

### 4.3 Multi-tenant: App Registration

```
POST /admin/apps
Authorization: Bearer $PLATFORM_ADMIN_KEY
{"name": "scripture-tui", "owner_email": "dev@example.com"}

→ {"client_id": "app_tui_abc123", "client_secret": "secret_xyz..."}
```

### 4.4 Multi-tenant: Token Exchange

The registered app exchanges its client credentials + authenticated user identity
for a platform access token. This is **token exchange / identity bridging**, not
an OAuth authorization flow. The platform trusts the registered app to have
already authenticated the user externally.

```
POST /token/exchange
{
  "client_id": "app_tui_abc123",
  "client_secret": "secret_xyz...",
  "user": {"sub": "google:12345", "email": "user@example.com"}
}
```

**Trust model:** The platform fully trusts the registered app to have
authenticated the user. The `user.sub` field is asserted by the app, not verified
by the platform. This is acceptable because:

- Apps are registered tenants, not public integrations
- The platform enforces isolation at the namespace boundary via RLS
- A compromised app secret only exposes that app's own users' namespaces

**Future hardening:** If semi-trusted or public integrations emerge, this can
evolve to accept signed assertions (OIDC ID tokens, JWT bearer grants) that the
platform validates against the external identity provider.

Platform:
  a) Validates client_id + secret
  b) Looks up existing **default namespace** for (app_id, user_sub)
  c) If none exists, creates one: "user-<google_sub>"
  d) Returns short-lived JWT

→ {
    "access_token": "<jwt>",
    "token_type": "bearer",
    "expires_in": 3600,
    "namespace_id": "ns_abc"
  }

**Note on namespace mapping:** The `(app_id, user_sub) → namespace` default
mapping is a convenience, not a permanent invariant. It provides each user
a default workspace on first login. Nothing prevents a user from having
multiple namespaces later (e.g., teams, shared workspaces, multiple projects).
The `app_user_links` table records the *default* link; additional links can
be added via `POST /platform/namespaces`.
```

### 4.5 Multi-tenant: Using a User Token

```
# All API calls carry the user JWT
POST /collections/
Authorization: Bearer <jwt>
{"name": "docs", "strategy": "vector", ...}
```

### 4.6 Admin Portal Auth (multi-tenant only)

For the admin portal (app management UI):

```
Admin visits /admin/login → redirected to Google/GitHub OAuth
→ OAuth callback validates identity
→ Platform issues admin session cookie
→ Admin can manage apps via /admin/apps
```

Scoped to admin portal only. Regular API requests use tokens as above.

---

## 5. Data Model

### Modified: `namespaces`

| New column | Type | Notes |
|---|---|---|
| `api_key_hash` | String(128) (nullable) | bcrypt hash of namespace API key (self-hosted) |
| `api_key_prefix` | String(8) (nullable) | Display prefix, e.g., "ns_key_a3f" |
| `owner_app_id` | UUID FK (nullable) | App that owns this namespace (multi-tenant) |
| `owner_user_sub` | String(256) (nullable) | User who owns this namespace (multi-tenant) |
| `metadata` | JSONB (nullable) | Extensible per-namespace data |

Existing columns (`id`, `name`, `created_at`) unchanged.

### New: `registered_apps` (multi-tenant only)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `client_id` | String(64) | Unique, returned to developer |
| `client_secret_hash` | String(128) | bcrypt hash of the secret |
| `name` | String(128) | Human-readable app name |
| `owner_email` | String(256) | Contact email |
| `active` | Boolean | Soft-delete support |
| `created_at` | timestamptz | Creation time |

### New: `app_user_links` (multi-tenant only)

| Column | Type | Notes |
|---|---|---|
| `app_id` | UUID FK | → registered_apps.id |
| `user_sub` | String(256) | External user identifier (e.g., "google:12345") |
| `namespace_id` | UUID FK | → namespaces.id |
| `created_at` | timestamptz | First-link time |

PK: `(app_id, user_sub)`

---

## 6. Auth Dependency

Single `get_auth_context` dependency that handles both modes transparently:

```python
async def get_auth_context(
    authorization: str = Header(default=None),
    x_namespace_id: str = Header(default=""),
) -> AuthContext:
    """
    Resolve auth context from request headers.

    Priority:
    1. Authorization: Bearer <jwt>        → multi-tenant (user token)
    2. Authorization: Bearer <admin_key>  → self-hosted (admin)
    3. Authorization: Bearer <ns_key_...> → self-hosted (namespace)
    4. X-Namespace-ID: <uuid>             → legacy (deprecated, self-hosted only)
    """
```

The dependency returns `AuthContext(namespace_id, mode, ...)` and sets the
`current_namespace_id` contextvar for RLS.

---

## 7. API Surface

### Self-hosted

```
POST   /platform/namespaces              Create namespace (admin key) → returns api_key
GET    /platform/namespaces              List namespaces (admin key)
GET    /platform/namespaces/me           Get current namespace (namespace key)
POST   /platform/namespaces/{id}/rotate-key  Regenerate api_key (admin key)

# All existing endpoints — auth via namespace api_key or admin key
GET  /platform/capabilities
POST /platform/credentials
POST /platform/profiles
POST /collections/
GET  /collections/
POST /collections/{id}/ingest/chunk
POST /collections/{id}/ingest/document
POST /collections/{id}/query
GET  /jobs/{id}
```

### Multi-tenant

```
# Admin — app management
POST   /admin/apps                       Register new app (admin key)
GET    /admin/apps                       List all apps (admin key)
GET    /admin/apps/{client_id}           Get app details (admin key)
POST   /admin/apps/{client_id}/rotate    Rotate client_secret (admin key)
DELETE /admin/apps/{client_id}           Deactivate app (admin key)

# Token exchange — user token issuance
POST   /token/exchange                   Exchange client creds + user identity for JWT

# Platform — namespace management
POST   /platform/namespaces              Create namespace (user JWT, optional)
GET    /platform/namespaces/me           Get current namespace (user JWT)

# All existing endpoints — auth via user JWT
GET  /platform/capabilities
POST /platform/credentials
POST /platform/profiles
POST /collections/
GET  /collections/
POST /collections/{id}/ingest/chunk
POST /collections/{id}/ingest/document
POST /collections/{id}/query
GET  /jobs/{id}
```

---

## 8. Token Design

### Self-hosted: Namespace API Key

- Format: `ns_key_<32 random hex chars>`
- Stored in DB as bcrypt hash (`namespaces.api_key_hash`)
- Long-lived, no expiration (can be rotated)
- Scoped to a single namespace

### Multi-tenant: User JWT

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

- Algorithm: HS256 (symmetric, simple)
- Key: `JWT_SECRET` env var
- TTL: 1 hour (configurable via `JWT_EXPIRATION_SECONDS`)
- Only the platform signs JWTs (not the consuming app)

**Migration path to asymmetric signing:** HS256 is sufficient initially
(single platform instance, no external validation). If the platform evolves
to multi-service, polyglot, or externally-validated deployments, migrate to
RS256 or EdDSA so services can verify tokens without sharing the signing secret.
The `get_auth_context` dependency abstracts the algorithm, so migration is
a config + key change, not a code change.

---

## 9. Dependencies (new packages)

| Package | Purpose | Used in |
|---|---|---|
| `PyJWT>=2.8.0` | JWT signing/validation | Multi-tenant only |
| `passlib[bcrypt]>=1.7.4` | Hashing secrets | Both modes |
| `secrets` | Stdlib, API key generation | Both modes |

No heavy dependencies. No OAuth library — `/token/exchange` is just credential
validation + JWT issuance.

---

## 10. Configuration

### Common (both modes)

| Env var | Required | Description |
|---|---|---|
| `PLATFORM_ADMIN_KEY` | Yes | Primary admin key (operator secret) |
| `PLATFORM_ADMIN_KEY_SECONDARY` | No | Secondary key for rotation window — both keys accepted |

**Admin key rotation:** Set `PLATFORM_ADMIN_KEY_SECONDARY` to the new key while
keeping `PLATFORM_ADMIN_KEY` as the old key. Both are accepted. Once all clients
rotate, swap: move secondary to primary, drop old primary.

### Self-hosted

| Env var | Required | Description |
|---|---|---|
| `PLATFORM_MODE` | No | Set to `self_hosted` (default) |

### Multi-tenant

| Env var | Required | Description |
|---|---|---|
| `PLATFORM_MODE` | Yes | Set to `multi_tenant` |
| `JWT_SECRET` | Yes | 32+ byte secret for signing JWTs |
| `JWT_EXPIRATION_SECONDS` | No | Token TTL (default: 3600) |
| `GOOGLE_OAUTH_CLIENT_ID` | No | Admin portal Google OAuth |
| `GOOGLE_OAUTH_CLIENT_SECRET` | No | Admin portal Google OAuth |
| `GITHUB_OAUTH_CLIENT_ID` | No | Admin portal GitHub OAuth |
| `GITHUB_OAUTH_CLIENT_SECRET` | No | Admin portal GitHub OAuth |

---

## 11. Implementation Phases

### Phase 1: Core auth + admin key rotation (both modes)

New files:
- `src/graph_core/models/registered_app.py` — `RegisteredApp` + `AppUserLink` models
- `src/graph_core/models/namespace.py` — add `api_key_hash`, `api_key_prefix`, `owner_app_id`, `owner_user_sub`, `metadata` columns
- `src/graph_core/services/auth_service.py` — token validation, namespace key management
- `src/graph_core/api/auth.py` — `get_auth_context` dependency, `AuthContext` dataclass
- `src/graph_core/api/namespaces.py` — `POST /platform/namespaces`, key rotation
- `alembic/versions/0008_auth_tables.py` — migration for new tables + namespace columns

Changes:
- `src/graph_core/config.py` — add `platform_mode`, `platform_admin_key`, `platform_admin_key_secondary` settings
- `src/graph_core/api/dependencies.py` — deprecate `get_namespace_id`

Admin key rotation: `get_auth_context` accepts both `PLATFORM_ADMIN_KEY` and
`PLATFORM_ADMIN_KEY_SECONDARY` when both are set. Env-only, no DB state needed.

**This phase covers self-hosted mode fully.** Multi-tenant tables are created
but unused until Phase 2.

### Phase 2: Multi-tenant (token exchange, app registration, JWT)

New files:
- `src/graph_core/services/token_service.py` — JWT issuance, client validation, namespace bootstrap
- `src/graph_core/api/token_exchange.py` — `POST /token/exchange` endpoint
- `src/graph_core/api/admin.py` — `POST /admin/apps` and other app management endpoints

Changes:
- `src/graph_core/api/auth.py` — add JWT validation path to `get_auth_context`
- `src/graph_core/config.py` — add `jwt_secret`, `jwt_expiration_seconds` settings
- `pyproject.toml` — add `PyJWT`, `passlib[bcrypt]` dependencies

### Phase 3: Migrate endpoints

Changes:
- `src/graph_core/api/platform.py` — replace `get_namespace_id` with `get_auth_context`
- `src/graph_core/api/collections.py` — same
- `src/graph_core/api/ingest.py` — same
- `src/graph_core/api/query.py` — same
- `src/graph_core/api/jobs.py` — same

### Phase 4: Admin portal auth (Google/GitHub OAuth)

New files:
- `src/graph_core/api/admin_auth.py` — Google/GitHub OAuth flow for admin portal

Changes:
- `src/graph_core/config.py` — OAuth client credentials
- `src/graph_core/api/admin.py` — add login/logout/session endpoints

### Phase 5: Cleanup

Changes:
- `src/graph_core/scripts/smoke_test.py` — update to use self-hosted auth flow
- `tests/conftest.py` — update fixtures for authenticated requests
- Remove raw DB namespace creation from test helpers
- Remove `X-Namespace-ID` header support (deprecation period first)

---

## 12. Backward Compatibility

During Phase 3, `get_auth_context` accepts (in priority order):
1. `Authorization: Bearer <jwt>` — multi-tenant user token
2. `Authorization: Bearer <admin_key>` — self-hosted admin
3. `Authorization: Bearer <ns_key_...>` — self-hosted namespace key
4. `X-Namespace-ID: <uuid>` — legacy, deprecated, logs warning

After deprecation period, option 4 is removed.

---

## 13. Security Considerations

1. **Namespace API key** stored as bcrypt hash — cannot be reversed from DB
2. **Client secret** (multi-tenant) stored as bcrypt hash — same
3. **JWT signing key** is separate from credential encryption key
4. **Short JWT TTL** in multi-tenant limits exposure
5. **RLS policies** unchanged — namespace isolation enforced at DB level
6. **Admin key** is env-only, never stored in DB; dual-key rotation supported
7. **Self-hosted mode** has no network-facing auth — relies on operator securing their deployment
8. **`client_secret` rotation** supported in multi-tenant

### Trust Model (multi-tenant)

The platform **fully trusts registered apps** to have authenticated users
before calling `/token/exchange`. The `user.sub` claim is asserted by the app,
not verified by the platform. This is correct because:

- Apps are registered tenants with vetted credentials
- The platform enforces isolation at the namespace boundary via RLS
- A compromised app only exposes its own users' namespaces

If semi-trusted or public integrations emerge, `/token/exchange` can evolve
to accept signed external assertions (OIDC ID tokens, JWT bearer grants)
validated against the issuing identity provider.

---

## 14. Future Considerations

- **RBAC within a namespace** — owned by consuming app
- **Signed user assertions** — OIDC/JWT bearer exchange for semi-trusted integrations
- **Asymmetric JWT signing** — RS256/EdDSA for multi-service or external validation
- **Token revocation before expiry** — TTL is short enough; add if needed
- **SAML / enterprise SSO** — future phase
- **Rate limiting** — future phase
- **Hosted mode user portal** — self-service app registration
- **Distribution divergence** — self-hosted and multi-tenant may eventually
  diverge into separate distributions (community/self-hosted vs hosted SaaS)
  as operational tooling, scaling, and security expectations differ

---

## 15. Design Principles

The strongest property of this design: **auth complexity scales with deployment
complexity**. A self-hosted user gets admin key + namespace keys with zero OAuth
or JWT overhead. A multi-tenant deployment gets full app registration, token
exchange, and user-scoped JWTs.

This preserves the platform boundary: graph-core remains infrastructure (namespace-aware,
collection-aware, credential-aware), not business-user-aware. The consuming app
owns users, RBAC, billing, and identity.
