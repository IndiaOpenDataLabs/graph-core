"""Identity for the extraction contract and prompt inputs of a cached payload.

``raw_chunk_extractions`` stores the exact LLM extraction for a chunk so an
identical chunk is never re-extracted. A cached payload is only reusable while
everything that shaped the prompt is unchanged: a payload written by one set of
prompt inputs can be missing whole arrays that a later set requires, and reusing
it silently produces a degraded graph instead of an error.

Two identities are recorded, because they fail in different ways:

``extraction_contract``
    The prompt and structured-schema family shipped by the code. It can be
    inferred for historical rows, because it is a property of the deployed code
    rather than of the input, and it is bumped by hand (see "Bumping a contract").

``prompt_fingerprint``
    A fingerprint of the domain configuration actually resolved when the payload
    was produced: the relationship vocabulary and the three guidance strings
    interpolated into the extraction and gleaning prompts. Domain configuration is
    mutable at runtime (``register_domain``), and distinct domains resolve to
    distinct prompts, so a domain label cannot stand in for the inputs. This
    cannot be reconstructed for historical rows, which are therefore recorded as
    ``LEGACY_FINGERPRINT`` and miss exactly once.

The pair is the cache identity, and both parts are in the table's unique key. A
row that matches the chunk but not the identity is a miss, not a hit.

Keying on resolved inputs rather than labels is deliberate: two domain labels whose
resolved configuration is identical produce identical prompts, and sharing one
payload between them is correct rather than a collision.

Bumping a contract
------------------

When the extraction prompt, the structured schema, or the shape of
``ExtractionResult`` changes:

1. add a new constant with the next version suffix and make
   ``extraction_contract_for`` return it;
2. leave the previous constant in place until those rows are no longer served,
   so existing rows keep resolving to the contract that produced them;
3. ship the constant change with no other behavioral coupling: old rows stay
   readable under their own contract and are only re-extracted when the content
   is ingested again under the new one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from graph_core.models.domain_config import DomainConfig, get_domain_config

#: Domain whose extractor uses the fixed code taxonomy schema.
CODE_DOMAIN = "code"

#: Relationships-only response with entities derived from relationship
#: endpoints. This is the contract every existing row was written under.
GENERIC_ENDPOINTS_V0 = "generic-endpoints-v0"

#: Fixed code taxonomy with endpoints as typed code objects.
CODE_TAXONOMY_V0 = "code-taxonomy-v0"

#: Fingerprint recorded for rows written before fingerprints existed. Nothing
#: about the configuration those runs resolved can be reconstructed, so those
#: rows never match a computed fingerprint and are rewritten on the next ingest.
#: Matches the column server default and the migration's backfill.
LEGACY_FINGERPRINT = "legacy"

#: Domain configuration fields interpolated into the extraction and gleaning
#: prompts. ``use_ast_chunking`` is excluded because it changes how text is split,
#: which changes the chunk hash itself, and ``requires_exact_resolution`` is
#: excluded because it acts on entity resolution after extraction and cannot make
#: a cached payload wrong.
PROMPT_FIELDS: tuple[str, ...] = (
    "rel_types",
    "entity_guidance",
    "relationship_guidance",
    "rel_type_guidance",
)


def extraction_contract_for(domain: str | None) -> str:
    """Return the contract id for an extraction run over ``domain``.

    LightRAG and custom Graph RAG share the same generic prompt family, so they
    share a contract; a collection binds one strategy immutably and cannot mix
    cached payloads between them.
    """
    if (domain or "").strip().lower() == CODE_DOMAIN:
        return CODE_TAXONOMY_V0
    return GENERIC_ENDPOINTS_V0


def prompt_inputs(config: DomainConfig) -> dict[str, Any]:
    """Return the configuration parts that reach the prompt.

    Whitespace is collapsed because it is not semantically meaningful in the
    rendered prompt, so an edit that only rewraps guidance text does not
    invalidate otherwise identical payloads. ``rel_types`` order is preserved
    because it is the rendered vocabulary order.
    """
    inputs: dict[str, Any] = {}
    for field in PROMPT_FIELDS:
        value = getattr(config, field, None)
        if isinstance(value, str):
            value = " ".join(value.split())
        elif isinstance(value, list):
            value = [" ".join(str(item).split()) for item in value]
        inputs[field] = value
    return inputs


def prompt_fingerprint(config: DomainConfig) -> str:
    """Fingerprint of the prompt-affecting parts of ``config``."""
    payload = json.dumps(
        prompt_inputs(config),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def extraction_identity_for(domain: str | None) -> tuple[str, str]:
    """Return ``(contract, prompt_fingerprint)`` for the run's active inputs.

    Resolved through :func:`get_domain_config`, the same lookup the prompt builder
    performs, so the fingerprint describes the configuration the imminent prompt
    will actually interpolate. Callers must therefore compute it at extraction
    time, after any ``register_domain`` for the current input: the domain registry
    is process-global and resolves per call.
    """
    return (
        extraction_contract_for(domain),
        prompt_fingerprint(get_domain_config(domain)),
    )
