"""Identity for the extraction contract that produced a cached payload.

``raw_chunk_extractions`` stores the exact LLM extraction for a chunk so an
identical chunk is never re-extracted. A cached payload is only reusable while
the prompt, structured schema, and parser shape that produced it are unchanged:
a payload written by one contract can be missing whole arrays that a later
contract requires, and reusing it silently produces a degraded graph instead of
an error.

Every writer therefore stamps a contract id on the row and every reader selects
on it, so "a row exists for this chunk" and "a row exists for this chunk under
the contract we are running" are different questions.

Bumping a contract
-----------------

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

#: Domain whose extractor uses the fixed code taxonomy schema.
CODE_DOMAIN = "code"

#: Relationships-only response with entities derived from relationship
#: endpoints. This is the contract every existing row was written under.
GENERIC_ENDPOINTS_V0 = "generic-endpoints-v0"

#: Fixed code taxonomy with endpoints as typed code objects.
CODE_TAXONOMY_V0 = "code-taxonomy-v0"

#: Contract assumed for rows written before contracts existed. Matches
#: ``RawChunkExtraction.extraction_contract``'s server default and the backfill
#: performed by migration 0028.
LEGACY_CONTRACT = GENERIC_ENDPOINTS_V0


def extraction_contract_for(domain: str | None) -> str:
    """Return the contract id for an extraction run over ``domain``.

    LightRAG and custom Graph RAG share the same generic prompt family, so they
    share a contract; a collection binds one strategy immutably and cannot mix
    cached payloads between them.
    """
    if (domain or "").strip().lower() == CODE_DOMAIN:
        return CODE_TAXONOMY_V0
    return GENERIC_ENDPOINTS_V0
