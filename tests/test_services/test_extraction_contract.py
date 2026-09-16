"""Tests for the raw extraction cache contract and endpoint descriptions.

These cover two failure modes that were silent on ``main``:

- a cached payload written under a different prompt/schema contract was served
  to a reader running a newer contract, because the cache key said nothing about
  what produced the payload;
- a relationship endpoint missing from the entity inventory was created with an
  empty description, which makes it invisible to retrieval.
"""

from __future__ import annotations

import logging
import uuid

import pytest

from graph_core.services.graph.ingestion import chunk_processor
from graph_core.services.graph.ingestion.chunk_processor import (
    _get_raw_extraction,
    _save_raw_extraction,
)
from graph_core.services.graph_rag.contracts import (
    CODE_TAXONOMY_V0,
    GENERIC_ENDPOINTS_V0,
    extraction_contract_for,
)
from graph_core.services.graph_rag.extractor import (
    ExtractedEntity,
    ExtractionResult,
    LLMGraphExtractor,
)

OLD_CONTRACT = GENERIC_ENDPOINTS_V0
NEW_CONTRACT = "generic-independent-entities-v1"


def _patch_contract(monkeypatch: pytest.MonkeyPatch, contract: str) -> None:
    """Pretend the process is running ``contract``, simulating a contract bump."""
    monkeypatch.setattr(chunk_processor, "extraction_contract_for", lambda _: contract)


def _extraction(description: str) -> ExtractionResult:
    return ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Krishna", entity_type="Person", description=description
            )
        ],
        relationships=[],
    )


def test_contract_is_chosen_by_domain() -> None:
    assert extraction_contract_for("code") == CODE_TAXONOMY_V0
    assert extraction_contract_for(" Code ") == CODE_TAXONOMY_V0
    assert extraction_contract_for(None) == GENERIC_ENDPOINTS_V0
    assert extraction_contract_for("general") == GENERIC_ENDPOINTS_V0


def test_generic_endpoint_descriptions_survive_parsing() -> None:
    payload = [
        {
            "source": {"name": "Krishna", "description": "The teacher."},
            "target": {"name": "Arjuna", "description": "The student."},
            "description": "Krishna teaches Arjuna.",
            "keywords": ["teaching"],
            "weight": 0.9,
            "rel_type": [{"name": "TEACHES", "description": "Gives instruction."}],
        }
    ]

    relationships = LLMGraphExtractor._extract_generic_relationships(
        payload, rel_vocab=[], domain=None
    )

    assert len(relationships) == 1
    assert relationships[0].source_description == "The teacher."
    assert relationships[0].target_description == "The student."


def test_gleaning_cannot_erase_first_pass_endpoint_descriptions() -> None:
    current = LLMGraphExtractor._extract_generic_relationships(
        [
            {
                "source": {"name": "Krishna", "description": "The teacher."},
                "target": {"name": "Arjuna", "description": "The student."},
                "description": "Krishna teaches Arjuna.",
                "keywords": ["teaching"],
                "weight": 0.9,
                "rel_type": [{"name": "TEACHES"}],
            }
        ],
        rel_vocab=[],
        domain=None,
    )
    # A gleaned correction of the same edge that names endpoints without
    # describing them.
    additions = LLMGraphExtractor._extract_generic_relationships(
        [
            {
                "source": "Krishna",
                "target": "Arjuna",
                "description": "Krishna instructs Arjuna on duty.",
                "keywords": ["teaching", "duty"],
                "weight": 1.0,
                "rel_type": [{"name": "TEACHES"}],
            }
        ],
        rel_vocab=[],
        domain=None,
    )

    merged, added = LLMGraphExtractor._merge_relationships(current, additions)

    assert added == 0
    assert len(merged) == 1
    assert merged[0].description == "Krishna instructs Arjuna on duty."
    assert merged[0].source_description == "The teacher."
    assert merged[0].target_description == "The student."


async def test_payload_from_another_contract_is_not_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_hash = "a" * 64
    collection_id = uuid.uuid4()

    _patch_contract(monkeypatch, NEW_CONTRACT)
    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("written under the new contract"),
    )

    _patch_contract(monkeypatch, OLD_CONTRACT)
    assert await _get_raw_extraction(chunk_hash, collection_id) is None

    _patch_contract(monkeypatch, NEW_CONTRACT)
    cached = await _get_raw_extraction(chunk_hash, collection_id)
    assert cached is not None
    assert cached.entities[0].description == "written under the new contract"


async def test_contracts_coexist_for_one_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_hash = "b" * 64
    collection_id = uuid.uuid4()

    _patch_contract(monkeypatch, OLD_CONTRACT)
    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("old contract payload"),
    )
    _patch_contract(monkeypatch, NEW_CONTRACT)
    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("new contract payload"),
    )

    _patch_contract(monkeypatch, OLD_CONTRACT)
    old = await _get_raw_extraction(chunk_hash, collection_id)
    _patch_contract(monkeypatch, NEW_CONTRACT)
    new = await _get_raw_extraction(chunk_hash, collection_id)

    assert old is not None and new is not None
    assert old.entities[0].description == "old contract payload"
    assert new.entities[0].description == "new contract payload"


async def test_contract_miss_reports_what_is_cached(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    chunk_hash = "c" * 64
    collection_id = uuid.uuid4()

    _patch_contract(monkeypatch, OLD_CONTRACT)
    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("stale payload"),
    )

    _patch_contract(monkeypatch, NEW_CONTRACT)
    with caplog.at_level(logging.INFO, logger=chunk_processor.__name__):
        assert await _get_raw_extraction(chunk_hash, collection_id) is None

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cache miss under active contract" in m for m in messages)
    assert any(OLD_CONTRACT in m for m in messages)


async def test_endpoint_descriptions_round_trip_through_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_hash = "d" * 64
    collection_id = uuid.uuid4()
    relationship = LLMGraphExtractor._extract_generic_relationships(
        [
            {
                "source": {"name": "Krishna", "description": "The teacher."},
                "target": {"name": "Duty", "description": "Action to be performed."},
                "description": "Krishna teaches duty.",
                "keywords": ["teaching"],
                "weight": 0.8,
                "rel_type": [{"name": "TEACHES"}],
            }
        ],
        rel_vocab=[],
        domain=None,
    )

    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=ExtractionResult(entities=[], relationships=relationship),
    )

    cached = await _get_raw_extraction(chunk_hash, collection_id)

    assert cached is not None
    assert cached.relationships[0].target_description == "Action to be performed."
