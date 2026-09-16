"""Tests for raw extraction cache identity and endpoint descriptions.

These cover failure modes that were silent on ``main``:

- a cached payload written under a different prompt/schema contract, or by a
  different domain configuration, was served to newer code;
- two domains whose prompts differ shared one cache row, so the domain written
  second could never be stored and missed its own cache forever;
- a relationship endpoint missing from the entity inventory was created with an
  empty description, which makes it invisible to retrieval.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator

import pytest

from graph_core.models import domain_config
from graph_core.models.domain_config import (
    DOMAIN_CONFIGS,
    DomainConfig,
    register_domain,
)
from graph_core.services.graph.ingestion import chunk_processor
from graph_core.services.graph.ingestion.chunk_processor import (
    _get_raw_extraction,
    _save_raw_extraction,
)
from graph_core.services.graph_rag.contracts import (
    CODE_TAXONOMY_V0,
    GENERIC_ENDPOINTS_V0,
    LEGACY_FINGERPRINT,
    extraction_contract_for,
    extraction_identity_for,
    prompt_fingerprint,
)
from graph_core.services.graph_rag.extractor import (
    ExtractedEntity,
    ExtractionResult,
    LLMGraphExtractor,
)

OLD_CONTRACT = GENERIC_ENDPOINTS_V0
NEW_CONTRACT = "generic-independent-entities-v1"


@pytest.fixture(autouse=True)
def _restore_domain_registry() -> Iterator[None]:
    """Keep dynamically registered domains from leaking into other tests."""
    original = dict(domain_config.DOMAIN_CONFIGS)
    yield
    domain_config.DOMAIN_CONFIGS.clear()
    domain_config.DOMAIN_CONFIGS.update(original)


def _patch_identity(
    monkeypatch: pytest.MonkeyPatch,
    contract: str,
    fingerprint: str = "forced-fingerprint",
) -> None:
    """Pretend the process runs this identity, standing in for a contract bump."""
    monkeypatch.setattr(
        chunk_processor,
        "extraction_identity_for",
        lambda _: (contract, fingerprint),
    )


def _use_real_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        chunk_processor,
        "extraction_identity_for",
        extraction_identity_for,
    )


def _register_domain(name: str, **overrides: object) -> None:
    """Register a dynamically classified domain, as ingestion does."""
    base = DOMAIN_CONFIGS["general"]
    register_domain(
        DomainConfig(
            name=name,
            rel_types=list(overrides.get("rel_types", base.rel_types)),
            entity_guidance=str(overrides.get("entity_guidance", base.entity_guidance)),
            relationship_guidance=str(
                overrides.get("relationship_guidance", base.relationship_guidance)
            ),
            rel_type_guidance=str(
                overrides.get("rel_type_guidance", base.rel_type_guidance)
            ),
        )
    )


def _extraction(description: str) -> ExtractionResult:
    return ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Krishna", entity_type="Person", description=description
            )
        ],
        relationships=[],
    )


# ── identity ──


def test_contract_is_chosen_by_domain() -> None:
    assert extraction_contract_for("code") == CODE_TAXONOMY_V0
    assert extraction_contract_for(" Code ") == CODE_TAXONOMY_V0
    assert extraction_contract_for(None) == GENERIC_ENDPOINTS_V0
    assert extraction_contract_for("general") == GENERIC_ENDPOINTS_V0


def test_distinct_prompts_under_one_label_are_distinct_identities() -> None:
    """A dynamically registered domain can change a label's prompt in place."""
    _register_domain("krishna_texts", entity_guidance="Prefer devotional concepts.")
    first = extraction_identity_for("krishna_texts")

    _register_domain("krishna_texts", entity_guidance="Prefer historical persons.")
    second = extraction_identity_for("krishna_texts")

    assert first[0] == second[0]
    assert first[1] != second[1]


def test_cosmetic_prompt_edit_does_not_change_fingerprint() -> None:
    _register_domain("plain", entity_guidance="Prefer concepts.")
    _register_domain("spaced", entity_guidance="Prefer    concepts.")

    assert extraction_identity_for("plain") == extraction_identity_for("spaced")


def test_builtin_domains_with_different_prompts_differ() -> None:
    """The built-in ``books`` and ``personal`` resolve different vocabularies."""
    assert extraction_identity_for("books") != extraction_identity_for("personal")


def test_fingerprint_ignores_fields_that_cannot_affect_the_prompt() -> None:
    base = DOMAIN_CONFIGS["general"]
    chunking_only = DomainConfig(
        name=base.name,
        rel_types=list(base.rel_types),
        entity_guidance=base.entity_guidance,
        relationship_guidance=base.relationship_guidance,
        rel_type_guidance=base.rel_type_guidance,
        use_ast_chunking=not base.use_ast_chunking,
        requires_exact_resolution=not base.requires_exact_resolution,
    )

    assert prompt_fingerprint(base) == prompt_fingerprint(chunking_only)


# ── cache read/write ──


async def test_payload_from_another_contract_is_not_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_hash = "a" * 64
    collection_id = uuid.uuid4()

    _patch_identity(monkeypatch, NEW_CONTRACT)
    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("written under the new contract"),
    )

    _patch_identity(monkeypatch, OLD_CONTRACT)
    assert await _get_raw_extraction(chunk_hash, collection_id) is None

    _patch_identity(monkeypatch, NEW_CONTRACT)
    cached = await _get_raw_extraction(chunk_hash, collection_id)
    assert cached is not None
    assert cached.entities[0].description == "written under the new contract"


async def test_legacy_fingerprint_never_matches_a_computed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_hash = "e" * 64
    collection_id = uuid.uuid4()

    _patch_identity(monkeypatch, OLD_CONTRACT, LEGACY_FINGERPRINT)
    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("written before fingerprints"),
    )

    _use_real_identity(monkeypatch)
    assert await _get_raw_extraction(chunk_hash, collection_id) is None


async def test_two_domains_are_both_stored_and_both_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The books/personal collision: one row each, neither missing forever."""
    _use_real_identity(monkeypatch)
    chunk_hash = "f" * 64
    collection_id = uuid.uuid4()

    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("books payload"),
        domain="books",
    )
    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("personal payload"),
        domain="personal",
    )

    books = await _get_raw_extraction(chunk_hash, collection_id, domain="books")
    personal = await _get_raw_extraction(chunk_hash, collection_id, domain="personal")

    assert books is not None, "books row was not stored or not found"
    assert personal is not None, "personal row was not stored or not found"
    assert books.entities[0].description == "books payload"
    assert personal.entities[0].description == "personal payload"


async def test_reclassified_domain_does_not_read_the_old_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_real_identity(monkeypatch)
    chunk_hash = "g" * 64
    collection_id = uuid.uuid4()
    _register_domain("poetry", relationship_guidance="Emphasise imagery.")

    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("guidance v1 payload"),
        domain="poetry",
    )

    _register_domain("poetry", relationship_guidance="Emphasise meter and rhyme.")
    assert (
        await _get_raw_extraction(chunk_hash, collection_id, domain="poetry") is None
    )

    _register_domain("poetry", relationship_guidance="Emphasise imagery.")
    cached = await _get_raw_extraction(chunk_hash, collection_id, domain="poetry")
    assert cached is not None
    assert cached.entities[0].description == "guidance v1 payload"


async def test_identity_mismatch_is_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    chunk_hash = "c" * 64
    collection_id = uuid.uuid4()

    _patch_identity(monkeypatch, OLD_CONTRACT, "cached-fingerprint")
    await _save_raw_extraction(
        chunk_hash=chunk_hash,
        collection_id=collection_id,
        extraction=_extraction("stale payload"),
    )

    _patch_identity(monkeypatch, OLD_CONTRACT, "active-fingerprint")
    with caplog.at_level(logging.INFO, logger=chunk_processor.__name__):
        assert await _get_raw_extraction(chunk_hash, collection_id) is None

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("cache miss under active identity" in m for m in messages)
    assert any("cached-fingerprint" in m for m in messages)


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


# ── extractor parsing ──


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
