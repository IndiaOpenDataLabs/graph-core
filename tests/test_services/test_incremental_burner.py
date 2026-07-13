import uuid

from graph_core.models.collection import Collection
from graph_core.scripts.vedas_ingest_burner import compile_rows
from graph_core.services.graph_rag.extractor import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)


def _collection() -> Collection:
    return Collection(
        id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        namespace_id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        name="test",
        strategy="custom_graph_rag",
    )


def test_compile_rows_materializes_executable_reasoning_structure():
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity("Practice A", "PRACTICE", "A breathing practice."),
            ExtractedEntity("Calm", "OUTCOME", "A calm state."),
        ],
        relationships=[
            ExtractedRelationship(
                source_name="Practice A",
                target_name="Calm",
                description="Practice A promotes calm for a prepared practitioner.",
                keywords=["breath", "calm"],
                weight=0.9,
                rel_type="PROMOTES",
                conditions=("The practitioner is prepared",),
                exceptions=("Acute respiratory distress",),
                scopes=("Supervised practice",),
                modality="normative",
            )
        ],
    )

    entities, relationships = compile_rows(_collection(), extraction, "chunk-a")

    assert {
        "PRACTICE",
        "OUTCOME",
        "EVIDENCE_CHUNK",
        "PROPOSITION",
        "CONDITION",
        "EXCEPTION",
        "SCOPE",
        "RULE",
    } <= {entity.primary_type for entity in entities}
    assert {
        "PROMOTES",
        "SUBJECT",
        "OBJECT",
        "SUPPORTS",
        "CONDITION",
        "EXCEPTION",
        "APPLIES_TO",
        "ANTECEDENT_OF",
        "BLOCKS",
        "CONCLUDES",
    } <= {relationship.rel_type for relationship in relationships}


def test_cross_chunk_reasoning_ids_are_stable_but_evidence_is_local():
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity("A", "CONCEPT", "A."),
            ExtractedEntity("B", "CONCEPT", "B."),
        ],
        relationships=[
            ExtractedRelationship(
                "A",
                "B",
                "A causes B under C.",
                [],
                1.0,
                "CAUSES",
                conditions=("C",),
            )
        ],
    )

    first_entities, _ = compile_rows(_collection(), extraction, "chunk-a")
    second_entities, _ = compile_rows(_collection(), extraction, "chunk-b")
    first_by_type = {row.primary_type: row.id for row in first_entities}
    second_by_type = {row.primary_type: row.id for row in second_entities}

    assert first_by_type["PROPOSITION"] == second_by_type["PROPOSITION"]
    assert first_by_type["RULE"] == second_by_type["RULE"]
    assert first_by_type["CONDITION"] == second_by_type["CONDITION"]
    assert first_by_type["EVIDENCE_CHUNK"] != second_by_type["EVIDENCE_CHUNK"]
