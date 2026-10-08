"""Network-free regressions for the independent generic extraction contract."""

import logging
from copy import deepcopy

import pytest

from graph_core.llm.interface import LLMProvider
from graph_core.models import domain_config
from graph_core.models.domain_config import DomainConfig, register_domain
from graph_core.services.graph.ingestion.chunk_processor import (
    _get_raw_extraction,
    _save_raw_extraction,
)
from graph_core.services.graph_rag.contracts import (
    GENERIC_EXTRACTION_CONTRACT,
)
from graph_core.services.graph_rag.extractor import (
    _CODE_EXTRACTION_SCHEMA,
    _GENERIC_EXTRACTION_SCHEMA,
    _RELATIONSHIP_ITEM_SCHEMA,
    LLMGraphExtractor,
)

TEXT = "Krishna teaches Arjuna to perform duty without attachment to results."


class RecordingLLM(LLMProvider):
    def __init__(self, responses):
        self.responses = responses
        self.prompts = []
        self.schemas = []

    async def chat(self, messages):
        return "unused"

    async def chat_stream(self, messages):
        if False:
            yield ""

    async def structured_extract(self, prompt, schema):
        index = len(self.prompts)
        self.prompts.append(prompt)
        self.schemas.append(deepcopy(schema))
        return deepcopy(self.responses[index])


def entity(name, kind="Concept", description=None):
    return {
        "name": name,
        "type": kind,
        "description": description
        if description is not None
        else f"Meaning of {name}.",
    }


def relationship(source="Krishna", target="Arjuna"):
    return {
        "source": {"name": source, "description": "Endpoint teacher description."},
        "target": {"name": target, "description": "Endpoint student description."},
        "description": "Krishna teaches Arjuna how duty should be performed.",
        "keywords": ["teaching", "duty"],
        "weight": 0.95,
        "rel_type": [{"name": "TEACHES"}],
    }


def payload():
    return {
        "entities": [
            entity(
                "Duty", description="Action or obligation that should be performed."
            ),
            entity("Arjuna", "Person", "The recipient of Krishna's teaching."),
            entity(
                "Non-Attachment To Results",
                description="Acting without attachment to outcomes.",
            ),
            entity("Krishna", "Person", "The teacher addressing Arjuna."),
        ],
        "relationships": [relationship()],
    }


@pytest.mark.parametrize("domain", [None, "general", "chat", "classified-prose"])
async def test_independent_concepts_survive_one_structured_call(
    domain, monkeypatch, caplog
):
    if domain == "classified-prose":
        monkeypatch.setattr(
            domain_config, "DOMAIN_CONFIGS", dict(domain_config.DOMAIN_CONFIGS)
        )
        register_domain(
            DomainConfig(
                name=domain,
                rel_types=["TEACHES"],
                entity_guidance="Use meaningful concepts from this passage.",
                relationship_guidance="Keep edges source-grounded.",
                rel_type_guidance="Use reusable types.",
            )
        )
    llm = RecordingLLM([payload()])
    with caplog.at_level(logging.INFO):
        result = await LLMGraphExtractor(llm).extract_with_gleaning(
            TEXT, max_gleaning=0, domain=domain
        )

    assert len(llm.prompts) == 1
    assert {e.name for e in result.entities} == {
        "Krishna",
        "Arjuna",
        "Duty",
        "Non-Attachment To Results",
    }
    assert all(e.description for e in result.entities)
    assert {e.entity_type for e in result.entities} == {"PERSON", "CONCEPT"}
    assert [
        (r.source_name, r.rel_type, r.target_name) for r in result.relationships
    ] == [("Krishna", "TEACHES", "Arjuna")]
    # Entity descriptions, not the competing endpoint descriptions, are authoritative.
    assert next(e for e in result.entities if e.name == "Krishna").description == (
        "The teacher addressing Arjuna."
    )
    prompt = llm.prompts[0]
    assert "An entity does not require an edge" in prompt
    assert "Non-Attachment To Results" in prompt
    assert "Do not evaluate or connect every possible pair" in prompt
    assert "Preserve source-to-target direction" in prompt
    assert "Treat relationships as undirected" not in prompt
    assert "repairs=0 entities=4 relationships=1" in caplog.text
    assert f"contract={GENERIC_EXTRACTION_CONTRACT}" in caplog.text


def test_schema_requires_independent_descriptions_without_mutating_code_contract():
    assert _GENERIC_EXTRACTION_SCHEMA["required"] == ["entities", "relationships"]
    item = _GENERIC_EXTRACTION_SCHEMA["properties"]["entities"]["items"]
    assert item["required"] == ["name", "type", "description"]
    assert item["properties"]["description"]["minLength"] == 1
    generic_endpoint = _GENERIC_EXTRACTION_SCHEMA["properties"]["relationships"][
        "items"
    ]["properties"]["source"]
    assert generic_endpoint is not _RELATIONSHIP_ITEM_SCHEMA["properties"]["source"]
    assert _CODE_EXTRACTION_SCHEMA["required"] == ["relationships"]
    assert "entities" not in _CODE_EXTRACTION_SCHEMA["properties"]
    for category in _CODE_EXTRACTION_SCHEMA["properties"]["relationships"][
        "properties"
    ].values():
        for bucket in category["properties"].values():
            assert bucket["items"] is _RELATIONSHIP_ITEM_SCHEMA
            assert "rel_type" not in bucket["items"]["properties"]


async def test_normalized_case_whitespace_and_length_bind_to_inventory(caplog):
    long_name = "X" * 256
    response = {
        "entities": [entity("  Non-Attachment   To Results "), entity(long_name)],
        "relationships": [
            relationship(" non-attachment\tto results ", long_name + "suffix")
        ],
    }
    llm = RecordingLLM([response])
    with caplog.at_level(logging.INFO):
        result = await LLMGraphExtractor(llm).extract(TEXT)
    assert len(result.entities) == 2
    assert result.relationships[0].source_name == "Non-Attachment To Results"
    assert result.relationships[0].target_name == long_name
    assert "repairs=0" in caplog.text


async def test_duplicate_inventory_entries_are_normalized_and_typed():
    response = {
        "entities": [
            entity(" Duty ", "UNKNOWN", "Short."),
            entity("duty", "Concept", "Duty is the action one should perform."),
        ],
        "relationships": [],
    }
    result = await LLMGraphExtractor(RecordingLLM([response])).extract(TEXT)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Duty"
    assert result.entities[0].entity_type == "CONCEPT"
    assert result.entities[0].description == "Duty is the action one should perform."


async def test_missing_endpoint_repaired_once_with_richest_description(caplog):
    first = relationship()
    second = relationship("krishna", "Arjuna")
    second["target"]["description"] = (
        "Arjuna is the student receiving instruction about duty."
    )
    response = {
        "entities": [entity("Krishna", "Person")],
        "relationships": [first, second],
    }
    with caplog.at_level(logging.INFO):
        result = await LLMGraphExtractor(RecordingLLM([response])).extract(TEXT)
    assert len(result.entities) == 2
    repaired = next(e for e in result.entities if e.name == "Arjuna")
    assert repaired.entity_type == "UNKNOWN"
    assert repaired.description == second["target"]["description"]
    assert all(r.source_name == "Krishna" for r in result.relationships)
    assert "repairs=1 entities=2 relationships=2" in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        {"relationships": [relationship()]},
        {"entities": []},
        {"entities": None, "relationships": []},
    ],
)
async def test_generic_output_requires_both_arrays(response, caplog):
    result = await LLMGraphExtractor(RecordingLLM([response])).extract(TEXT)
    assert result.entities == []
    assert result.relationships == []
    assert "entities and relationships arrays are required" in caplog.text


@pytest.mark.parametrize("description", ["", " \n ", None, 42, {}, []])
async def test_empty_or_invalid_independent_descriptions_are_rejected(
    description, caplog
):
    response = {
        "entities": [{"name": "Duty", "type": "Concept", "description": description}],
        "relationships": [],
    }
    result = await LLMGraphExtractor(RecordingLLM([response])).extract(TEXT)
    assert result.entities == []
    assert "Rejecting entity with empty description" in caplog.text


async def test_rejected_description_can_be_repaired_from_endpoint():
    response = {
        "entities": [entity("Krishna", "Person", "  "), entity("Arjuna", "Person")],
        "relationships": [relationship()],
    }
    result = await LLMGraphExtractor(RecordingLLM([response])).extract(TEXT)
    repaired = next(e for e in result.entities if e.name == "Krishna")
    assert repaired.entity_type == "UNKNOWN"
    assert repaired.description == "Endpoint teacher description."


async def test_missing_endpoint_description_drops_edge_without_text_fallback(caplog):
    rel = relationship()
    rel["target"]["description"] = " "
    response = {"entities": [entity("Krishna", "Person")], "relationships": [rel]}
    with caplog.at_level(logging.INFO):
        result = await LLMGraphExtractor(RecordingLLM([response])).extract(TEXT)
    assert [e.name for e in result.entities] == ["Krishna"]
    assert result.relationships == []
    assert "dropped_relationships=1" in caplog.text


async def test_generic_endpoints_must_use_the_declared_object_shape():
    response = payload()
    response["relationships"][0]["source"] = "Krishna"
    response["relationships"][0]["target"] = "Arjuna"
    result = await LLMGraphExtractor(RecordingLLM([response])).extract(TEXT)
    assert len(result.entities) == 4
    assert result.relationships == []


async def test_invalid_gleaning_response_does_not_replace_the_inventory(caplog):
    llm = RecordingLLM([payload(), {"relationships": [relationship()]}])
    result = await LLMGraphExtractor(llm).extract_with_gleaning(TEXT, max_gleaning=2)
    assert len(llm.prompts) == 2
    assert len(result.entities) == 4
    assert len(result.relationships) == 1
    assert "entities and relationships arrays are required" in caplog.text


async def test_entity_only_first_pass_and_gleaning_preserve_independent_inventory(
    caplog,
):
    first = {"entities": [entity("Duty")], "relationships": []}
    additions = {"entities": [entity("Non-Attachment To Results")], "relationships": []}
    llm = RecordingLLM([first, additions, {"entities": [], "relationships": []}])
    with caplog.at_level(logging.INFO):
        result = await LLMGraphExtractor(llm).extract_with_gleaning(
            TEXT, max_gleaning=5
        )
    assert len(llm.prompts) == 3
    assert [e.name for e in result.entities] == ["Duty", "Non-Attachment To Results"]
    assert result.relationships == []
    assert all(schema == llm.schemas[0] for schema in llm.schemas)
    assert "Previously extracted entities and relationships" in llm.prompts[1]
    assert "Do not emit an entities section" not in llm.prompts[1]
    assert "phase=gleaning-1" in caplog.text


async def test_gleaning_binds_existing_entities_and_corrects_casing_without_repairs(
    caplog,
):
    correction = relationship("krishna", " arjuna ")
    correction["description"] = "Corrected teaching about duty."
    llm = RecordingLLM([payload(), {"entities": [], "relationships": [correction]}])
    with caplog.at_level(logging.INFO):
        result = await LLMGraphExtractor(llm).extract_with_gleaning(
            TEXT, max_gleaning=1
        )
    assert len(result.entities) == 4
    assert len(result.relationships) == 1
    assert result.relationships[0].description == correction["description"]
    assert "phase=gleaning-1 repairs=0" in caplog.text


async def test_gleaning_promotes_repair_to_typed_independent_entity():
    first = {
        "entities": [entity("Krishna", "Person")],
        "relationships": [relationship()],
    }
    second = {"entities": [entity("arjuna", "Person", "Student.")], "relationships": []}
    result = await LLMGraphExtractor(
        RecordingLLM([first, second])
    ).extract_with_gleaning(TEXT, max_gleaning=1)
    arjuna = next(e for e in result.entities if e.name == "Arjuna")
    assert arjuna.entity_type == "PERSON"
    assert arjuna.description == "Student."


async def test_gleaning_corrects_existing_type_and_shortens_description():
    first = {
        "entities": [entity("Duty", "Object", "A much longer incorrect description.")],
        "relationships": [],
    }
    second = {
        "entities": [entity("duty", "Concept", "Obligation.")],
        "relationships": [],
    }
    result = await LLMGraphExtractor(
        RecordingLLM([first, second])
    ).extract_with_gleaning(TEXT, max_gleaning=1)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Duty"
    assert result.entities[0].entity_type == "CONCEPT"
    assert result.entities[0].description == "Obligation."


async def test_independent_inventory_round_trips(test_graph_rag_collection):
    collection_id = test_graph_rag_collection.id
    chunk_hash = "a" * 64
    llm = RecordingLLM([payload()])
    fresh = await LLMGraphExtractor(llm).extract(TEXT)
    await _save_raw_extraction(chunk_hash, collection_id, fresh)
    assert await _get_raw_extraction(chunk_hash, collection_id) == fresh
    assert len(llm.prompts) == 1
