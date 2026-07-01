"""LLM-based graph extractor for Custom Graph RAG.

Extracts entities and relationships from text chunks using structured extraction
(function calling / JSON mode).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from graph_core.llm.interface import LLMProvider
from graph_core.models.domain_config import (
    CODE_REL_TYPE_TAXONOMY,
    get_domain_config,
)
from graph_core.models.rel_types import (
    DEFAULT_REL_TYPE,
    normalize_rel_type,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str


@dataclass
class ExtractedRelationship:
    source_name: str
    target_name: str
    description: str
    keywords: list[str]
    weight: float
    rel_type: str = DEFAULT_REL_TYPE


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


_RELATIONSHIP_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
        },
        "target": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
        },
        "description": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        },
        "weight": {"type": "number"},
    },
    "required": [
        "source",
        "target",
        "description",
        "keywords",
        "weight",
    ],
}


_GENERIC_EXTRACTION_SCHEMA: dict[str, Any] = {
    "title": "graph_rag_relationship_extraction",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                **_RELATIONSHIP_ITEM_SCHEMA,
                "properties": {
                    **_RELATIONSHIP_ITEM_SCHEMA["properties"],
                    "rel_type": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "weight": {"type": "number"},
                            },
                            "required": ["name"],
                        },
                        "minItems": 1,
                    },
                },
                "required": [
                    "source",
                    "target",
                    "description",
                    "keywords",
                    "weight",
                    "rel_type",
                ],
            },
        },
    },
    "required": ["relationships"],
}


def _build_code_taxonomy_schema() -> dict[str, Any]:
    category_properties: dict[str, Any] = {}
    category_required: list[str] = []
    for category_name, rel_types in CODE_REL_TYPE_TAXONOMY:
        category_required.append(category_name)
        category_properties[category_name] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                rel_type.upper(): {
                    "type": "array",
                    "items": _RELATIONSHIP_ITEM_SCHEMA,
                }
                for rel_type in rel_types
            },
            "required": [rel_type.upper() for rel_type in rel_types],
        }

    return {
        "title": "graph_rag_code_extraction",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relationships": {
                "type": "object",
                "additionalProperties": False,
                "properties": category_properties,
                "required": category_required,
            },
        },
        "required": ["relationships"],
    }


_CODE_EXTRACTION_SCHEMA = _build_code_taxonomy_schema()


_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting
relationships from input text.

---Instructions---
1. Relationship extraction:
   - Treat the input text as one source context/chunk. Extract claims that
     are true inside this context, not global facts detached from the source.
   - Identify direct relationships between concrete mentions, entities, or
     concepts that are explicitly supported by the text.
   - For N-ary relationships, decompose them into binary pairs.
   - For each relationship, extract: source, target, description, keywords,
     weight, rel_type.
   - Relationship descriptions must explain the nature of the connection,
     the context in which it holds, and why it matters. Include the local
     evidence phrase or sentence when possible.
   - Treat relationships as undirected unless the text clearly indicates
     direction.
   - Avoid duplicate relationships.
   - Do not merge two things just because their surface names are similar.
     If a name appears in this chunk, extract the local mention from this
     chunk. Concept merging is handled later by the graph layer.

2. Output requirements:
   - Return structured JSON with one object: "relationships".
   - Each relationship item must use endpoint objects for source and target:
       source: {{name (string), type (string), description (string)}}
       target: {{name (string), type (string), description (string)}}
   - Endpoint objects are source-local mentions. Use generic, domain-neutral
     type labels such as PERSON, ORG, ROLE, SKILL, METHOD, FUNCTION, MODULE,
     PLACE, REQUIREMENT, CLAIM, EVENT, CONCEPT, DOCUMENT, SECTION, or the
     closest useful type for the text. Do not use a domain-specific schema.
   - Keep source and target descriptions separate from the relationship
     description. Endpoint descriptions describe the mention in this context.
   - Each relationship must also have top-level:
       description (string),
       keywords    (array of strings),
       weight      (float 0..1)
   - "rel_type" is a list of one or more objects, each with:
       name        (string)
       description (string, role-specific: explains the connection in
                    the semantic role of THIS rel_type entry)
       keywords    (array of strings, role-specific)
       weight      (float 0..1, role-specific confidence)
     {domain_rel_type_guidance}
   - Emit every distinct rel_type that is genuinely supported for the
     pair. Multi-entry rel_type lists are expected when the same pair
     carries different semantic roles. Each entry must have its own
     role-specific description and keywords. If two entries would say
     the same thing, collapse them to the single most specific one.
   - Each entry becomes its own edge in the graph; emitting an entry you
     cannot justify with a role-specific description produces duplicate
     noise.
   - {domain_relationship_guidance}
   - Use third-person phrasing and avoid pronouns where possible.
   - Only extract relationships explicitly supported by the text.
"""


_EXTRACTION_USER_PROMPT = """Extract all relationships
from the following text.

Text:
{text}

Return only the structured relationships object.
"""


_CODE_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting
entities and relationships from source code.

---Instructions---
1. Relationship extraction:
   - Treat the source code chunk as one source context. Extract local code
     mentions and relationships inside this chunk; do not collapse them into
     global concepts during extraction.
   - Identify direct code relationships between concrete code objects.
   - Use the fixed taxonomy below and keep the output structured by
     category and rel_type.
   - For every chunk, evaluate every rel_type, even when no evidence
     exists. Emit an empty array for unsupported rel_types.
   - Relationship descriptions should be short pseudo-code-flavored
     statements grounded in the source text.
   - Make each relationship description rich enough that a reader can
     understand what happens, when, why, and with what effect without
     needing to inspect the code again.
   - {domain_relationship_guidance}

2. Output requirements:
   - Return structured JSON with one object: "relationships".
   - Each relationship item must use endpoint objects for source and
     target:
       source: {{name (string), type (string), description (string)}}
       target: {{name (string), type (string), description (string)}}
   - Endpoint objects are source-local mentions. For code, useful endpoint
     types include FUNCTION, METHOD, CLASS, MODULE, VARIABLE, PARAMETER,
     API, FILE, CONFIG, TABLE, EVENT, CONCEPT, or CODE_OBJECT.
   - Keep source and target descriptions separate from the relationship
     description. Endpoint descriptions describe the mention in this chunk.
   - "relationships" must contain every category and every rel_type in
     the fixed taxonomy below.
   - Each rel_type field is an array of relationship objects with:
       source      (endpoint object)
       target      (endpoint object)
       description (string)
       keywords    (array of strings)
       weight      (float 0..1)
   - A rel_type array may be empty when the chunk does not support that
     operation.
   - {domain_rel_type_guidance}

{taxonomy_guidance}

3. Only extract relationships explicitly supported by the text.
"""


_CODE_RECONSTRUCTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist completing a first-pass code
extraction so the graph fully captures the logic of the source code.

---Instructions---
1. You will be given a first-pass graph extracted from the code.
2. Try to reconstruct the code's logic using only the extracted
   relationships. Identify what is missing to recreate that logic.
3. Emit only the additional relationships needed to recreate the logic,
   plus corrected versions of existing relationships whose description,
   keywords, weight, or endpoints do not yet match the code.
4. Do not repeat relationships that were already extracted correctly.
5. Keep naming consistent with the previously extracted endpoints so that
   corrections attach to the same (source, target, rel_type) edges.
6. Keep additions grounded in the code's actual execution flow, state
   changes, control flow, and API boundaries.
6a. Keep endpoint names as local code mentions from this chunk. Do not merge
    two mentions simply because they have similar names; concept resolution
    happens later.
7. Prefer semantically central code symbols, concepts, and operations.
   Do not add edges for ephemeral locals, loop counters, temporary
   accumulators, or helper builtins unless they are essential to the logic.
8. Relationship descriptions must be rich enough to understand what
   happens, when, why, and with what effect without looking at the code.
9. Use the fixed taxonomy below and keep the output structured by
   category and rel_type.
10. For every chunk, evaluate every rel_type, even when no evidence
    exists. Emit an empty array for unsupported rel_types.
11. Return only new or corrected relationships, not the whole graph. The
    additions are merged into the existing graph; nothing you omit is
    deleted.
12. Do not emit an entities section. The ingestion adapter will derive
    entities from the relationship endpoints after parsing.
13. {domain_relationship_guidance}

{taxonomy_guidance}

---Candidate Graph---
{existing_info}

---Source Code---
{text}
"""


def _code_taxonomy_prompt() -> str:
    lines = [
        "   - Use only the fixed code rel_type taxonomy below.",
        "   - For every chunk, return all categories and all rel_types.",
        "   - Leave unsupported rel_types as empty arrays.",
        "   - Do not invent new rel_types for code ingestion.",
    ]
    for category_name, rel_types in CODE_REL_TYPE_TAXONOMY:
        rendered = ", ".join(rel_type.upper() for rel_type in rel_types)
        lines.append(f"   - {category_name}: {rendered}")
    return "\n".join(lines)


def _format_code_existing_info(
    existing_entities: list[str],
    existing_relationships: list[tuple[str, str, str]],
) -> str:
    lines = ["Already extracted:"]
    lines.append(
        "Entities: " + ", ".join(existing_entities)
        if existing_entities
        else "Entities: (none)"
    )
    if existing_relationships:
        rel_text = ", ".join(
            f"{source} -[{rel_type}]-> {target}"
            for source, target, rel_type in existing_relationships
        )
    else:
        rel_text = "(none)"
    lines.append("Relationships: " + rel_text)
    return "\n".join(lines)


def _format_code_candidate_graph(
    entities: list[ExtractedEntity],
    relationships: list[ExtractedRelationship],
) -> str:
    lines: list[str] = ["Entities:"]
    if entities:
        for entity in entities:
            lines.append(
                f"- {entity.name} [{entity.entity_type}]: {entity.description}"
            )
    else:
        lines.append("- (none)")
    lines.append("Relationships:")
    if relationships:
        for rel in relationships:
            lines.append(
                f"- {rel.source_name} -[{rel.rel_type}]-> {rel.target_name}: "
                f"{rel.description} | keywords={', '.join(rel.keywords)} "
                f"| weight={rel.weight}"
            )
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _format_candidate_graph(
    entities: list[ExtractedEntity],
    relationships: list[ExtractedRelationship],
) -> str:
    return _format_code_candidate_graph(entities, relationships)


def _normalize_code_entity_name(name: str) -> str:
    if not name:
        return ""
    normalized = " ".join(name.strip().split())
    if len(normalized) > 256:
        normalized = normalized[:256]
    return normalized


def _parse_relationship_endpoint(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, str):
        name = value.strip()
        return (name, "", "UNKNOWN") if name else None
    if not isinstance(value, dict):
        return None
    name = value.get("name", "")
    description = value.get("description", "")
    entity_type = value.get("type", "UNKNOWN")
    if not isinstance(name, str) or not isinstance(description, str):
        return None
    if not isinstance(entity_type, str):
        entity_type = "UNKNOWN"
    normalized_name = " ".join(name.strip().split())
    if not normalized_name:
        return None
    return normalized_name, description.strip(), normalize_rel_type(entity_type)


def _parse_code_endpoint(value: Any) -> tuple[str, str, str] | None:
    parsed = _parse_relationship_endpoint(value)
    if parsed is None:
        return None
    name, description, entity_type = parsed
    normalized_name = _normalize_code_entity_name(name)
    if not normalized_name:
        return None
    return normalized_name, description, entity_type or "CODE_OBJECT"


def _collect_code_entities(relationships: Any) -> list[ExtractedEntity]:
    if not isinstance(relationships, dict):
        return []

    entity_descriptions: dict[str, str] = {}
    entity_types: dict[str, str] = {}
    entity_order: list[str] = []

    for category_name, rel_types in CODE_REL_TYPE_TAXONOMY:
        category_value = relationships.get(category_name, {})
        if not isinstance(category_value, dict):
            continue
        for rel_type in rel_types:
            bucket = category_value.get(rel_type.upper(), [])
            if isinstance(bucket, dict):
                bucket = [bucket]
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                if not isinstance(item, dict):
                    continue
                for endpoint_key in ("source", "target"):
                    parsed = _parse_code_endpoint(item.get(endpoint_key))
                    if parsed is None:
                        continue
                    name, description, entity_type = parsed
                    if name not in entity_descriptions:
                        entity_order.append(name)
                        entity_descriptions[name] = description
                        entity_types[name] = entity_type or "CODE_OBJECT"
                        continue
                    if (
                        description
                        and len(description) > len(entity_descriptions[name])
                    ):
                        entity_descriptions[name] = description

    return [
        ExtractedEntity(
            name=name,
            entity_type=entity_types.get(name) or "CODE_OBJECT",
            description=entity_descriptions.get(name, ""),
        )
        for name in entity_order
    ]


def _collect_generic_entities(relationships: Any) -> list[ExtractedEntity]:
    if not isinstance(relationships, list):
        return []

    entity_descriptions: dict[str, str] = {}
    entity_types: dict[str, str] = {}
    entity_order: list[str] = []
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        for endpoint_key in ("source", "target"):
            parsed = _parse_relationship_endpoint(rel.get(endpoint_key))
            if parsed is None:
                continue
            name, description, entity_type = parsed
            if name not in entity_descriptions:
                entity_order.append(name)
                entity_descriptions[name] = description
                entity_types[name] = entity_type or "UNKNOWN"
                continue
            if description and len(description) > len(entity_descriptions[name]):
                entity_descriptions[name] = description

    return [
        ExtractedEntity(
            name=name,
            entity_type=entity_types.get(name) or "UNKNOWN",
            description=entity_descriptions.get(name, ""),
        )
        for name in entity_order
    ]


_GLEANING_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist completing a first-pass relationship
graph so it fully captures the logic of the source passage.

---Instructions---
Read the source passage and the relationships already extracted from it.
Ask yourself: using only the extracted relationships, could a reader
reconstruct the meaning and logic of the passage? Identify what is missing
to recreate that logic, and emit only those additions or corrections.

1. Do not repeat relationships that were already extracted correctly.
2. Focus on:
   - relationships needed to recreate the passage's logic that the first
     pass missed
   - relationships that need a corrected description, keywords, weight, or
     rel_type to match the required structure or the passage's meaning
3. Keep naming consistent with the previously extracted endpoints so that
   corrections attach to the same (source, target, rel_type) edges.
3a. Preserve source-local mention identity. Do not introduce global canonical
    entities or merge across sources during gleaning; the graph layer will
    project mentions to concepts later.
4. Relationship descriptions must still explain the nature, context,
   and significance of the connection. Make them rich enough to
   understand the behavior without looking at the text again.
4a. {domain_relationship_guidance}
5. Each relationship's "rel_type" must follow this guidance:
   {domain_rel_type_guidance}
6. Preserve multiple genuinely distinct rel_type entries for the same
   source/target pair when the text supports them; do not collapse them
   to one generic edge unless the evidence really supports only one.
7. Return only new or corrected relationships, not the whole graph. The
   additions are merged into the existing graph; nothing you omit is
   deleted.
8. Do not emit an entities section. The ingestion adapter will derive
   entities from the relationship endpoints after parsing.
9. {domain_entity_guidance}
10. Only include items explicitly supported by the text.
"""


_GLEANING_USER_PROMPT = """Previously extracted relationships:
{existing_info}

Identify what is still missing to recreate the logic of the passage, then
extract those additional or corrected relationships from:

{text}

Only output new or corrected relationships in the structured relationships
object.
"""


class LLMGraphExtractor:
    """Extracts entities and relationships using LLM structured extraction."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._cache: dict[str, ExtractionResult] = {}

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        if not name:
            return ""
        normalized = " ".join(name.strip().split())
        if len(normalized) > 256:
            normalized = normalized[:256]
        return normalized

    @staticmethod
    def _coerce_rel_type_entries(
        value: Any,
        fallback_description: str,
        fallback_keywords: list[str],
        fallback_weight: float,
        vocab: list[str] | None = None,
        domain: str | None = None,
        strict_vocab: bool = False,
    ) -> list[dict[str, Any]]:
        """Normalize the LLM's ``rel_type`` field into a list of per-edge
        dicts, each carrying its own validated name, description,
        keywords, and weight.

        Accepts (in order of preference):
        1. List of objects: ``[{name, description, keywords, weight}, ...]``
        2. List of strings: ``["EXPLAINS", "CAUSES"]`` (back-compat;
           each entry inherits the relationship-level description,
           keywords, and weight)
        3. A single string (very old shape): wrapped into a single entry

        Each entry's name is normalized (uppercase, snake_case,
        alpha-leading) and optionally validated against the active
        domain vocab. Unknown normalized rel_types are accepted when
        the existing set is not expressive enough unless ``strict_vocab``
        is true.

        Returns at least one entry for non-strict callers; the fallback
        is ``[{"name": DEFAULT_REL_TYPE, ...}]`` when the LLM emitted
        nothing usable. Strict callers receive an empty list when the
        input does not match the active vocabulary.
        """
        if value is None:
            value = []
        if isinstance(value, str):
            raw_entries: list[Any] = [{"name": value}]
        elif isinstance(value, list):
            raw_entries = []
            for item in value:
                if isinstance(item, str):
                    raw_entries.append({"name": item})
                elif isinstance(item, dict):
                    raw_entries.append(item)
        else:
            raw_entries = [{"name": str(value)}]

        vocab_set = {v.upper() for v in vocab} if vocab else None
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            raw_name = entry.get("name")
            if not raw_name or not isinstance(raw_name, str):
                continue
            name = normalize_rel_type(raw_name)
            if vocab_set is not None and name not in vocab_set:
                if strict_vocab:
                    logger.info(
                        "Skipping rel_type=%r outside strict %s vocab",
                        raw_name,
                        domain,
                    )
                    continue
                logger.info(
                    "LLM emitted new rel_type=%r outside active %s vocab; "
                    "accepting normalized type %s",
                    raw_name,
                    domain,
                    name,
                )
            if name in seen:
                continue
            seen.add(name)

            description = entry.get("description", fallback_description)
            if not isinstance(description, str):
                description = fallback_description
            description = description.strip() or fallback_description

            keywords_value = entry.get("keywords", fallback_keywords)
            if isinstance(keywords_value, str):
                keywords = [
                    k.strip() for k in keywords_value.split(",") if k.strip()
                ]
            elif isinstance(keywords_value, list):
                keywords = [
                    str(k).strip()
                    for k in keywords_value
                    if k is not None and str(k).strip()
                ]
            else:
                keywords = list(fallback_keywords)

            weight_value = entry.get("weight", fallback_weight)
            try:
                weight = float(weight_value)
                weight = max(0.0, min(1.0, weight))
            except (ValueError, TypeError):
                weight = fallback_weight

            out.append({
                "rel_type": name,
                "description": description,
                "keywords": keywords,
                "weight": weight,
            })

        if not out and not strict_vocab:
            out.append({
                "rel_type": DEFAULT_REL_TYPE,
                "description": fallback_description,
                "keywords": list(fallback_keywords),
                "weight": fallback_weight,
            })
        return out

    @staticmethod
    def _build_extraction_prompt(
        text: str,
        entity_types: str,
        rel_type_vocab: list[str],
        domain: str | None,
    ) -> str:
        cfg = get_domain_config(domain)
        return (
            _EXTRACTION_SYSTEM_PROMPT.format(
                entity_types=entity_types,
                rel_type_vocab=", ".join(rel_type_vocab),
                domain_entity_guidance=cfg.entity_guidance,
                domain_rel_type_guidance=cfg.rel_type_guidance,
                domain_relationship_guidance=cfg.relationship_guidance,
            )
            + "\n\n"
            + _EXTRACTION_USER_PROMPT.format(text=text)
        )

    @staticmethod
    def _build_code_extraction_prompt(
        text: str,
        entity_types: str,
        domain: str | None,
    ) -> str:
        cfg = get_domain_config(domain)
        return (
            _CODE_EXTRACTION_SYSTEM_PROMPT.format(
                entity_types=entity_types,
                domain_entity_guidance=cfg.entity_guidance,
                domain_rel_type_guidance=cfg.rel_type_guidance,
                domain_relationship_guidance=cfg.relationship_guidance,
                taxonomy_guidance=_code_taxonomy_prompt(),
            )
            + "\n\n"
            + f"""Extract relationships only from the following code.

Text:
{text}

Return only the structured relationships object.
"""
        )

    @staticmethod
    def _build_gleaning_prompt(
        text: str,
        entity_types: str,
        existing_info: str,
        rel_type_vocab: list[str],
        domain: str | None,
    ) -> str:
        cfg = get_domain_config(domain)
        return (
            _GLEANING_SYSTEM_PROMPT.format(
                entity_types=entity_types,
                rel_type_vocab=", ".join(rel_type_vocab),
                domain_entity_guidance=cfg.entity_guidance,
                domain_rel_type_guidance=cfg.rel_type_guidance,
                domain_relationship_guidance=cfg.relationship_guidance,
            )
            + "\n\n"
            + _GLEANING_USER_PROMPT.format(existing_info=existing_info, text=text)
        )

    @staticmethod
    def _build_code_gleaning_prompt(
        text: str,
        entity_types: str,
        existing_info: str,
        domain: str | None,
    ) -> str:
        cfg = get_domain_config(domain)
        return (
            _CODE_EXTRACTION_SYSTEM_PROMPT.format(
                entity_types=entity_types,
                domain_entity_guidance=cfg.entity_guidance,
                domain_rel_type_guidance=cfg.rel_type_guidance,
                domain_relationship_guidance=cfg.relationship_guidance,
                taxonomy_guidance=_code_taxonomy_prompt(),
            )
            + "\n\n"
            + f"""Previously extracted relationships:
{existing_info}

Now extract any additional or corrected relationships from:

{text}

Only output the structured relationships object.
"""
        )

    @staticmethod
    def _build_code_reconstruction_prompt(
        text: str,
        entity_types: str,
        existing_info: str,
        domain: str | None,
    ) -> str:
        cfg = get_domain_config(domain)
        return (
            _CODE_RECONSTRUCTION_SYSTEM_PROMPT.format(
                entity_types=entity_types,
                domain_entity_guidance=cfg.entity_guidance,
                domain_relationship_guidance=cfg.relationship_guidance,
                taxonomy_guidance=_code_taxonomy_prompt(),
                existing_info=existing_info,
                text=text,
            )
        )

    @staticmethod
    def _parse_relationship_keywords(value: Any, fallback: list[str]) -> list[str]:
        if isinstance(value, str):
            return [k.strip() for k in value.split(",") if k.strip()]
        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ]
        return list(fallback)

    @classmethod
    def _extract_entities(cls, entities: Any) -> list[ExtractedEntity]:
        extracted: list[ExtractedEntity] = []
        if not isinstance(entities, list):
            return extracted
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            name = cls._normalize_entity_name(ent.get("name", ""))
            if not name:
                continue
            extracted.append(
                ExtractedEntity(
                    name=name,
                    entity_type=str(ent.get("type", "UNKNOWN")).upper(),
                    description=str(ent.get("description", "") or ""),
                )
            )
        return extracted

    @classmethod
    def _extract_generic_relationships(
        cls,
        relationships: Any,
        rel_vocab: list[str],
        domain: str | None,
        strict_vocab: bool = False,
    ) -> list[ExtractedRelationship]:
        extracted: list[ExtractedRelationship] = []
        if not isinstance(relationships, list):
            return extracted

        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            source_endpoint = _parse_relationship_endpoint(rel.get("source"))
            target_endpoint = _parse_relationship_endpoint(rel.get("target"))
            if not (source_endpoint and target_endpoint):
                continue
            source, _source_description, _source_type = source_endpoint
            target, _target_description, _target_type = target_endpoint
            rel_description = rel.get("description", "")
            rel_keywords = cls._parse_relationship_keywords(
                rel.get("keywords", []),
                fallback=[],
            )
            try:
                rel_weight = float(rel.get("weight", 1.0))
                rel_weight = max(0.0, min(1.0, rel_weight))
            except (ValueError, TypeError):
                rel_weight = 1.0
            for entry in cls._coerce_rel_type_entries(
                rel.get("rel_type"),
                fallback_description=rel_description,
                fallback_keywords=rel_keywords,
                fallback_weight=rel_weight,
                vocab=rel_vocab,
                domain=domain,
                strict_vocab=strict_vocab,
            ):
                extracted.append(
                    ExtractedRelationship(
                        source_name=source,
                        target_name=target,
                        description=entry["description"],
                        keywords=list(entry["keywords"]),
                        weight=entry["weight"],
                        rel_type=entry["rel_type"],
                    )
                )
        return extracted

    @classmethod
    def _extract_code_relationships(
        cls,
        relationships: Any,
        domain: str | None,
    ) -> list[ExtractedRelationship]:
        extracted: list[ExtractedRelationship] = []
        if not isinstance(relationships, dict):
            return extracted

        relationship_keys = {str(key).lower(): key for key in relationships.keys()}

        for category_name, rel_types in CODE_REL_TYPE_TAXONOMY:
            category_key = relationship_keys.get(category_name.lower(), category_name)
            category_value = relationships.get(category_key, {})
            if not isinstance(category_value, dict):
                continue
            rel_type_keys = {
                str(key).upper(): key for key in category_value.keys()
            }
            for rel_type in rel_types:
                rel_key = rel_type.upper()
                bucket_key = rel_type_keys.get(rel_key, rel_key)
                bucket = category_value.get(bucket_key, [])
                if isinstance(bucket, dict):
                    bucket = [bucket]
                if not isinstance(bucket, list):
                    continue
                for item in bucket:
                    if not isinstance(item, dict):
                        continue
                    source_endpoint = _parse_code_endpoint(item.get("source"))
                    target_endpoint = _parse_code_endpoint(item.get("target"))
                    if not (source_endpoint and target_endpoint):
                        continue
                    source, _source_description, _source_type = source_endpoint
                    target, _target_description, _target_type = target_endpoint
                    description = item.get("description", "")
                    if not isinstance(description, str):
                        description = ""
                    description = description.strip()
                    keywords = cls._parse_relationship_keywords(
                        item.get("keywords", []),
                        fallback=[],
                    )
                    try:
                        weight = float(item.get("weight", 1.0))
                        weight = max(0.0, min(1.0, weight))
                    except (ValueError, TypeError):
                        weight = 1.0
                    extracted.append(
                        ExtractedRelationship(
                            source_name=source,
                            target_name=target,
                            description=description,
                            keywords=keywords,
                            weight=weight,
                            rel_type=rel_key,
                        )
                    )
        return extracted

    @classmethod
    def _extract_relationships(
        cls,
        relationships: Any,
        rel_vocab: list[str],
        domain: str | None,
        *,
        is_code_domain: bool,
    ) -> list[ExtractedRelationship]:
        if is_code_domain:
            extracted = cls._extract_code_relationships(
                relationships,
                domain=domain,
            )
            if extracted:
                return extracted
            return cls._extract_generic_relationships(
                relationships,
                rel_vocab=rel_vocab,
                domain=domain,
                strict_vocab=True,
            )
        return cls._extract_generic_relationships(
            relationships,
            rel_vocab=rel_vocab,
            domain=domain,
        )

    @staticmethod
    def _merge_relationships(
        current: list[ExtractedRelationship],
        additions: list[ExtractedRelationship],
    ) -> tuple[list[ExtractedRelationship], int]:
        """Additively merge gleaned relationships into the current set.

        New ``(source, target, rel_type)`` edges are appended. When an
        edge already exists, the gleaned version replaces it in place so
        descriptions/keywords/weights can be corrected — but no existing
        edge is ever dropped. Returns the merged list and the count of
        newly added edges.
        """
        merged: dict[tuple[str, str, str], ExtractedRelationship] = {}
        order: list[tuple[str, str, str]] = []
        for rel in current:
            key = (rel.source_name, rel.target_name, rel.rel_type)
            if key not in merged:
                order.append(key)
            merged[key] = rel

        added = 0
        for rel in additions:
            key = (rel.source_name, rel.target_name, rel.rel_type)
            if key not in merged:
                order.append(key)
                added += 1
            merged[key] = rel

        return [merged[key] for key in order], added

    @staticmethod
    def _merge_entities(
        current: list[ExtractedEntity],
        additions: list[ExtractedEntity],
    ) -> tuple[list[ExtractedEntity], int]:
        """Additively merge entities by name, preferring the richer
        (longer) description and never dropping an existing entity.
        Returns the merged list and the count of newly added entities.
        """
        merged: dict[str, ExtractedEntity] = {}
        order: list[str] = []
        for ent in current:
            if ent.name not in merged:
                order.append(ent.name)
            merged[ent.name] = ent

        added = 0
        for ent in additions:
            existing = merged.get(ent.name)
            if existing is None:
                order.append(ent.name)
                merged[ent.name] = ent
                added += 1
                continue
            if len(ent.description or "") > len(existing.description or ""):
                merged[ent.name] = ExtractedEntity(
                    name=existing.name,
                    entity_type=existing.entity_type or ent.entity_type,
                    description=ent.description,
                )

        return [merged[name] for name in order], added

    async def extract(
        self,
        text: str,
        entity_types: list[str] | None = None,
        domain: str | None = None,
    ) -> ExtractionResult:
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in self._cache:
            logger.debug("Cache hit for chunk hash=%s", content_hash[:8])
            return self._cache[content_hash]

        types_str = ", ".join(entity_types) if entity_types else "general"
        cfg = get_domain_config(domain)
        rel_vocab = cfg.rel_types
        is_code_domain = (domain or "").strip().lower() == "code"
        if is_code_domain:
            prompt = self._build_code_extraction_prompt(
                text=text,
                entity_types=types_str,
                domain=domain,
            )
            schema = _CODE_EXTRACTION_SCHEMA
        else:
            prompt = self._build_extraction_prompt(
                text=text,
                entity_types=types_str,
                rel_type_vocab=rel_vocab,
                domain=domain,
            )
            schema = _GENERIC_EXTRACTION_SCHEMA

        try:
            result = await self._llm.structured_extract(
                prompt=prompt,
                schema=schema,
            )
        except Exception as e:
            logger.warning("LLM extraction failed, retrying once: %s", e)
            try:
                result = await self._llm.structured_extract(
                    prompt=prompt,
                    schema=schema,
                )
            except Exception as e2:
                logger.error("LLM extraction failed after retry: %s", e2)
                return ExtractionResult(entities=[], relationships=[])

        if is_code_domain:
            relationships_payload = result.get("relationships")
            entities = _collect_code_entities(relationships_payload)
            relationships = self._extract_code_relationships(
                relationships_payload,
                domain=domain,
            )
        else:
            relationships_payload = result.get("relationships")
            entities = _collect_generic_entities(relationships_payload)
            relationships = self._extract_relationships(
                relationships_payload,
                rel_vocab=rel_vocab,
                domain=domain,
                is_code_domain=is_code_domain,
            )

        extraction_result = ExtractionResult(
            entities=entities,
            relationships=relationships,
        )
        self._cache[content_hash] = extraction_result

        logger.info(
            "Extracted %d entities, %d relationships",
            len(entities),
            len(relationships),
        )
        return extraction_result

    async def extract_with_gleaning(
        self,
        text: str,
        entity_types: list[str] | None = None,
        max_gleaning: int = 1,
        domain: str | None = None,
    ) -> ExtractionResult:
        result = await self.extract(text, entity_types, domain=domain)
        current_entities = list(result.entities)
        current_relationships = list(result.relationships)

        if not current_entities and not current_relationships:
            return result

        cfg = get_domain_config(domain)
        rel_vocab = cfg.rel_types
        is_code_domain = (domain or "").strip().lower() == "code"
        for gleaning_pass in range(max_gleaning):
            types_str = ", ".join(entity_types) if entity_types else "general"
            if is_code_domain:
                gleaning_prompt = self._build_code_reconstruction_prompt(
                    text=text,
                    entity_types=types_str,
                    existing_info=_format_code_candidate_graph(
                        current_entities,
                        current_relationships,
                    ),
                    domain=domain,
                )
                schema = _CODE_EXTRACTION_SCHEMA
            else:
                gleaning_prompt = self._build_gleaning_prompt(
                    text=text,
                    entity_types=types_str,
                    existing_info=_format_candidate_graph(
                        current_entities,
                        current_relationships,
                    ),
                    rel_type_vocab=rel_vocab,
                    domain=domain,
                )
                schema = _GENERIC_EXTRACTION_SCHEMA

            try:
                gleamed = await self._llm.structured_extract(
                    prompt=gleaning_prompt,
                    schema=schema,
                )
            except Exception as e:
                logger.warning("Gleaning failed: %s", e)
                break

            if is_code_domain:
                relationships_payload = gleamed.get("relationships")
                gleaned_entities = _collect_code_entities(relationships_payload)
                gleaned_relationships = self._extract_code_relationships(
                    relationships_payload,
                    domain=domain,
                )
            else:
                relationships_payload = gleamed.get("relationships")
                gleaned_entities = _collect_generic_entities(relationships_payload)
                gleaned_relationships = self._extract_relationships(
                    relationships_payload,
                    rel_vocab=rel_vocab,
                    domain=domain,
                    is_code_domain=False,
                )

            current_relationships, added_rels = self._merge_relationships(
                current_relationships,
                gleaned_relationships,
            )
            current_entities, added_entities = self._merge_entities(
                current_entities,
                gleaned_entities,
            )

            # Stop once a pass introduces nothing new. Corrections to
            # existing edges still take effect because merging happens
            # before this check; we only bail when no edges/entities were
            # added.
            if added_rels == 0 and added_entities == 0:
                break

        return ExtractionResult(
            entities=current_entities,
            relationships=current_relationships,
        )
