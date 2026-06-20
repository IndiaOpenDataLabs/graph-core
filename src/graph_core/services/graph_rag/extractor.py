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
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
        },
        "target": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
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
    "title": "graph_rag_extraction",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "description": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "weight": {"type": "number"},
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
    "required": ["entities", "relationships"],
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
entities and relationships from input text.

---Instructions---
1. Entity extraction:
   - Identify clearly defined, meaningful entities that are
     explicitly supported by the text.
   - For each entity, extract: name, type, description.
   - Entity types to consider: {entity_types}. If none apply, use "Other".
   - {domain_entity_guidance}
   - Do not extract generic filler terms or vague concepts that are
     not functioning as real entities in context.

2. Relationship extraction:
   - Identify direct relationships between extracted entities.
   - For N-ary relationships, decompose them into binary pairs.
   - For each relationship, extract: source, target, description, keywords, weight, rel_type.
   - Relationship descriptions must explain the nature of the
     connection, the context in which it holds, and why it matters.
   - Treat relationships as undirected unless the text clearly indicates direction.
   - Avoid duplicate relationships.

3. Output requirements:
   - Return structured JSON with two arrays: "entities" and "relationships".
   - Each entity must have: name, type, description.
   - Each relationship must have: source, target, rel_type.
   - Each relationship must also have top-level:
       description (string),
       keywords    (array of strings),
       weight      (float 0..1)
   - "rel_type" is a list of one or more objects, each with:
       name        (string)
       description (string, role-specific: explains the connection
                    in the semantic role of THIS rel_type entry)
       keywords    (array of strings, role-specific)
       weight      (float 0..1, role-specific confidence)
     {domain_rel_type_guidance}
   - Emit every distinct rel_type that is genuinely supported for the
     pair. Multi-entry rel_type lists are expected when the same pair
     carries different semantic roles. Each entry must have its own
     role-specific description and keywords. If two entries would say
     the same thing, collapse them to the single most specific one.
   - Each entry becomes its own edge in the graph; emitting an
     entry you cannot justify with a role-specific description
     produces duplicate noise.
   - {domain_relationship_guidance}
   - Use third-person phrasing and avoid pronouns where possible.
   - Only extract entities and relationships explicitly supported by the text.
"""


_EXTRACTION_USER_PROMPT = """Extract all entities and relationships
from the following text.

Text:
{text}

Output all entities first, then all relationships.
"""


_CODE_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting
entities and relationships from source code.

---Instructions---
1. Relationship extraction:
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
       source: {{name (string), description (string)}}
       target: {{name (string), description (string)}}
   - Keep source and target descriptions separate from the relationship
     description. The endpoint descriptions are entity descriptions.
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
You are a Knowledge Graph Specialist responsible for refining a first-pass
code extraction into a cleaner final graph.

---Instructions---
1. You will be given a first-pass graph extracted from code.
2. Reconstruct the code logic from the graph.
3. If the graph is missing code logic, add more relationships or update
   the current ones.
4. If the graph includes unnecessary or overly local relationship items,
   remove them.
5. If the graph includes relationships that are too generic or too
   abstract to explain the code, replace them with better ones.
6. Keep the final result grounded in the code's actual execution flow,
   state changes, control flow, and API boundaries.
7. Keep only semantically central code symbols, concepts, and operations.
   Exclude ephemeral locals, loop counters, temporary accumulators, and
   helper builtins unless they are essential to the logic.
8. Relationship descriptions must be rich enough to understand what
   happens, when, why, and with what effect without looking at the code.
9. Use the fixed taxonomy below and keep the output structured by
   category and rel_type.
10. For every chunk, evaluate every rel_type, even when no evidence
    exists. Emit an empty array for unsupported rel_types.
11. Return the final graph, not a diff.
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


def _parse_code_endpoint(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name", "")
    description = value.get("description", "")
    if not isinstance(name, str) or not isinstance(description, str):
        return None
    normalized_name = _normalize_entity_name(name)
    if not normalized_name:
        return None
    return normalized_name, description.strip()


def _collect_code_entities(relationships: Any) -> list[ExtractedEntity]:
    if not isinstance(relationships, dict):
        return []

    entity_descriptions: dict[str, str] = {}
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
                    name, description = parsed
                    if name not in entity_descriptions:
                        entity_order.append(name)
                        entity_descriptions[name] = description
                        continue
                    if description and len(description) > len(entity_descriptions[name]):
                        entity_descriptions[name] = description

    return [
        ExtractedEntity(
            name=name,
            entity_type="CODE_OBJECT",
            description=entity_descriptions.get(name, ""),
        )
        for name in entity_order
    ]


_GLEANING_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting
entities and relationships from input text.

---Instructions---
Based on the previous extraction, identify only missed or incorrectly
formatted entities and relationships.

1. Do not repeat entities or relationships that were already extracted correctly.
2. Focus on:
   - entities missed in the first pass
   - relationships missed in the first pass
   - items that need correction to match the required structure
3. Entity types to consider: {entity_types}. If none apply, use "Other".
4. {domain_entity_guidance}
5. Keep naming consistent with the previously extracted entities.
6. Relationship descriptions must still explain the nature, context,
   and significance of the connection. Make them rich enough to
   understand the behavior without looking at the code again.
6a. {domain_relationship_guidance}
7. Each relationship's "rel_type" must follow this guidance:
   {domain_rel_type_guidance}
8. Preserve multiple genuinely distinct rel_type entries for the same
   source/target pair when the text supports them; do not collapse them
   to one generic edge unless the evidence really supports only one.
9. Return only new or corrected items in the same JSON structure as
   the main extraction.
10. Only include items explicitly supported by the text.
"""


_GLEANING_USER_PROMPT = """Previously extracted:
{existing_info}

Now extract any additional or corrected entities and relationships from:

{text}

Only output new or corrected items.
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
                    "LLM emitted new rel_type=%r outside active %s vocab; accepting normalized type %s",
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
            source = cls._normalize_entity_name(rel.get("source", ""))
            target = cls._normalize_entity_name(rel.get("target", ""))
            if not (source and target):
                continue
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
                    source, _source_description = source_endpoint
                    target, _target_description = target_endpoint
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
            entities = self._extract_entities(result.get("entities"))
            relationships = self._extract_relationships(
                result.get("relationships"),
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
                existing_info = "Already extracted:\n"
                existing_info += (
                    "Entities: "
                    + ", ".join(e.name for e in current_entities)
                    + "\n"
                )
                existing_info += "Relationships: " + ", ".join(
                    f"{r.source_name} -[{r.rel_type}]-> {r.target_name}"
                    for r in current_relationships
                )
                gleaning_prompt = self._build_gleaning_prompt(
                    text=text,
                    entity_types=types_str,
                    existing_info=existing_info,
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

            prev_entity_count = len(current_entities)
            prev_rel_count = len(current_relationships)
            if is_code_domain:
                relationships_payload = gleamed.get("relationships")
                cleaned_entities = _collect_code_entities(relationships_payload)
                cleaned_relationships = self._extract_code_relationships(
                    relationships_payload,
                    domain=domain,
                )
                if cleaned_entities or cleaned_relationships:
                    current_entities = cleaned_entities
                    current_relationships = cleaned_relationships
            else:
                for ent in gleamed.get("entities", []):
                    name = self._normalize_entity_name(ent.get("name", ""))
                    if name:
                        current_entities.append(ExtractedEntity(
                            name=name,
                            entity_type=ent.get("type", "UNKNOWN").upper(),
                            description=ent.get("description", ""),
                        ))

                current_entities = list({e.name: e for e in current_entities}.values())

                new_relationships = self._extract_relationships(
                    gleamed.get("relationships"),
                    rel_vocab=rel_vocab,
                    domain=domain,
                    is_code_domain=False,
                )

                current_relationships.extend(new_relationships)

                current_relationships = list(
                    {
                        (r.source_name, r.target_name, r.rel_type): r
                        for r in current_relationships
                    }.values()
                )

            if (
                len(current_entities) == prev_entity_count
                and len(current_relationships) == prev_rel_count
            ):
                break

        return ExtractionResult(
            entities=current_entities,
            relationships=current_relationships,
        )
