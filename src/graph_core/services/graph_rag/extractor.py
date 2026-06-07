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
from graph_core.models.rel_types import (
    DEFAULT_REL_TYPE,
    normalize_rel_type,
    rel_types_for_domain,
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


_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
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
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "rel_type": {
                        "type": "array",
                        "items": {
                            "type": "object",
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
                "required": ["source", "target", "rel_type"],
            },
        },
    },
    "required": ["entities", "relationships"],
}


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
   - "rel_type" is a list of one or more objects, each with:
       name        (string, guided by: {rel_type_vocab})
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
   and significance of the connection.
6a. {domain_relationship_guidance}
7. Each relationship's "rel_type" must follow this guidance:
   {rel_type_vocab}. {domain_rel_type_guidance} DEFAULT TO A SINGLE rel_type PER PAIR; only
   emit multiple when the text genuinely supports distinct,
   simultaneously-true roles and you can write a meaningfully
   different description for each. Pick the best single fit.
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
        normalized = name.strip().title()
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
        domain vocab. For ``code`` and ``books`` we allow new rel_types
        to be introduced when the existing set is not expressive enough;
        for other domains, unknown names fall back to ``DEFAULT_REL_TYPE``.

        Returns at least one entry; the fallback is
        ``[{"name": DEFAULT_REL_TYPE, "description": ..., "keywords": ..., "weight": ...}]``
        when the LLM emitted nothing usable.
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
        allow_new_rel_types = domain in {"code", "books"}
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
                if allow_new_rel_types:
                    logger.info(
                        "LLM emitted new rel_type=%r outside active %s vocab; accepting normalized type %s",
                        raw_name,
                        domain,
                        name,
                    )
                else:
                    logger.warning(
                        "LLM emitted rel_type=%r not in active domain vocab; "
                        "falling back to %s. Add it to DOMAIN_VOCAB if intentional.",
                        raw_name,
                        DEFAULT_REL_TYPE,
                    )
                    name = DEFAULT_REL_TYPE
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

        if not out:
            out.append({
                "rel_type": DEFAULT_REL_TYPE,
                "description": fallback_description,
                "keywords": list(fallback_keywords),
                "weight": fallback_weight,
            })
        return out

    @staticmethod
    def _domain_relationship_guidance(domain: str | None) -> str:
        if domain != "code":
            return (
                "For non-code relationships, let rel_type labels capture the "
                "ideas, reasoning, causal or explanatory logic, and conceptual "
                "roles in the text. Reuse an existing rel_type from the current "
                "set when it fits; if none fits, create a concise new upper-snake "
                "rel_type that reflects the idea or logical role precisely."
            )
        return (
            "For code-domain relationships, write the description in short "
            "pseudo-code-flavored prose that captures execution semantics. "
            "Mention guards, conditions, branches, loops, sequencing, fan-out, "
            "or fan-in when the text supports them. Conditions may be written in "
            "plain English, but keep the style compact and code-like using cues "
            "such as IF, WHEN, THEN, ELSE, FOR EACH, WHILE, TRY/CATCH, RETURNS, "
            "SPLITS INTO, or MERGES WITH. Do not invent control flow that is not "
            "grounded in the text. When choosing rel_type labels, prefer the "
            "current set if one is relevant; otherwise create a concise new "
            "upper-snake rel_type that captures the architecture, data flow, or "
            "execution role more precisely."
        )

    @staticmethod
    def _domain_rel_type_guidance(domain: str | None) -> str:
        if domain == "code":
            return (
                "Prefer an existing rel_type from this current set when it truly "
                "fits. If none fits, create a concise new UPPER_SNAKE rel_type "
                "that captures the architecture, data flow, or execution role "
                "more precisely."
            )
        if domain is not None:
            return (
                "Prefer an existing rel_type from this current set when it truly "
                "fits. If none fits, create a concise new UPPER_SNAKE rel_type "
                "that captures the underlying idea, logic, or conceptual role "
                "more precisely."
            )
        return (
            "Do not invent names outside this current set; any unknown rel_type "
            "will fall back to RELATES_TO."
        )

    @staticmethod
    def _domain_entity_guidance(domain: str | None) -> str:
        if domain != "code":
            return (
                "Use concise title case names and keep naming consistent across "
                "the extraction."
            )
        return (
            "For code-domain entities, prefer the exact symbol names that appear "
            "in the source whenever the entity is a concrete class, function, "
            "method, module, variable, exception, or config symbol. Preserve the "
            "source casing and punctuation style for those symbols instead of "
            "rewriting them into generic title case. Use broader natural-language "
            "names only for genuinely higher-level code concepts that are not "
            "named symbols in the source."
        )

    @staticmethod
    def _build_extraction_prompt(
        text: str,
        entity_types: str,
        rel_type_vocab: list[str],
        domain: str | None,
    ) -> str:
        return (
            _EXTRACTION_SYSTEM_PROMPT.format(
                entity_types=entity_types,
                rel_type_vocab=", ".join(rel_type_vocab),
                domain_entity_guidance=LLMGraphExtractor._domain_entity_guidance(domain),
                domain_rel_type_guidance=LLMGraphExtractor._domain_rel_type_guidance(domain),
                domain_relationship_guidance=LLMGraphExtractor._domain_relationship_guidance(domain),
            )
            + "\n\n"
            + _EXTRACTION_USER_PROMPT.format(text=text)
        )

    @staticmethod
    def _build_gleaning_prompt(
        text: str,
        entity_types: str,
        existing_info: str,
        rel_type_vocab: list[str],
        domain: str | None,
    ) -> str:
        return (
            _GLEANING_SYSTEM_PROMPT.format(
                entity_types=entity_types,
                rel_type_vocab=", ".join(rel_type_vocab),
                domain_entity_guidance=LLMGraphExtractor._domain_entity_guidance(domain),
                domain_rel_type_guidance=LLMGraphExtractor._domain_rel_type_guidance(domain),
                domain_relationship_guidance=LLMGraphExtractor._domain_relationship_guidance(domain),
            )
            + "\n\n"
            + _GLEANING_USER_PROMPT.format(existing_info=existing_info, text=text)
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
        rel_vocab = rel_types_for_domain(domain)
        prompt = self._build_extraction_prompt(
            text=text,
            entity_types=types_str,
            rel_type_vocab=rel_vocab,
            domain=domain,
        )

        try:
            result = await self._llm.structured_extract(
                prompt=prompt,
                schema=_EXTRACTION_SCHEMA,
            )
        except Exception as e:
            logger.warning("LLM extraction failed, retrying once: %s", e)
            try:
                result = await self._llm.structured_extract(
                    prompt=prompt,
                    schema=_EXTRACTION_SCHEMA,
                )
            except Exception as e2:
                logger.error("LLM extraction failed after retry: %s", e2)
                return ExtractionResult(entities=[], relationships=[])

        entities = []
        for ent in result.get("entities", []):
            name = self._normalize_entity_name(ent.get("name", ""))
            if name:
                entities.append(ExtractedEntity(
                    name=name,
                    entity_type=ent.get("type", "UNKNOWN").upper(),
                    description=ent.get("description", ""),
                ))

        relationships = []
        for rel in result.get("relationships", []):
            source = self._normalize_entity_name(rel.get("source", ""))
            target = self._normalize_entity_name(rel.get("target", ""))
            if not (source and target):
                continue
            rel_description = rel.get("description", "")
            rel_keywords_str = rel.get("keywords", [])
            if isinstance(rel_keywords_str, str):
                rel_keywords_str = [
                    k.strip()
                    for k in rel_keywords_str.split(",")
                    if k.strip()
                ]
            rel_keywords_list = rel_keywords_str if isinstance(rel_keywords_str, list) else []
            try:
                rel_weight = float(rel.get("weight", 1.0))
                rel_weight = max(0.0, min(1.0, rel_weight))
            except (ValueError, TypeError):
                rel_weight = 1.0
            for entry in self._coerce_rel_type_entries(
                rel.get("rel_type"),
                fallback_description=rel_description,
                fallback_keywords=rel_keywords_list,
                fallback_weight=rel_weight,
                vocab=rel_vocab,
                domain=domain,
            ):
                relationships.append(ExtractedRelationship(
                    source_name=source,
                    target_name=target,
                    description=entry["description"],
                    keywords=list(entry["keywords"]),
                    weight=entry["weight"],
                    rel_type=entry["rel_type"],
                ))

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

        rel_vocab = rel_types_for_domain(domain)
        for gleaning_pass in range(max_gleaning):
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

            types_str = ", ".join(entity_types) if entity_types else "general"
            gleaning_prompt = self._build_gleaning_prompt(
                text=text,
                entity_types=types_str,
                existing_info=existing_info,
                rel_type_vocab=rel_vocab,
                domain=domain,
            )

            try:
                gleamed = await self._llm.structured_extract(
                    prompt=gleaning_prompt,
                    schema=_EXTRACTION_SCHEMA,
                )
            except Exception as e:
                logger.warning("Gleaning failed: %s", e)
                break

            prev_entity_count = len(current_entities)
            prev_rel_count = len(current_relationships)

            for ent in gleamed.get("entities", []):
                name = self._normalize_entity_name(ent.get("name", ""))
                if name:
                    current_entities.append(ExtractedEntity(
                        name=name,
                        entity_type=ent.get("type", "UNKNOWN").upper(),
                        description=ent.get("description", ""),
                    ))

            current_entities = list({e.name: e for e in current_entities}.values())

            for rel in gleamed.get("relationships", []):
                source = self._normalize_entity_name(rel.get("source", ""))
                target = self._normalize_entity_name(rel.get("target", ""))
                if not (source and target):
                    continue
                rel_description = rel.get("description", "")
                rel_keywords_str = rel.get("keywords", [])
                if isinstance(rel_keywords_str, str):
                    rel_keywords_str = [
                        k.strip()
                        for k in rel_keywords_str.split(",")
                        if k.strip()
                    ]
                rel_keywords_list = (
                    rel_keywords_str if isinstance(rel_keywords_str, list) else []
                )
                try:
                    rel_weight = max(0.0, min(1.0, float(rel.get("weight", 1.0))))
                except (ValueError, TypeError):
                    rel_weight = 1.0
                for entry in self._coerce_rel_type_entries(
                    rel.get("rel_type"),
                    fallback_description=rel_description,
                    fallback_keywords=rel_keywords_list,
                    fallback_weight=rel_weight,
                    vocab=rel_vocab,
                    domain=domain,
                ):
                    current_relationships.append(ExtractedRelationship(
                        source_name=source,
                        target_name=target,
                        description=entry["description"],
                        keywords=list(entry["keywords"]),
                        weight=entry["weight"],
                        rel_type=entry["rel_type"],
                    ))

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
