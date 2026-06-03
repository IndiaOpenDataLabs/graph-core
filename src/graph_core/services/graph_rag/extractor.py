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
                    "description": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "weight": {"type": "number"},
                    "rel_type": {"type": "string"},
                },
                "required": ["source", "target", "description"],
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
   - Use concise title case names and keep naming consistent across the extraction.
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
   - Each relationship must have: source, target, description, keywords, weight, rel_type.
   - "keywords" should be a compact list of relevant terms.
   - "weight" should be a float between 0 and 1 representing confidence or salience.
   - "rel_type" must be one of: {rel_type_vocab}. Pick the single best fit. Use
     "RELATES_TO" only if none of the specific types apply.
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
4. Keep naming consistent with the previously extracted entities.
5. Relationship descriptions must still explain the nature, context,
   and significance of the connection.
6. Each relationship's "rel_type" must be one of: {rel_type_vocab}.
   Pick the single best fit; use "RELATES_TO" only if none of the
   specific types apply.
7. Return only new or corrected items in the same JSON structure as
   the main extraction.
8. Only include items explicitly supported by the text.
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
    def _build_extraction_prompt(
        text: str, entity_types: str, rel_type_vocab: list[str]
    ) -> str:
        return (
            _EXTRACTION_SYSTEM_PROMPT.format(
                entity_types=entity_types, rel_type_vocab=", ".join(rel_type_vocab)
            )
            + "\n\n"
            + _EXTRACTION_USER_PROMPT.format(text=text)
        )

    @staticmethod
    def _build_gleaning_prompt(
        text: str, entity_types: str, existing_info: str, rel_type_vocab: list[str]
    ) -> str:
        return (
            _GLEANING_SYSTEM_PROMPT.format(
                entity_types=entity_types, rel_type_vocab=", ".join(rel_type_vocab)
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
            text=text, entity_types=types_str, rel_type_vocab=rel_vocab
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
            if source and target:
                keywords_str = rel.get("keywords", [])
                if isinstance(keywords_str, str):
                    keywords_str = [
                        k.strip()
                        for k in keywords_str.split(",")
                        if k.strip()
                    ]
                weight = rel.get("weight", 1.0)
                try:
                    weight = float(weight)
                    weight = max(0.0, min(1.0, weight))
                except (ValueError, TypeError):
                    weight = 1.0

                rel_type = normalize_rel_type(rel.get("rel_type"))

                relationships.append(ExtractedRelationship(
                    source_name=source,
                    target_name=target,
                    description=rel.get("description", ""),
                    keywords=keywords_str if isinstance(keywords_str, list) else [],
                    weight=weight,
                    rel_type=rel_type,
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
                if source and target:
                    keywords_str = rel.get("keywords", [])
                    if isinstance(keywords_str, str):
                        keywords_str = [
                            k.strip()
                            for k in keywords_str.split(",")
                            if k.strip()
                        ]
                    weight = rel.get("weight", 1.0)
                    try:
                        weight = max(0.0, min(1.0, float(weight)))
                    except (ValueError, TypeError):
                        weight = 1.0

                    rel_type = normalize_rel_type(rel.get("rel_type"))
                    current_relationships.append(ExtractedRelationship(
                        source_name=source,
                        target_name=target,
                        description=rel.get("description", ""),
                        keywords=keywords_str if isinstance(keywords_str, list) else [],
                        weight=weight,
                        rel_type=rel_type,
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
