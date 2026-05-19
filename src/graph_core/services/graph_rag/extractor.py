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
                },
                "required": ["source", "target", "description"],
            },
        },
    },
    "required": ["entities", "relationships"],
}


_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph extractor. Given a text passage, extract all entities and their relationships.

Entity types to look for: {entity_types}

Return structured JSON with two arrays: "entities" and "relationships".
Each entity has: name, type, description.
Each relationship has: source, target, description, keywords (array), weight (0-1).

Only extract entities that are meaningfully mentioned in the text. Do not extract generic terms.
"""


_EXTRACTION_USER_PROMPT = """Extract entities and relationships from this text:

{text}
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

    async def extract(
        self,
        text: str,
        entity_types: list[str] | None = None,
    ) -> ExtractionResult:
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in self._cache:
            logger.debug("Cache hit for chunk hash=%s", content_hash[:8])
            return self._cache[content_hash]

        types_str = ", ".join(entity_types) if entity_types else "general"
        prompt = _EXTRACTION_USER_PROMPT.format(text=text)

        try:
            result = await self._llm.structured_extract(
                prompt=_EXTRACTION_SYSTEM_PROMPT.format(entity_types=types_str) + "\n\n" + prompt,
                schema=_EXTRACTION_SCHEMA,
            )
        except Exception as e:
            logger.warning("LLM extraction failed, retrying once: %s", e)
            try:
                result = await self._llm.structured_extract(
                    prompt=_EXTRACTION_SYSTEM_PROMPT.format(entity_types=types_str) + "\n\n" + prompt,
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
                    keywords_str = [k.strip() for k in keywords_str.split(",") if k.strip()]
                weight = rel.get("weight", 1.0)
                try:
                    weight = float(weight)
                    weight = max(0.0, min(1.0, weight))
                except (ValueError, TypeError):
                    weight = 1.0

                relationships.append(ExtractedRelationship(
                    source_name=source,
                    target_name=target,
                    description=rel.get("description", ""),
                    keywords=keywords_str if isinstance(keywords_str, list) else [],
                    weight=weight,
                ))

        extraction_result = ExtractionResult(entities=entities, relationships=relationships)
        self._cache[content_hash] = extraction_result

        logger.info("Extracted %d entities, %d relationships", len(entities), len(relationships))
        return extraction_result

    async def extract_with_gleaning(
        self,
        text: str,
        entity_types: list[str] | None = None,
        max_gleaning: int = 1,
    ) -> ExtractionResult:
        result = await self.extract(text, entity_types)
        current_entities = list(result.entities)
        current_relationships = list(result.relationships)

        if not current_entities and not current_relationships:
            return result

        for gleaning_pass in range(max_gleaning):
            existing_info = "Already extracted:\n"
            existing_info += "Entities: " + ", ".join(e.name for e in current_entities) + "\n"
            existing_info += "Relationships: " + ", ".join(
                f"{r.source_name} -> {r.target_name}" for r in current_relationships
            )

            types_str = ", ".join(entity_types) if entity_types else "general"
            gleaning_prompt = (
                _EXTRACTION_SYSTEM_PROMPT.format(entity_types=types_str)
                + "\n\nFind ADDITIONAL entities and relationships NOT already extracted.\n\n"
                + existing_info + "\n\n"
                + _EXTRACTION_USER_PROMPT.format(text=text)
            )

            try:
                gleamed = await self._llm.structured_extract(prompt=gleaning_prompt, schema=_EXTRACTION_SCHEMA)
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
                        keywords_str = [k.strip() for k in keywords_str.split(",") if k.strip()]
                    weight = rel.get("weight", 1.0)
                    try:
                        weight = max(0.0, min(1.0, float(weight)))
                    except (ValueError, TypeError):
                        weight = 1.0

                    current_relationships.append(ExtractedRelationship(
                        source_name=source,
                        target_name=target,
                        description=rel.get("description", ""),
                        keywords=keywords_str if isinstance(keywords_str, list) else [],
                        weight=weight,
                    ))

            current_relationships = list({f"{r.source_name}__{r.target_name}": r for r in current_relationships}.values())

            if len(current_entities) == prev_entity_count and len(current_relationships) == prev_rel_count:
                break

        return ExtractionResult(entities=current_entities, relationships=current_relationships)
