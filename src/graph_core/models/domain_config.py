"""Domain-specific extraction configuration for Custom Graph RAG.

An LLM classifies document content, generates tailored extraction guidance,
and returns a `DomainConfig` — no hardcoded regex or domain lists needed.
The registry still holds well-known fallbacks (`general`, `code`) for when
the LLM is unavailable or the caller provides an explicit domain name.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainConfig:
    """All extraction parameters for one content domain."""

    name: str
    rel_types: list[str]
    entity_guidance: str
    relationship_guidance: str
    rel_type_guidance: str
    use_ast_chunking: bool = False
    requires_exact_resolution: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainConfig:
        return cls(
            name=data["name"],
            rel_types=data.get("rel_types", list(DOMAIN_CONFIGS["general"].rel_types)),
            entity_guidance=data.get("entity_guidance", ""),
            relationship_guidance=data.get("relationship_guidance", ""),
            rel_type_guidance=data.get("rel_type_guidance", ""),
            use_ast_chunking=data.get("use_ast_chunking", False),
            requires_exact_resolution=data.get("requires_exact_resolution", False),
        )


# ---------------------------------------------------------------------------
# Static fallback configs (LLM unavailable / explicit name match)
# ---------------------------------------------------------------------------

_REL_TYPES_GENERAL: list[str] = [
    "RELATES_TO", "DEFINES", "DESCRIBES", "EXPLAINS", "MENTIONED_IN",
    "QUOTES", "CITES", "IS_AN_EXAMPLE_OF", "IS_ANALOGY_OF", "INTRODUCES",
    "PREDICTS", "PROPOSES", "ARGUES", "CLAIMS", "CONCLUDES", "RECOMMENDS",
    "CAUSES", "LEADS_TO", "RESULTS_IN", "INFLUENCES", "DEPENDS_ON",
    "PRECEDES", "FOLLOWS", "PROVIDES_EVIDENCE_FOR", "JUSTIFIES",
    "CHALLENGES", "REFUTES", "CRITIQUES", "QUALIFIES", "LIMITS",
    "COMPARES", "CATEGORIZES", "CLASSIFIES", "GENERALIZES", "SPECIALIZES",
    "ILLUSTRATES", "DEMONSTRATES", "MOTIVATES", "SUMMARIZES", "SYNTHESIZES",
    "DERIVES", "PROVES", "ASSUMES", "IMPLIES", "HISTORICAL_CONTEXT_FOR",
    "BACKGROUND_FOR", "APPLICATION_OF", "EXTENSION_OF", "AUTHORED_BY",
    "ABOUT", "TARGETS", "INFLUENCED_BY", "CONTAINS", "PART_OF",
    "HAS_CHAPTER", "HAS_SECTION", "FEATURES", "CHARACTERIZES",
    "OCCURS_IN", "CONTRASTS_WITH", "SUPPORTS", "ELABORATES",
]

_REL_TYPES_CODE: list[str] = [
    "RELATES_TO", "CALLS", "USES", "IMPORTS", "DEFINES", "IMPLEMENTS",
    "EXTENDS", "DEPENDS_ON", "RAISES", "CATCHES", "READS", "WRITES",
    "RETURNS", "YIELDS", "LOOPS_OVER", "DECORATES", "GUARDS", "ASSIGNS",
    "INITIALIZES", "MUTATES", "VALIDATES", "FILTERS", "MAPS", "REDUCES",
    "TRANSFORMS", "SERIALIZES", "DESERIALIZES", "PARSES", "FORMATS",
    "LOGS", "CONFIGURES", "AUTHENTICATES", "AUTHORIZES", "SENDS",
    "RECEIVES", "SUBSCRIBES_TO", "EMITS", "AWAITS", "SPAWNS", "SCHEDULES",
    "RETRIES", "TIMES_OUT", "LOCKS", "UNLOCKS", "ALLOCATES", "RELEASES",
    "OPENS", "CLOSES", "CONNECTS_TO", "QUERIES", "UPDATES", "DELETES",
    "CREATES", "MOCKS", "ASSERTS", "OVERRIDES", "ALIAS_OF", "CONTAINS",
    "EXPOSES", "HIDES", "DEPRECATED_BY", "REPLACES", "IS_INSTANCE_OF",
    "TESTS", "DOCUMENTS", "REFERENCES",
]

_GENERIC_ENTITY_GUIDANCE = (
    "Use concise title case names and keep naming consistent across "
    "the extraction."
)

_CODE_ENTITY_GUIDANCE = (
    "For code-domain entities, prefer the exact symbol names that appear "
    "in the source whenever the entity is a concrete class, function, "
    "method, module, variable, exception, or config symbol. Preserve the "
    "source casing and punctuation style for those symbols instead of "
    "rewriting them into generic title case. Use broader natural-language "
    "names only for genuinely higher-level code concepts that are "
    "explicitly named in the source text or documentation. Do not invent "
    "new conceptual node names just to represent logic that can be "
    "captured as a relationship between existing entities. Prefer to put "
    "what/where/why/how/when semantics into relationship descriptions and "
    "rel_type labels unless the concept itself is explicitly named."
)

_GENERIC_RELATIONSHIP_GUIDANCE = (
    "Let rel_type labels capture the ideas, reasoning, causal or "
    "explanatory logic, and conceptual roles in the text. Reuse an "
    "existing rel_type from the current set when it fits; if none fits, "
    "create a concise new upper-snake rel_type that reflects the idea "
    "or logical role precisely."
)

_CODE_RELATIONSHIP_GUIDANCE = (
    "Write the description in short pseudo-code-flavored prose that "
    "captures execution semantics. Mention guards, conditions, branches, "
    "loops, sequencing, fan-out, or fan-in when the text supports them. "
    "Use cues such as IF, WHEN, THEN, ELSE, FOR EACH, WHILE, TRY/CATCH, "
    "RETURNS, SPLITS INTO, or MERGES WITH. Do not invent control flow "
    "that is not grounded in the text. Focus on extracting logic in "
    "terms of: what is happening, where it happens, why it happens, "
    "how it happens, and when it starts, ends, or switches path. "
    "Always preserve exact explicit structural relationships such as "
    "imports, calls, returns, assignments, configuration, state updates, "
    "and fallback branches."
)

_GENERIC_REL_TYPE_GUIDANCE = (
    "Prefer an existing rel_type from this current set when it truly "
    "fits. If none fits, create a concise new UPPER_SNAKE rel_type "
    "that is semantically precise."
)

_CODE_REL_TYPE_GUIDANCE = (
    "Do not optimize for any fixed vocabulary. Derive concise "
    "UPPER_SNAKE rel_type labels from the logic actually present in "
    "the text. Preserve exact direct relationships when they are "
    "explicit, for example calls, imports, returns, assigns, "
    "updates, or configures."
)

DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    "general": DomainConfig(
        name="general",
        rel_types=_REL_TYPES_GENERAL,
        entity_guidance=_GENERIC_ENTITY_GUIDANCE,
        relationship_guidance=_GENERIC_RELATIONSHIP_GUIDANCE,
        rel_type_guidance=_GENERIC_REL_TYPE_GUIDANCE,
    ),
    "code": DomainConfig(
        name="code",
        rel_types=_REL_TYPES_CODE,
        entity_guidance=_CODE_ENTITY_GUIDANCE,
        relationship_guidance=_CODE_RELATIONSHIP_GUIDANCE,
        rel_type_guidance=_CODE_REL_TYPE_GUIDANCE,
        use_ast_chunking=True,
        requires_exact_resolution=True,
    ),
}

_ALL_DOMAIN_NAMES: tuple[str, ...] = tuple(DOMAIN_CONFIGS.keys())
ALL_DOMAIN_NAMES = _ALL_DOMAIN_NAMES
_DEFAULT_CONFIG: DomainConfig = DOMAIN_CONFIGS["general"]


def get_domain_config(domain: str | None) -> DomainConfig:
    """Return the config for *domain*, falling back to ``general``."""
    if not domain:
        return _DEFAULT_CONFIG
    return DOMAIN_CONFIGS.get(domain, _DEFAULT_CONFIG)


def register_domain(config: DomainConfig) -> None:
    """Register a new domain configuration at runtime."""
    DOMAIN_CONFIGS[config.name] = config


# ---------------------------------------------------------------------------
# LLM-based domain classification
# ---------------------------------------------------------------------------

_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "title": "domain_classification",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "domain_name": {
            "type": "string",
            "description": "A short lowercase identifier for the content "
                           "domain, e.g. 'stocks', 'job_posting', 'code', "
                           "'medical', 'legal', 'academic_paper', 'personal'.",
        },
        "entity_guidance": {
            "type": "string",
            "description": "Tailored instructions for how to name and "
                           "describe entities in this type of content. "
                           "Be specific to the domain.",
        },
        "relationship_guidance": {
            "type": "string",
            "description": "Tailored instructions for how to write "
                           "relationship descriptions in this type of "
                           "content. Describe the style, what to capture, "
                           "and what to omit.",
        },
        "rel_type_guidance": {
            "type": "string",
            "description": "Instructions for choosing or inventing "
                           "UPPER_SNAKE rel_type labels for this domain.",
        },
        "suggested_rel_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of 20-50 UPPER_SNAKE rel_type labels "
                           "that are most relevant for this content domain. "
                           "Always include RELATES_TO as the first entry.",
        },
        "use_ast_chunking": {
            "type": "boolean",
            "description": "True if this content should be chunked with "
                           "AST-aware code splitting (only for source code).",
        },
        "requires_exact_resolution": {
            "type": "boolean",
            "description": "True if entity names must match exactly (no "
                           "fuzzy matching) — typically only for source code "
                           "symbols.",
        },
    },
    "required": [
        "domain_name",
        "entity_guidance",
        "relationship_guidance",
        "rel_type_guidance",
        "suggested_rel_types",
    ],
}

_CLASSIFICATION_PROMPT = """\
You are a Knowledge Graph domain specialist. Your job is to analyze the \
content type of the provided text and generate tailored extraction \
instructions for building a knowledge graph from it.

Analyze what kind of document this is — its subject matter, genre, \
structure, and the kind of entities and relationships it is likely to contain.

Then produce domain-specific guidance that will be inserted into the \
extraction system prompt used by an LLM to extract entities and \
relationships from chunks of this document.

Return your answer as structured JSON.

Text to analyze:
{text}
"""


async def classify_document(
    llm: Any,
    text: str,
) -> DomainConfig:
    """Ask the LLM to classify document content and generate extraction guidance.

    Samples the first 4000 characters for classification (covers the \
    document's genre, structure, and subject without wasting tokens).

    Falls back to ``general`` config on any error.
    """
    sample = text[:4000] if len(text) > 4000 else text

    try:
        result = await llm.structured_extract(
            prompt=_CLASSIFICATION_PROMPT.format(text=sample),
            schema=_CLASSIFICATION_SCHEMA,
        )
    except Exception as e:
        logger.warning(
            "LLM domain classification failed, falling back to general: %s", e
        )
        return _DEFAULT_CONFIG

    try:
        domain_name = result.get("domain_name", "general").lower().strip()
        if not domain_name:
            domain_name = "general"

        rel_types = result.get("suggested_rel_types", [])
        if not rel_types or not isinstance(rel_types, list):
            rel_types = list(_DEFAULT_CONFIG.rel_types)
        # Ensure RELATES_TO is always present
        if "RELATES_TO" not in rel_types:
            rel_types.insert(0, "RELATES_TO")

        cfg = DomainConfig(
            name=domain_name,
            rel_types=rel_types,
            entity_guidance=result.get("entity_guidance", _DEFAULT_CONFIG.entity_guidance)
                or _DEFAULT_CONFIG.entity_guidance,
            relationship_guidance=result.get("relationship_guidance", _DEFAULT_CONFIG.relationship_guidance)
                or _DEFAULT_CONFIG.relationship_guidance,
            rel_type_guidance=result.get("rel_type_guidance", _DEFAULT_CONFIG.rel_type_guidance)
                or _DEFAULT_CONFIG.rel_type_guidance,
            use_ast_chunking=bool(result.get("use_ast_chunking", False)),
            requires_exact_resolution=bool(result.get("requires_exact_resolution", False)),
        )

        logger.info(
            "LLM classified document domain: %s (%d rel_types)",
            domain_name,
            len(rel_types),
        )
        return cfg

    except Exception as e:
        logger.error(
            "Failed to parse LLM domain classification result, "
            "falling back to general: %s", e
        )
        return _DEFAULT_CONFIG
