"""Domain-specific extraction configuration for Custom Graph RAG.

Each domain carries its own prompt guidance, relationship vocabulary,
and extraction behaviour flags.  The registry is the single source of
truth; ``extractor.py``, ``rel_types.py``, ``chunking.py``, and
``entity_resolver.py`` all read from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

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


# ---------------------------------------------------------------------------
# Prompt guidance (migrated from extractor.py static methods)
# ---------------------------------------------------------------------------

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
    "For non-code relationships, let rel_type labels capture the "
    "ideas, reasoning, causal or explanatory logic, and conceptual "
    "roles in the text. Reuse an existing rel_type from the current "
    "set when it fits; if none fits, create a concise new upper-snake "
    "rel_type that reflects the idea or logical role precisely."
)

_CODE_RELATIONSHIP_GUIDANCE = (
    "For code-domain relationships, write the description in short "
    "pseudo-code-flavored prose that captures execution semantics. "
    "Mention guards, conditions, branches, loops, sequencing, fan-out, "
    "or fan-in when the text supports them. Conditions may be written in "
    "plain English, but keep the style compact and code-like using cues "
    "such as IF, WHEN, THEN, ELSE, FOR EACH, WHILE, TRY/CATCH, RETURNS, "
    "SPLITS INTO, or MERGES WITH. Do not invent control flow that is not "
    "grounded in the text. Focus on extracting logic in terms of: "
    "what is happening, where it happens, why it happens, how it "
    "happens, and when it starts, ends, or switches path. When two "
    "symbols are contrasted, preserve both sides of the contrast "
    "explicitly instead of collapsing them into one generic summary. "
    "Always preserve exact explicit structural relationships that are "
    "directly present in the code or docs, such as imports, calls, "
    "returns, assignments, configuration, state updates, and fallback "
    "branches. In addition to those direct edges, extract richer logical "
    "relationships between the same entities when the text supports "
    "them. Choose concise rel_type labels only after you have identified "
    "the underlying logic, data flow, control flow, conditions, "
    "lifecycle, or purpose described in the text."
)

_GENERIC_REL_TYPE_GUIDANCE = (
    "Prefer an existing rel_type from this current set when it truly "
    "fits. If none fits, create a concise new UPPER_SNAKE rel_type "
    "that is semantically precise."
)

_NAMED_DOMAIN_REL_TYPE_GUIDANCE = (
    "Prefer an existing rel_type from this current set when it truly "
    "fits. If none fits, create a concise new UPPER_SNAKE rel_type "
    "that captures the underlying idea, logic, or conceptual role "
    "more precisely."
)

_CODE_REL_TYPE_GUIDANCE = (
    "Do not optimize for any fixed vocabulary. Derive concise "
    "UPPER_SNAKE rel_type labels from the logic actually present in "
    "the text. Preserve exact direct relationships when they are "
    "explicit, for example calls, imports, returns, assigns, "
    "updates, or configures. Add richer rel_type labels only when "
    "they capture additional supported logic such as purpose, "
    "condition, contrast, lifecycle, task fit, or fallback."
)

# ---------------------------------------------------------------------------
# Relationship vocabularies (moved from rel_types.py)
# ---------------------------------------------------------------------------

_REL_TYPES_GENERAL: Final[list[str]] = [
    "RELATES_TO",
    "DEFINES",
    "DESCRIBES",
    "EXPLAINS",
    "MENTIONED_IN",
    "QUOTES",
    "CITES",
    "IS_AN_EXAMPLE_OF",
    "IS_ANALOGY_OF",
    "INTRODUCES",
    "PREDICTS",
    "PROPOSES",
    "ARGUES",
    "CLAIMS",
    "CONCLUDES",
    "RECOMMENDS",
    "CAUSES",
    "LEADS_TO",
    "RESULTS_IN",
    "INFLUENCES",
    "DEPENDS_ON",
    "PRECEDES",
    "FOLLOWS",
    "PROVIDES_EVIDENCE_FOR",
    "JUSTIFIES",
    "CHALLENGES",
    "REFUTES",
    "CRITIQUES",
    "QUALIFIES",
    "LIMITS",
    "COMPARES",
    "CATEGORIZES",
    "CLASSIFIES",
    "GENERALIZES",
    "SPECIALIZES",
    "ILLUSTRATES",
    "DEMONSTRATES",
    "MOTIVATES",
    "SUMMARIZES",
    "SYNTHESIZES",
    "DERIVES",
    "PROVES",
    "ASSUMES",
    "IMPLIES",
    "HISTORICAL_CONTEXT_FOR",
    "BACKGROUND_FOR",
    "APPLICATION_OF",
    "EXTENSION_OF",
    "AUTHORED_BY",
    "ABOUT",
    "TARGETS",
    "INFLUENCED_BY",
    "CONTAINS",
    "PART_OF",
    "HAS_CHAPTER",
    "HAS_SECTION",
    "FEATURES",
    "CHARACTERIZES",
    "OCCURS_IN",
    "CONTRASTS_WITH",
    "SUPPORTS",
    "ELABORATES",
]

_REL_TYPES_CODE: Final[list[str]] = [
    "RELATES_TO",
    "CALLS",
    "USES",
    "IMPORTS",
    "DEFINES",
    "IMPLEMENTS",
    "EXTENDS",
    "DEPENDS_ON",
    "RAISES",
    "CATCHES",
    "READS",
    "WRITES",
    "RETURNS",
    "YIELDS",
    "LOOPS_OVER",
    "DECORATES",
    "GUARDS",
    "ASSIGNS",
    "INITIALIZES",
    "MUTATES",
    "VALIDATES",
    "FILTERS",
    "MAPS",
    "REDUCES",
    "TRANSFORMS",
    "SERIALIZES",
    "DESERIALIZES",
    "PARSES",
    "FORMATS",
    "LOGS",
    "CONFIGURES",
    "AUTHENTICATES",
    "AUTHORIZES",
    "SENDS",
    "RECEIVES",
    "SUBSCRIBES_TO",
    "EMITS",
    "AWAITS",
    "SPAWNS",
    "SCHEDULES",
    "RETRIES",
    "TIMES_OUT",
    "LOCKS",
    "UNLOCKS",
    "ALLOCATES",
    "RELEASES",
    "OPENS",
    "CLOSES",
    "CONNECTS_TO",
    "QUERIES",
    "UPDATES",
    "DELETES",
    "CREATES",
    "MOCKS",
    "ASSERTS",
    "OVERRIDES",
    "ALIAS_OF",
    "CONTAINS",
    "EXPOSES",
    "HIDES",
    "DEPRECATED_BY",
    "REPLACES",
    "IS_INSTANCE_OF",
    "TESTS",
    "DOCUMENTS",
    "REFERENCES",
]

_REL_TYPES_PERSONAL: Final[list[str]] = [
    "RELATES_TO",
    "REMEMBERS",
    "MENTIONED",
    "EXPLAINS_TO",
    "PREFERS",
    "DISLIKES",
    "AVOIDS",
    "FAVORITES",
    "BELIEVES",
    "OPINION_ABOUT",
    "VALUES",
    "WANTS",
    "GOAL",
    "PLANS",
    "CONSIDERING",
    "DECIDED",
    "ABANDONED",
    "TRIED",
    "LEARNED",
    "SUCCEEDED_AT",
    "FAILED_AT",
    "OWNS",
    "USES",
    "SUBSCRIBES_TO",
    "WORKS_AT",
    "STUDIES",
    "LIVES_IN",
    "VISITED",
    "FROM",
    "KNOWS",
    "RELATED_TO",
    "MET",
    "COLLABORATES_WITH",
    "SCHEDULED",
    "ATTENDED",
    "COMMITTED_TO",
    "NEEDS",
    "BLOCKED_BY",
    "CONCERNED_ABOUT",
    "CHANGED_MIND_ABOUT",
    "UPGRADED_FROM",
    "REPLACED",
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    "general": DomainConfig(
        name="general",
        rel_types=_REL_TYPES_GENERAL,
        entity_guidance=_GENERIC_ENTITY_GUIDANCE,
        relationship_guidance=_GENERIC_RELATIONSHIP_GUIDANCE,
        rel_type_guidance=_GENERIC_REL_TYPE_GUIDANCE,
    ),
    "books": DomainConfig(
        name="books",
        rel_types=_REL_TYPES_GENERAL,
        entity_guidance=_GENERIC_ENTITY_GUIDANCE,
        relationship_guidance=_GENERIC_RELATIONSHIP_GUIDANCE,
        rel_type_guidance=_NAMED_DOMAIN_REL_TYPE_GUIDANCE,
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
    "personal": DomainConfig(
        name="personal",
        rel_types=_REL_TYPES_PERSONAL,
        entity_guidance=_GENERIC_ENTITY_GUIDANCE,
        relationship_guidance=_GENERIC_RELATIONSHIP_GUIDANCE,
        rel_type_guidance=_NAMED_DOMAIN_REL_TYPE_GUIDANCE,
    ),
}

_ALL_DOMAIN_NAMES: tuple[str, ...] = tuple(DOMAIN_CONFIGS.keys())
ALL_DOMAIN_NAMES = _ALL_DOMAIN_NAMES  # public alias

_DEFAULT_CONFIG: DomainConfig = DOMAIN_CONFIGS["general"]


def get_domain_config(domain: str | None) -> DomainConfig:
    """Return the config for *domain*, falling back to ``general``."""
    if not domain:
        return _DEFAULT_CONFIG
    return DOMAIN_CONFIGS.get(domain, _DEFAULT_CONFIG)


def register_domain(config: DomainConfig) -> None:
    """Register a new domain configuration at runtime.

    Call during application startup to add custom extraction profiles
    without modifying the core code.
    """
    DOMAIN_CONFIGS[config.name] = config


# ---------------------------------------------------------------------------
# Content-based domain detection
# ---------------------------------------------------------------------------

_CODE_PATTERNS = [
    r"^\s*from\s+\S+\s+import\s+",
    r"^\s*def\s+\w+\s*\(",
    r"^\s*class\s+\w+\s*[:(]",
    r"^\s*package\s+\w+",
    r"^\s*func\s+\w+\s*\(",
    r"^\s*use\s+\S+;",
    r"^\s*fn\s+\w+\s*\(",
    r"^\s*impl\b",
    r"^\s*import\s+[\w.*]+\s*;",
    r"^\s*public\s+class\b",
    r"^\s*private\s+\w+",
    r"^\s*#include\s+[<\"]",
    r"std::",
    r"^\s*template\s*<",
    r"^\s*using\s+\S+;",
    r"^\s*namespace\s+\S+",
    r"^\s*function\s+\w+\s*\(",
    r"^\s*const\s+\w+\s*=",
    r"^\s*export\s+default\b",
    r"^\s*<\?php",
    r"->",
    r"^\s*module\s+\w+",
    r"^\s*end\s*$",
    r"^\s*fun\s+\w+\s*\(",
    r"^\s*val\s+\w+\s*=",
    r"^\s*data\s+class\b",
    r"^\s*interface\s+\w+",
    r"^\s*enum\s+\w+",
    r"=>\s*{",
]

_CODE_RE = re.compile("|".join(_CODE_PATTERNS), re.MULTILINE)


def detect_domain(text: str) -> str:
    """Heuristically detect the content domain from raw text.

    Returns ``"code"`` when code patterns are found, ``"general"``
    otherwise.  Extend with additional heuristics (e.g., personal notes,
    medical, legal) as new domains are registered.
    """
    if _CODE_RE.search(text):
        return "code"
    return "general"
