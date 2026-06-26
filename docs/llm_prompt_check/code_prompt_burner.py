#!/usr/bin/env python3
"""Two-pass burner for replaying the code-domain extraction flow.

Pass 1 extracts a candidate graph from the target chunk.
Pass 2 reconstructs the logic from those candidates, removing noise and
adding anything missing so the final result is the cleaned graph.

The script reads a prompt bundle file containing a system prompt and a
target chunk separated by a marker line. The second-pass cleanup prompt is
constructed locally in this burner so the behavior stays here, not in the
extractor implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEPARATOR = "\n\n--- CHUNK ---\n\n"
DEFAULT_INPUT = ROOT / "docs" / "llm_prompt_check" / "code_prompt.txt"
DEFAULT_OUTPUT = ROOT / "docs" / "llm_prompt_check" / "code_response_two_pass.json"
DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF::UD-Q6_K_XL"
_GRAPH_SCHEMA_NAME = "code_graph_replay"

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_core.models.domain_config import CODE_REL_TYPE_TAXONOMY, get_domain_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
        help="Prompt bundle containing the system prompt and target chunk",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the combined two-pass response JSON",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name returned by /v1/models",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="HTTP timeout for each completion request",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_false",
        dest="json_mode",
        help="Disable structured JSON schema mode",
    )
    parser.set_defaults(json_mode=True)
    return parser.parse_args()


def _load_prompt_bundle(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8")
    if SEPARATOR not in content:
        raise ValueError(f"{path} does not contain the separator {SEPARATOR!r}")
    system_prompt, chunk = content.split(SEPARATOR, 1)
    return system_prompt.strip(), chunk.strip()


def _taxonomy_prompt() -> str:
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


def _build_graph_schema() -> dict[str, Any]:
    category_properties: dict[str, Any] = {}
    category_required: list[str] = []
    endpoint_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["name", "description"],
    }
    for category_name, rel_types in CODE_REL_TYPE_TAXONOMY:
        category_required.append(category_name)
        category_properties[category_name] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                rel_type.upper(): {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source": endpoint_schema,
                            "target": endpoint_schema,
                            "description": {"type": "string"},
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": [
                            "source",
                            "target",
                            "description",
                            "keywords",
                            "weight",
                        ],
                    },
                }
                for rel_type in rel_types
            },
            "required": [rel_type.upper() for rel_type in rel_types],
        }

    return {
        "title": _GRAPH_SCHEMA_NAME,
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


def _structured_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _GRAPH_SCHEMA_NAME,
            "schema": _build_graph_schema(),
            "strict": True,
        },
    }


def _build_reconstruction_system_prompt() -> str:
    cfg = get_domain_config("code")
    return f"""---Role---
You are a Knowledge Graph Specialist responsible for refining a first-pass
code extraction into a cleaner final graph summary.

---Instructions---
1. You will be given a first-pass relationship graph extracted from code.
2. Reconstruct the likely execution logic from those relationships.
3. Remove unnecessary, overly local, or purely syntactic relationship
   items that do not materially explain the code.
4. Add missing relationship items if they are needed to explain the logic
   more completely.
5. Prefer a compact final result that preserves the core execution flow,
   state changes, control flow, and API boundaries.
6. Keep only semantically central code symbols, concepts, and operations.
   Exclude ephemeral locals, loop counters, temporary accumulators, and
   helper builtins unless they are essential to the logic.
7. Relationship descriptions must be rich enough to understand what
   happens, when, why, and with what effect without looking at the code.
8. Use the fixed taxonomy below and keep the output structured by
   category and rel_type.
9. For every chunk, evaluate every rel_type, even when no evidence
   exists. Emit an empty array for unsupported rel_types.
10. Return the cleaned final relationship graph, not a diff.
11. Do not emit an entities section. The burner will derive entities from
    the relationship endpoints after parsing.
12. {cfg.relationship_guidance}

{_taxonomy_prompt()}
"""


def _build_first_pass_system_prompt(system_prompt: str) -> str:
    return system_prompt


def _strip_fences(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _iter_relationship_items(graph: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    relationships = graph.get("relationships", {})
    if not isinstance(relationships, dict):
        return items
    for rel_types in relationships.values():
        if not isinstance(rel_types, dict):
            continue
        for bucket in rel_types.values():
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                if isinstance(item, dict):
                    items.append(item)
    return items


def _derive_entities_from_relationships(graph: dict[str, Any]) -> list[dict[str, str]]:
    related: dict[str, list[str]] = {}
    original_case: dict[str, str] = {}
    for item in _iter_relationship_items(graph):
        for key in ("source", "target"):
            endpoint = item.get(key)
            if isinstance(endpoint, dict):
                raw_name = endpoint.get("name", "")
                raw_description = endpoint.get("description", "")
            else:
                raw_name = endpoint
                raw_description = ""
            if not isinstance(raw_name, str):
                continue
            name = raw_name.strip()
            if not name:
                continue
            norm = " ".join(name.split())
            lowered = norm.lower()
            original_case.setdefault(lowered, norm)
            related.setdefault(lowered, [])
            if isinstance(raw_description, str) and raw_description.strip():
                related[lowered].append(raw_description.strip())

    entities: list[dict[str, str]] = []
    for lowered, descriptions in sorted(
        related.items(),
        key=lambda x: original_case[x[0]].lower(),
    ):
        name = original_case[lowered]
        summary = descriptions[0] if descriptions else ""
        if len(summary) > 160:
            summary = summary[:157].rstrip() + "..."
        entities.append(
            {
                "name": name,
                "type": "code_object",
                "description": summary or "Concrete code object referenced by extracted relationships.",
            }
        )
    return entities


def _build_graph_with_derived_entities(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "entities": _derive_entities_from_relationships(graph),
        "relationships": graph.get("relationships", {}),
    }


def _parse_assistant_json(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Assistant content is not a string")
    cleaned = _strip_fences(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _extract_balanced_json(cleaned)
        if candidate is not None:
            return json.loads(candidate)
        raise


def _format_candidate_graph(graph: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Relationships:")
    relationships = graph.get("relationships", {})
    if isinstance(relationships, dict):
        for category_name, rel_types in relationships.items():
            lines.append(f"[{category_name}]")
            if isinstance(rel_types, dict):
                any_items = False
                for rel_type, items in rel_types.items():
                    if not isinstance(items, list) or not items:
                        continue
                    any_items = True
                    lines.append(f"  - {rel_type}:")
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        source = item.get("source", {})
                        target = item.get("target", {})
                        if isinstance(source, dict):
                            source_text = (
                                f"{source.get('name', '')}: "
                                f"{source.get('description', '')}"
                            )
                        else:
                            source_text = str(source)
                        if isinstance(target, dict):
                            target_text = (
                                f"{target.get('name', '')}: "
                                f"{target.get('description', '')}"
                            )
                        else:
                            target_text = str(target)
                        lines.append(
                            "    * "
                            f"{source_text} -> {target_text}: "
                            f"{item.get('description', '')} "
                            f"(keywords={', '.join(item.get('keywords', []) or [])}; "
                            f"weight={item.get('weight', '')})"
                        )
                if not any_items:
                    lines.append("  - (none)")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _post_chat_completion(
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
    messages: list[dict[str, str]],
    json_mode: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": messages,
    }
    if json_mode:
        body["response_format"] = _structured_response_format()

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.load(response)
    except Exception:
        if not json_mode:
            raise
        fallback_body = dict(body)
        fallback_body["response_format"] = {"type": "json_object"}
        fallback_request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(fallback_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(fallback_request, timeout=timeout_seconds) as response:
            return json.load(response)


def main() -> int:
    args = _parse_args()
    first_pass_system_prompt, chunk = _load_prompt_bundle(args.input_file)

    first_messages = [
        {"role": "system", "content": _build_first_pass_system_prompt(first_pass_system_prompt)},
        {"role": "user", "content": chunk},
    ]
    first_payload = _post_chat_completion(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        messages=first_messages,
        json_mode=args.json_mode,
    )
    first_graph = _parse_assistant_json(first_payload)
    first_output_graph = _build_graph_with_derived_entities(first_graph)

    second_messages = [
        {"role": "system", "content": _build_reconstruction_system_prompt()},
        {
            "role": "user",
            "content": (
                "---Candidate Graph---\n"
                f"{_format_candidate_graph(first_graph)}\n\n"
                "---Source Code---\n"
                f"{chunk}"
            ),
        },
    ]
    second_payload = _post_chat_completion(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        messages=second_messages,
        json_mode=args.json_mode,
    )
    second_graph = _parse_assistant_json(second_payload)
    second_output_graph = _build_graph_with_derived_entities(second_graph)

    result = {
        "input_file": str(args.input_file),
        "model": args.model,
        "first_pass": {
            "raw": first_payload,
            "graph": first_output_graph,
        },
        "second_pass": {
            "raw": second_payload,
            "graph": second_output_graph,
        },
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
