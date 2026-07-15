import uuid

from graph_core.scripts.vedas_reasoning_burner import (
    FrameArgumentPattern,
    FrameSeed,
    GoalPattern,
    Limits,
    QueryPlan,
    WorkingEdge,
    WorkingGraph,
    WorkingNode,
    compile_query,
    execute_operator,
    reason,
    seed_token_coverage,
    unify_goal,
)


def _id(value: int) -> uuid.UUID:
    return uuid.UUID(int=value)


def test_compile_query_selects_intent_operator() -> None:
    assert compile_query("When should I do A vs B?").operator == "choose"
    assert compile_query("Explain why this happens").operator == "explain"
    assert compile_query("How does Agni carry offerings?").operator == "explain"
    assert compile_query("How to understand the Vedas?").operator == "procedure"
    plan = compile_query("How and when to do nadi shodhana?")
    assert plan.operator == "procedure"
    assert plan.desired_outputs == ("steps", "timing", "conditions", "exceptions")
    assert compile_query("Suggest a redesign").operator == "redesign"


def test_seed_coverage_rejects_semantically_near_but_absent_entities() -> None:
    plan = compile_query("Nadi shodhana versus bhastrika")
    unrelated = [WorkingNode(_id(1), "Mandala 10", "SECTION", seed_score=0.5)]

    assert seed_token_coverage(plan, unrelated) == ()


def test_frame_coverage_preserves_relational_query_language() -> None:
    plan = compile_query("How does Agni carry offerings?")
    frame = FrameSeed(
        id=_id(10),
        kind="proposition",
        title="Agni carries offerings to the devas",
        text="Agni serves as messenger and carries ritual offerings to the devas.",
        predicate="CARRIES_TO",
        score=0.8,
        proposition_id=_id(11),
        argument_ids=(_id(12), _id(13)),
        executable_status="grounded_binary",
    )

    assert seed_token_coverage(plan, [], [frame]) == ("agni", "carry", "offerings")


def test_choose_operator_preserves_conditions_and_exceptions() -> None:
    plan = compile_query("When should I choose A versus B?")
    conclusions = [
        {
            "proposition_id": "p1",
            "status": "conditional",
            "statement": "A helps. Polarity=positive",
            "subjects": ["A"],
            "objects": ["calm"],
            "conditions": ["when agitated"],
            "exceptions": ["when feverish"],
        }
    ]

    result = execute_operator(plan, conclusions, [])

    assert result["applicable"][0]["conditions"] == ["when agitated"]


def test_reasoning_fires_rule_when_condition_is_active() -> None:
    graph = WorkingGraph(
        nodes={
            _id(1): WorkingNode(
                _id(1), "slow breath", "CONDITION", "slow breath", seed_score=0.9
            ),
            _id(2): WorkingNode(_id(2), "rule", "RULE"),
            _id(3): WorkingNode(
                _id(3),
                "claim",
                "PROPOSITION",
                "Practice calms the mind. Polarity=positive",
            ),
            _id(4): WorkingNode(_id(4), "practice", "PRACTICE", seed_score=0.8),
            _id(5): WorkingNode(_id(5), "calm", "OUTCOME"),
        },
        edges={
            _id(11): WorkingEdge(_id(11), _id(1), _id(2), "ANTECEDENT_OF", 1),
            _id(12): WorkingEdge(_id(12), _id(2), _id(3), "CONCLUDES", 1),
            _id(13): WorkingEdge(_id(13), _id(3), _id(4), "SUBJECT", 1),
            _id(14): WorkingEdge(_id(14), _id(3), _id(5), "OBJECT", 1),
        },
    )
    plan = QueryPlan("choose", "slow breath", ("slow", "breath"), ("conditions",))

    result = reason(plan, graph, Limits())

    assert result["rules"][0]["status"] == "fired"
    assert result["conclusions"][0]["status"] == "derived"


def test_reasoning_blocks_rule_when_exception_is_active() -> None:
    graph = WorkingGraph(
        nodes={
            _id(1): WorkingNode(_id(1), "condition", "CONDITION", seed_score=0.8),
            _id(2): WorkingNode(
                _id(2), "cold", "EXCEPTION", "when suffering a cold", seed_score=0.9
            ),
            _id(3): WorkingNode(_id(3), "rule", "RULE"),
            _id(4): WorkingNode(
                _id(4), "claim", "PROPOSITION", "Drink milk. Polarity=positive"
            ),
        },
        edges={
            _id(11): WorkingEdge(_id(11), _id(1), _id(3), "ANTECEDENT_OF", 1),
            _id(12): WorkingEdge(_id(12), _id(2), _id(3), "BLOCKS", 1),
            _id(13): WorkingEdge(_id(13), _id(3), _id(4), "CONCLUDES", 1),
        },
    )
    plan = QueryPlan(
        "prove_or_disprove", "milk with cold", ("cold", "milk"), ("support",)
    )

    result = reason(plan, graph, Limits())

    assert result["rules"][0]["status"] == "blocked"
    assert result["rules"][0]["active_blockers"] == [str(_id(2))]


def test_goal_unification_binds_frame_variables() -> None:
    goal = GoalPattern(
        "SUPPORTS",
        (
            FrameArgumentPattern("subject", entity_id=_id(1)),
            FrameArgumentPattern("object", entity_id=_id(2)),
        ),
    )
    candidate = FrameSeed(
        id=_id(10),
        kind="proposition",
        title="Something supports calmness",
        text="Something supports calmness.",
        predicate="SUPPORTS",
        score=0.0,
        proposition_id=_id(11),
        argument_ids=(_id(2),),
        executable_status="symbolic",
        arguments=(
            FrameArgumentPattern("subject", variable_name="x"),
            FrameArgumentPattern("object", entity_id=_id(2)),
        ),
    )

    assert unify_goal(goal, candidate) == {"x": str(_id(1))}


def test_backward_reasoning_proves_conclusion_from_asserted_proposition() -> None:
    graph = WorkingGraph(
        nodes={
            _id(1): WorkingNode(_id(1), "Practice", "PRACTICE", seed_score=0.9),
            _id(2): WorkingNode(_id(2), "Readiness", "STATE"),
            _id(3): WorkingNode(_id(3), "Calmness", "OUTCOME", seed_score=0.8),
            _id(4): WorkingNode(_id(4), "premise", "PROPOSITION"),
            _id(5): WorkingNode(_id(5), "rule", "RULE"),
            _id(6): WorkingNode(_id(6), "conclusion", "PROPOSITION"),
        },
        edges={
            _id(20): WorkingEdge(_id(20), _id(4), _id(5), "ANTECEDENT_OF", 1),
            _id(21): WorkingEdge(_id(21), _id(5), _id(6), "CONCLUDES", 1),
            _id(22): WorkingEdge(_id(22), _id(4), _id(1), "SUBJECT", 1),
            _id(23): WorkingEdge(_id(23), _id(4), _id(2), "OBJECT", 1),
            _id(24): WorkingEdge(_id(24), _id(6), _id(1), "SUBJECT", 1),
            _id(25): WorkingEdge(_id(25), _id(6), _id(3), "OBJECT", 1),
        },
    )
    frames = [
        FrameSeed(
            _id(30),
            "proposition",
            "Practice prepares Readiness",
            "Practice prepares readiness.",
            "PREPARES",
            0.4,
            _id(4),
            (_id(1), _id(2)),
            "grounded_binary",
        ),
        FrameSeed(
            _id(31),
            "proposition",
            "Practice enables Calmness",
            "Practice enables calmness.",
            "ENABLES",
            0.9,
            _id(6),
            (_id(1), _id(3)),
            "grounded_binary",
        ),
    ]

    result = reason(
        compile_query("How does Practice enable Calmness?"),
        graph,
        Limits(),
        frames,
    )

    backward = result["backward_reasoning"]
    assert backward["status"] == "proved"
    enables_goal = next(
        goal for goal in backward["goals"] if goal["goal"]["predicate"] == "ENABLES"
    )
    assert enables_goal["status"] == "proved"
    assert enables_goal["matches"][0]["alternatives"][0]["status"] == "proved"


def test_backward_reasoning_keeps_unmatched_query_terms_as_subgoals() -> None:
    graph = WorkingGraph(
        nodes={
            _id(1): WorkingNode(_id(1), "Agni", "CONCEPT", seed_score=0.9),
            _id(2): WorkingNode(_id(2), "Devas", "CONCEPT", seed_score=0.8),
            _id(3): WorkingNode(_id(3), "claim", "PROPOSITION"),
        },
        edges={
            _id(10): WorkingEdge(_id(10), _id(3), _id(1), "SUBJECT", 1),
            _id(11): WorkingEdge(_id(11), _id(3), _id(2), "OBJECT", 1),
        },
    )
    frame = FrameSeed(
        _id(20),
        "proposition",
        "Agni serves Devas",
        "Agni serves as the seat of Devas.",
        "SERVES_AS_SEAT_OF",
        0.9,
        _id(3),
        (_id(1), _id(2)),
        "grounded_binary",
    )

    result = reason(
        compile_query("How does Agni carry offerings to Devas?"),
        graph,
        Limits(),
        [frame],
    )

    backward = result["backward_reasoning"]
    assert backward["status"] == "insufficient_goal_coverage"
    assert "carry" in backward["unmatched_query_terms"]
    assert "offerings" in backward["unmatched_query_terms"]
