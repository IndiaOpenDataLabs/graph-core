from graph_core.storage.meta_collections import (
    base_collection_name,
    is_meta_collection_name,
    meta_collection_level,
    meta_collection_name,
    parse_meta_collection_name,
)


def test_parse_legacy_meta_collection_name():
    assert parse_meta_collection_name("rlm__meta") == ("rlm", 1)
    assert base_collection_name("rlm__meta") == "rlm"
    assert meta_collection_level("rlm__meta") == 1
    assert is_meta_collection_name("rlm__meta") is True


def test_parse_leveled_meta_collection_name():
    assert parse_meta_collection_name("rlm__meta__l2") == ("rlm", 2)
    assert base_collection_name("rlm__meta__l2") == "rlm"
    assert meta_collection_level("rlm__meta__l2") == 2
    assert meta_collection_name("rlm__meta__l2", 3) == "rlm__meta__l3"


def test_non_meta_collection_name_round_trips():
    assert parse_meta_collection_name("rlm") is None
    assert base_collection_name("rlm") == "rlm"
    assert meta_collection_level("rlm") == 0
    assert is_meta_collection_name("rlm") is False
    assert meta_collection_name("rlm", 1) == "rlm__meta__l1"
