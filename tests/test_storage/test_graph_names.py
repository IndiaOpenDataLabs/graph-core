import uuid

from graph_core.storage.graph_names import (
    collection_graph_name,
    legacy_collection_graph_name,
)


def test_collection_graph_name_uses_readable_slug():
    collection_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    graph_name = collection_graph_name(
        collection_id=collection_id,
        collection_name="Ayurveda Corpus v2",
    )

    assert graph_name == "collection_ayurveda_corpus_v2_12345678"


def test_legacy_collection_graph_name_uses_uuid_hex():
    collection_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    assert (
        legacy_collection_graph_name(collection_id)
        == "collection_12345678123456781234567812345678"
    )
