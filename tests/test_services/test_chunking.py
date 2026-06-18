from graph_core.services.chunking import DocumentChunker


def test_document_chunker_uses_zero_overlap_for_code_paths() -> None:
    chunker = DocumentChunker(chunk_size_tokens=400, chunk_overlap_tokens=40)

    assert chunker._prose_splitter._chunk_overlap == 40
    assert chunker._generic_code_splitter._chunk_overlap == 0
