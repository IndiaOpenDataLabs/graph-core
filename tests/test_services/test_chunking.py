import unicodedata

from graph_core.services.chunking import DocumentChunker


def test_document_chunker_uses_zero_overlap_for_code_paths() -> None:
    chunker = DocumentChunker(chunk_size_tokens=512, chunk_overlap_tokens=0)

    assert chunker._prose_splitter._chunk_overlap == 0
    assert chunker._generic_code_splitter._chunk_overlap == 0


def test_document_chunker_preserves_document_and_markdown_hierarchy() -> None:
    chunker = DocumentChunker(chunk_size_tokens=24, chunk_overlap_tokens=0)
    chunks = chunker.chunk_document(
        (
            "# Book\n\n"
            "Intro text.\n\n"
            "## Technique 1\n\n"
            "Use this for one condition.\n\n"
            "## Technique 2\n\n"
            "Use this for a different condition."
        ),
        document_path="library/book.md",
    )

    assert all(chunk.startswith("Source hierarchy:") for chunk in chunks)
    assert any("Document: library/book.md" in chunk for chunk in chunks)
    assert any("Folder: library" in chunk for chunk in chunks)
    assert any("Section: Book > Technique 2" in chunk for chunk in chunks)


def test_document_chunker_tracks_nested_markdown_heading_stack() -> None:
    chunker = DocumentChunker(chunk_size_tokens=512, chunk_overlap_tokens=0)
    text = (
        "# Book\n\n"
        "## Part A\n\n"
        "### Technique 1\n\n"
        "Alpha details.\n\n"
        "## Part B\n\n"
        "### Technique 1\n\n"
        "Beta details.\n"
    )

    alpha_hierarchy = chunker._source_hierarchy_at(
        text,
        text.index("Alpha details"),
        document_path="library/books/manual.md",
    )
    beta_hierarchy = chunker._source_hierarchy_at(
        text,
        text.index("Beta details"),
        document_path="library/books/manual.md",
    )

    assert alpha_hierarchy.folder_path == "library/books"
    assert alpha_hierarchy.headings == ("Book", "Part A", "Technique 1")
    assert beta_hierarchy.headings == ("Book", "Part B", "Technique 1")


def test_document_chunker_supports_setext_headings() -> None:
    chunker = DocumentChunker(chunk_size_tokens=512, chunk_overlap_tokens=0)
    text = (
        "Book Title\n"
        "==========\n\n"
        "Technique 1\n"
        "-----------\n\n"
        "Details.\n"
    )

    hierarchy = chunker._source_hierarchy_at(
        text,
        text.index("Details"),
        document_path="book.md",
    )

    assert hierarchy.headings == ("Book Title", "Technique 1")


def test_document_chunker_ignores_headings_inside_fenced_code() -> None:
    chunker = DocumentChunker(chunk_size_tokens=512, chunk_overlap_tokens=0)
    text = "# Real\n\n```\n## Not a heading\n```\n\nContent.\n"

    hierarchy = chunker._source_hierarchy_at(
        text,
        text.index("Content"),
        document_path="book.md",
    )

    assert hierarchy.headings == ("Real",)


def test_document_chunker_repairs_combining_mark_boundaries() -> None:
    chunks = DocumentChunker._clean_chunks(["ལ", "\u0f74འོ"])

    assert chunks == ["ལ\u0f74", "འོ"]
    assert unicodedata.combining(chunks[1][0]) == 0


def test_document_chunker_normalizes_text_and_bounds_malformed_headings() -> None:
    chunker = DocumentChunker(chunk_size_tokens=20, chunk_overlap_tokens=0)
    repeated_heading = "ཡུལ་" * 300
    chunks = chunker.chunk_document(
        f"## {repeated_heading}\n\n" + ("ལ\u0f74འོ་" * 200),
        document_path="book.md",
    )

    assert chunks
    for chunk in chunks:
        prefix, _, body = chunk.partition("\n\nChunk text:\n")
        section = next(
            line.removeprefix("Section: ")
            for line in prefix.splitlines()
            if line.startswith("Section: ")
        )
        assert len(section) <= 512
        assert body == unicodedata.normalize("NFC", body)
        assert not body or unicodedata.combining(body[0]) == 0
