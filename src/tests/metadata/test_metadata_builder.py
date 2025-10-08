import pytest
from uuid import UUID
from vectordb.metadata.metadata_builder import MetadataBuilder, MetadataModel


@pytest.fixture(scope='module')
def builder():
    return MetadataBuilder(field_mapping={'author':'doc_author'})

def test_basic_build(builder: MetadataBuilder):
    meta = builder.build(
        row_data={"author": "Толстой", "title": "Война и мир"},
        chunk_index=0,
        total_chunks=2,
        source_id="src_1"
    )
    assert isinstance(meta, dict)
    assert meta["chunk_index"] == 0
    assert meta["total_chunks"] == 2
    assert meta["source_id"] == "src_1"
    UUID(meta["chunk_id"])
    assert meta["data"]["doc_author"] == "Толстой"
    assert "author" not in meta["data"]

def test_auto_chunk_id_generation():
    builder = MetadataBuilder()
    meta1 = builder.build({"text": "A"}, 0, 1)
    meta2 = builder.build({"text": "B"}, 0, 1)

    assert meta1["chunk_id"] != meta2["chunk_id"]
    UUID(meta1["chunk_id"])
    UUID(meta2["chunk_id"])

def test_field_mapping_disabled():
    builder = MetadataBuilder()
    meta = builder.build({"field": "value"}, 0, 1)

    assert "field" in meta["data"]
    assert meta["data"]["field"] == "value"

def test_validation_error_chunk_index_exceeds_total():
    builder = MetadataBuilder()
    with pytest.raises(ValueError):
        builder.build(
            row_data={"title": "Example"},
            chunk_index=5,
            total_chunks=3,
        )

def test_system_fields_included_by_default():
    builder = MetadataBuilder()
    meta = builder.build({"content": "X"}, 0, 1)

    assert "chunk_id" in meta
    assert "chunk_index" in meta
    assert "data" in meta

def test_custom_field_mapping(builder):
    meta = builder.build({"author": "Пушкин"}, 0, 3)

    assert "doc_author" in meta["data"]
    assert meta["data"]["doc_author"] == "Пушкин"

def test_multiple_chunks_integration():
    builder = MetadataBuilder(field_mapping={"author": "creator"})

    row_data = {"author": "Достоевский", "title": "Преступление и наказание"}
    source_id = "book_001"

    total_chunks = 3
    metas = []

    for i in range(total_chunks):
        meta = builder.build(
            row_data=row_data,
            chunk_index=i,
            total_chunks=total_chunks,
            source_id=source_id
        )
        metas.append(meta)

    chunk_ids = [m["chunk_id"] for m in metas]
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk_id должны быть уникальны"

    assert all(m["total_chunks"] == total_chunks for m in metas)
    assert all(m["source_id"] == source_id for m in metas)

    assert [m["chunk_index"] for m in metas] == list(range(total_chunks))

    assert all("creator" in m["data"] for m in metas)
    assert all(m["data"]["creator"] == "Достоевский" for m in metas)

    assert all("title" in m["data"] for m in metas)
    assert all(m["data"]["title"] == "Преступление и наказание" for m in metas)

import pytest
from pydantic import ValidationError
from vectordb.metadata.metadata_builder import MetadataBuilder, MetadataModel


def test_invalid_field_types():
    builder = MetadataBuilder()

    with pytest.raises(ValidationError) as exc_info1:
        MetadataModel(
            chunk_index="abc",  # должно быть int
            total_chunks=2,
            data={"key": "value"},
        )
    assert "chunk_index" in str(exc_info1.value)

    with pytest.raises(ValidationError) as exc_info2:
        MetadataModel(
            chunk_index=1,
            total_chunks="wrong",  # должно быть int
            data={"key": "value"},
        )
    assert "total_chunks" in str(exc_info2.value)

    with pytest.raises(ValidationError) as exc_info3:
        builder.build(
            row_data={"title": "test"},
            chunk_index="bad",  # строка вместо int
            total_chunks=3,
        )
    assert "chunk_index" in str(exc_info3.value)