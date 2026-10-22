from datetime import date

from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.dialects import postgresql

from umai.rag.document_models import RagDocument, chunk_document
from umai.rag.extractors.operational_text import OperationalTextExtractor
from umai.rag.vector_store import (
    _batched,
    _chunk_row,
    _entity_name_condition,
    _extra_metadata,
    _merge_hybrid_rows,
    _pgvector_extension_exists,
    _rag_chunks_table,
    _search_row_to_match,
)
from umai.tools.operational_context_tools import build_operational_context_tools


class FakeOperationalContextStore:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "query": kwargs["query"],
            "count": 1,
            "matches": [
                {
                    "source_type": "general_note",
                    "source_id": "1",
                    "site_id": kwargs["site_id"],
                    "event_date": "2026-05-10",
                    "entity_name": None,
                    "text": "General note. Comments: Pump issue mentioned.",
                }
            ],
        }


class FakeExtensionConnection:
    def __init__(self, exists):
        self.exists = exists
        self.statements = []

    def execute(self, statement):
        self.statements.append(str(statement))
        return self

    def mappings(self):
        return self

    def one(self):
        return {
            "database": "ometrics",
            "user": "ometrics",
            "server_addr": "127.0.0.1",
            "server_port": 5432,
            "installed": self.exists,
            "extension_schema": "public" if self.exists else None,
            "type_visible": self.exists,
        }


def test_chunk_document_keeps_metadata_and_stable_ids():
    document = RagDocument(
        source_type="general_note",
        source_id="123",
        site_id=4,
        event_date=date(2026, 5, 10),
        text="A" * 50,
    )

    chunks = chunk_document(document, max_chars=20, overlap_chars=5)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "general_note:123:chunk:0"
    assert chunks[0].metadata["site_id"] == 4
    assert chunks[0].metadata["event_date"] == "2026-05-10"
    assert chunks[0].metadata["event_date_key"] == 20260510


def test_vector_store_promotes_chunk_metadata_to_columns():
    chunk = chunk_document(
        RagDocument(
            source_type="general_note",
            source_id="123",
            site_id=4,
            event_date=date(2026, 5, 10),
            text="Pump issue mentioned.",
            metadata={"title": "General note"},
        )
    )[0]

    row = _chunk_row(chunk.metadata)

    assert row["source_type"] == "general_note"
    assert row["source_id"] == "123"
    assert row["site_id"] == 4
    assert row["event_date"] == date(2026, 5, 10)
    assert row["event_date_key"] == 20260510
    assert _extra_metadata(chunk.metadata) == {"title": "General note"}


def test_vector_store_table_uses_configured_embedding_dimensions():
    table = _rag_chunks_table(MetaData(), 1536)
    column_names = [column.name for column in table.columns]

    assert table.c.embedding.type.dim == 1536
    assert "text_search_vector" in table.c
    assert column_names.index("text_search_vector") == column_names.index("embedding") + 1
    assert "idx_rag_chunks_site_date" in {index.name for index in table.indexes}
    assert "idx_rag_chunks_source" in {index.name for index in table.indexes}
    assert "idx_rag_chunks_text_search" in {index.name for index in table.indexes}


def test_vector_store_batches_large_upserts():
    rows = [{"id": value} for value in range(5)]

    assert list(_batched(rows, 2)) == [
        [{"id": 0}, {"id": 1}],
        [{"id": 2}, {"id": 3}],
        [{"id": 4}],
    ]


def test_vector_store_search_row_format_matches_tool_contract():
    row = {
        "chunk_id": "general_note:1:chunk:0",
        "source_type": "general_note",
        "source_id": "1",
        "site_id": 4,
        "event_date": date(2026, 6, 10),
        "entity_type": None,
        "entity_id": None,
        "entity_name": None,
        "distance": 0.123,
        "hybrid_score": 0.032,
        "semantic_rank": 1,
        "keyword_rank": 2,
        "keyword_score": 0.45,
        "text": "General note. Comments: In range note.",
    }

    assert _search_row_to_match(row) == {
        "chunk_id": "general_note:1:chunk:0",
        "source_type": "general_note",
        "source_id": "1",
        "site_id": 4,
        "event_date": "2026-06-10",
        "entity_type": None,
        "entity_id": None,
        "entity_name": None,
        "distance": 0.123,
        "hybrid_score": 0.032,
        "semantic_rank": 1,
        "keyword_rank": 2,
        "keyword_score": 0.45,
        "text": "General note. Comments: In range note.",
    }


def test_hybrid_merge_keeps_semantic_keyword_and_dual_matches():
    base = {
        "source_type": "general_note",
        "source_id": "1",
        "site_id": 4,
        "event_date": date(2026, 6, 10),
        "entity_type": None,
        "entity_id": None,
        "entity_name": None,
        "text": "Operational note.",
    }
    semantic_rows = [
        {
            **base,
            "chunk_id": "dual",
            "distance": 0.1,
            "keyword_score": None,
        },
        {
            **base,
            "chunk_id": "semantic-only",
            "distance": 0.2,
            "keyword_score": None,
        },
    ]
    keyword_rows = [
        {
            **base,
            "chunk_id": "dual",
            "distance": None,
            "keyword_score": 0.9,
        },
        {
            **base,
            "chunk_id": "keyword-only",
            "distance": None,
            "keyword_score": 0.8,
        },
    ]

    rows = _merge_hybrid_rows(semantic_rows, keyword_rows, limit=3)

    assert [row["chunk_id"] for row in rows] == [
        "dual",
        "semantic-only",
        "keyword-only",
    ]
    assert rows[0]["semantic_rank"] == 1
    assert rows[0]["keyword_rank"] == 1
    assert rows[1]["semantic_rank"] == 2
    assert rows[1]["keyword_rank"] is None
    assert rows[2]["semantic_rank"] is None
    assert rows[2]["keyword_rank"] == 2


def test_entity_name_filter_also_allows_text_mentions():
    table = _rag_chunks_table(MetaData(), 1536)

    sql = str(
        _entity_name_condition(table, "6344").compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "rag_chunks.entity_name ILIKE '%%6344%%'" in sql
    assert "rag_chunks.text ILIKE '%%6344%%'" in sql
    assert " OR " in sql


def test_pgvector_extension_check_does_not_create_extension():
    connection = FakeExtensionConnection(exists=False)

    assert _pgvector_extension_exists(connection) is False
    assert "CREATE EXTENSION" not in connection.statements[0]
    assert "current_database()" in connection.statements[0]
    assert "pg_extension" in connection.statements[0]


def test_operational_context_tool_calls_store_with_site_scope():
    store = FakeOperationalContextStore()
    tool = build_operational_context_tools(store, site_id=4)[0]

    result = tool.invoke(
        {
            "query": "Why was the well down?",
            "start_date": "2026-05-01",
            "end_date": "2026-05-31",
            "entity_name": "HARTZOG DRAW UNIT 4048",
            "source_types": ["general_note"],
            "limit": 3,
        }
    )

    assert '"ok":true' in result
    assert store.calls[0]["site_id"] == 4
    assert store.calls[0]["start_date"] == date(2026, 5, 1)
    assert store.calls[0]["end_date"] == date(2026, 5, 31)
    assert store.calls[0]["entity_name"] == "HARTZOG DRAW UNIT 4048"
    assert store.calls[0]["source_types"] is None


def test_operational_context_tool_accepts_compact_yyyymmdd_dates():
    store = FakeOperationalContextStore()
    tool = build_operational_context_tools(store, site_id=4)[0]

    result = tool.invoke(
        {
            "query": "Summarize general notes.",
            "start_date": "20260601",
            "end_date": "20260630",
            "source_types": ["general_note"],
        }
    )

    assert '"ok":true' in result
    assert store.calls[0]["start_date"] == date(2026, 6, 1)
    assert store.calls[0]["end_date"] == date(2026, 6, 30)


def test_operational_context_tool_ignores_model_source_filter_for_broad_context_question():
    store = FakeOperationalContextStore()
    tool = build_operational_context_tools(store, site_id=4)[0]

    result = tool.invoke(
        {
            "query": "well 5823 chemical treatment hot water paraffin",
            "entity_name": "5823",
            "source_types": [
                "general_note",
                "work_order",
                "work_order_note",
                "well_shutdown",
                "alarm_log",
            ],
            "limit": 10,
        }
    )

    assert '"ok":true' in result
    assert store.calls[0]["source_types"] is None


def test_operational_context_tool_keeps_explicit_chart_note_filter():
    store = FakeOperationalContextStore()
    tool = build_operational_context_tools(store, site_id=4)[0]

    result = tool.invoke(
        {
            "query": "chart notes for 5823 mentioning paraffin",
            "entity_name": "5823",
            "source_types": ["chart_note"],
        }
    )

    assert '"ok":true' in result
    assert store.calls[0]["source_types"] == ["chart_note"]


def test_operational_context_tool_expands_general_work_order_note_filter():
    store = FakeOperationalContextStore()
    tool = build_operational_context_tools(store, site_id=4)[0]

    result = tool.invoke(
        {
            "query": "all work orders in May 2026",
            "start_date": "2026-05-01",
            "end_date": "2026-05-31",
            "source_types": ["work_order_note"],
            "limit": 20,
        }
    )

    assert '"ok":true' in result
    assert store.calls[0]["source_types"] == ["work_order_note", "work_order"]


def test_operational_context_tool_keeps_explicit_work_order_note_filter():
    store = FakeOperationalContextStore()
    tool = build_operational_context_tools(store, site_id=4)[0]

    result = tool.invoke(
        {
            "query": "work order notes in May 2026",
            "start_date": "2026-05-01",
            "end_date": "2026-05-31",
            "source_types": ["work_order_note"],
        }
    )

    assert '"ok":true' in result
    assert store.calls[0]["source_types"] == ["work_order_note"]


def test_operational_context_tool_schema_describes_work_order_source_types():
    field_description = (
        build_operational_context_tools(FakeOperationalContextStore(), site_id=4)[0]
        .args_schema.model_fields["source_types"]
        .description
    )

    assert "Use work_order for work order records" in field_description
    assert "Use work_order_note only for follow-up notes" in field_description


def test_operational_text_extractor_reads_general_notes_without_mysql_writes():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE general_notes (
                    id INTEGER PRIMARY KEY,
                    comments TEXT NOT NULL,
                    site_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    updated_at DATETIME
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO general_notes (id, comments, site_id, date, updated_at)
                VALUES (1, 'Pump issue mentioned.', 4, '2026-05-10', '2026-05-10 12:00:00')
                """
            )
        )

    documents = OperationalTextExtractor(engine).extract(
        site_id=4,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 31),
        source_types={"general_note"},
    )

    assert len(documents) == 1
    assert documents[0].source_type == "general_note"
    assert documents[0].site_id == 4
    assert documents[0].text == "General note. Comments: Pump issue mentioned."


def test_operational_text_extractor_indexes_onrr_codes_as_reference_data():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE onrr_codes (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    active_well INTEGER NOT NULL,
                    injection_well INTEGER NOT NULL,
                    description TEXT,
                    updated_at DATETIME
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO onrr_codes
                    (id, name, active_well, injection_well, description, updated_at)
                VALUES
                    (1, 'WIW', 1, 1, 'Water injection well', '2026-06-01 00:00:00')
                """
            )
        )

    documents = OperationalTextExtractor(engine).extract(
        site_id=4,
        source_types={"onrr_code"},
    )

    assert len(documents) == 1
    assert documents[0].source_type == "onrr_code"
    assert documents[0].site_id == 4
    assert documents[0].entity_type == "onrr_code"
    assert documents[0].entity_name == "WIW"
    assert "Injection Well: 1" in documents[0].text
    assert "Description: Water injection well" in documents[0].text


def test_operational_text_extractor_indexes_work_orders_by_work_order_time():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE work_orders (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    status TEXT NOT NULL,
                    vendor TEXT,
                    comments TEXT,
                    time DATETIME NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO work_orders (
                    id, site_id, subject, status, vendor, comments,
                    time, created_at, updated_at
                )
                VALUES (
                    21, 4, 'Replace transfer pump', 'open', 'Pump Vendor',
                    'Pump seal leaking.',
                    '2026-06-10 00:00:00',
                    '2026-05-01 09:00:00',
                    '2026-06-10 12:00:00'
                )
                """
            )
        )

    documents = OperationalTextExtractor(engine).extract(
        site_id=4,
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
        source_types={"work_order"},
    )

    assert len(documents) == 1
    assert documents[0].source_type == "work_order"
    assert documents[0].event_date == date(2026, 6, 10)
    assert documents[0].entity_type == "work_order"
    assert documents[0].entity_id == "21"
    assert documents[0].entity_name == "Replace transfer pump"
    assert documents[0].text == (
        "Work order. Subject: Replace transfer pump Status: open "
        "Vendor: Pump Vendor Comments: Pump seal leaking."
    )


def test_operational_text_extractor_indexes_chart_notes_for_rod_pump_well_entity():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_chart_note_source_tables(connection)
        connection.execute(
            text(
                """
                INSERT INTO wells (id, site_id, name)
                VALUES (4048, 4, 'HARTZOG DRAW UNIT 4048')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO chart_notes (
                    id, site_id, object_type, object_id, chart_name,
                    x_axis_value, note, updated_at
                )
                VALUES (
                    10, 4, 'RodPumpOilWell', 4048, 'Pump Fillage',
                    '2026-06-12 08:00:00', 'Fillage dropped after restart.',
                    '2026-06-12 08:10:00'
                )
                """
            )
        )

    documents = OperationalTextExtractor(engine).extract(
        site_id=4,
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
        source_types={"chart_note"},
    )

    assert len(documents) == 1
    assert documents[0].source_type == "chart_note"
    assert documents[0].entity_type == "well"
    assert documents[0].entity_id == "4048"
    assert documents[0].entity_name == "HARTZOG DRAW UNIT 4048"
    assert documents[0].event_date == date(2026, 6, 12)
    assert documents[0].text == (
        "Chart note. Entity Name: HARTZOG DRAW UNIT 4048 "
        "Chart Name: Pump Fillage Note: Fillage dropped after restart."
    )


def test_operational_text_extractor_indexes_chart_notes_for_equipment_entity():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_chart_note_source_tables(connection)
        connection.execute(
            text(
                """
                INSERT INTO tanks (id, site_id, name)
                VALUES (75, 4, 'Tank - 2-1 Float Over')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO chart_notes (
                    id, site_id, object_type, object_id, chart_name,
                    x_axis_value, note, updated_at
                )
                VALUES (
                    11, 4, 'Tank', 75, 'Level',
                    '2026-06-13 09:00:00', 'Level changed quickly.',
                    '2026-06-13 09:05:00'
                )
                """
            )
        )

    documents = OperationalTextExtractor(engine).extract(
        site_id=4,
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
        source_types={"chart_note"},
    )

    assert len(documents) == 1
    assert documents[0].source_type == "chart_note"
    assert documents[0].entity_type == "tank"
    assert documents[0].entity_id == "75"
    assert documents[0].entity_name == "Tank - 2-1 Float Over"
    assert documents[0].event_date == date(2026, 6, 13)
    assert documents[0].text == (
        "Chart note. Entity Name: Tank - 2-1 Float Over "
        "Chart Name: Level Note: Level changed quickly."
    )


def _create_chart_note_source_tables(connection):
    connection.execute(
        text(
            """
            CREATE TABLE chart_notes (
                id INTEGER PRIMARY KEY,
                site_id INTEGER NOT NULL,
                object_type TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                chart_name TEXT NOT NULL,
                x_axis_value DATETIME NOT NULL,
                note TEXT NOT NULL,
                updated_at DATETIME
            )
            """
        )
    )

    connection.execute(
        text(
            """
            CREATE TABLE wells (
                id INTEGER PRIMARY KEY,
                site_id INTEGER NOT NULL,
                name TEXT NOT NULL
            )
            """
        )
    )

    for table_name in (
        "lacts",
        "flares",
        "knock_outs",
        "tanks",
        "treaters",
        "water_plants",
        "pumps",
        "flow_meters",
    ):
        connection.execute(
            text(
                f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL
                )
                """
            )
        )

    connection.execute(
        text(
            """
            CREATE TABLE monitored_objects (
                id INTEGER PRIMARY KEY,
                site_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                name TEXT NOT NULL
            )
            """
        )
    )
