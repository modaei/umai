from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings
from umai.rag.document_models import RagDocument


class OperationalTextExtractorError(RuntimeError):
    """Raised when operational text cannot be extracted from Ometrics."""


class OperationalTextExtractor:
    """Read operational text records from the existing Ometrics database.

    This extractor is intentionally read-only. It never creates RAG tables, never
    stores a sync ledger, and never updates source rows. It only converts current
    Ometrics rows into normalized RagDocument objects for the vector index.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    @classmethod
    def from_settings(cls, settings: Settings) -> "OperationalTextExtractor":
        # Reuse the normal Ometrics DB settings. A read-only MySQL user is enough
        # because indexing reads source rows and writes only to the vector DB.
        settings.validate_database()
        url = URL.create(
            drivername="mysql+pymysql",
            username=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
        )
        engine = create_engine(
            url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
        )
        return cls(engine)

    def extract(
        self,
        site_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
        source_types: set[str] | None = None,
    ) -> list[RagDocument]:
        """Extract supported source records for a site/date/source selection.

        Ometrics installations can be at different migration levels. Inspecting
        table names lets the indexer skip unavailable sources instead of failing
        the whole run because one newer table does not exist.
        """

        available_tables = set(inspect(self.engine).get_table_names())
        documents: list[RagDocument] = []
        for source in SOURCE_DEFINITIONS:
            if source_types and source.source_type not in source_types:
                continue
            if not source.required_tables.issubset(available_tables):
                # Skip this source for older schemas or optional feature tables.
                continue
            documents.extend(source.extract(self.engine, site_id, start_date, end_date))
        return documents

    def extract_one(
        self,
        site_id: int,
        source_type: str,
        source_id: str,
    ) -> RagDocument | None:
        available_tables = set(inspect(self.engine).get_table_names())
        source = source_definition(source_type)
        if not source.required_tables.issubset(available_tables):
            return None
        return source.extract_one(self.engine, site_id, source_id)


class SourceDefinition:
    """Declarative mapping from one SQL source to normalized RAG documents.

    Most source tables follow the same pattern: filter by site/date, select a few
    useful text fields, and attach entity/date metadata. A declarative definition
    avoids one nearly identical extractor class per note/comment table.
    """

    def __init__(
        self,
        source_type: str,
        required_tables: set[str],
        query: str,
        text_fields: tuple[str, ...],
        event_date_field: str | None = None,
        entity_type: str | None = None,
        entity_type_field: str | None = None,
        entity_id_field: str | None = None,
        entity_name_field: str | None = None,
        title: str | None = None,
    ):
        self.source_type = source_type
        self.required_tables = required_tables
        self.query = query
        self.text_fields = text_fields
        self.event_date_field = event_date_field
        self.entity_type = entity_type
        self.entity_type_field = entity_type_field
        self.entity_id_field = entity_id_field
        self.entity_name_field = entity_name_field
        self.title = title or source_type.replace("_", " ").title()

    def extract(
        self,
        engine: Engine,
        site_id: int,
        start_date: date | None,
        end_date: date | None,
    ) -> list[RagDocument]:
        # All user-controlled values are bind params. The SQL text is static and
        # owned by the source definition.
        try:
            with engine.connect() as connection:
                rows = list(
                    connection.execute(
                        text(self.query),
                        {
                            "site_id": site_id,
                            "start_date": start_date,
                            "end_date": end_date,
                        },
                    ).mappings()
                )
        except SQLAlchemyError as exc:
            raise OperationalTextExtractorError(
                f"Could not extract {self.source_type}: {exc}"
            ) from exc

        documents = []
        for row in rows:
            document = self._row_to_document(row, site_id)
            if document is not None:
                documents.append(document)
        return documents

    def extract_one(
        self,
        engine: Engine,
        site_id: int,
        source_id: str,
    ) -> RagDocument | None:
        query = f"""
            SELECT *
            FROM (
                {self.query}
            ) source_rows
            WHERE source_rows.id = :source_id
        """
        try:
            with engine.connect() as connection:
                row = (
                    connection.execute(
                        text(query),
                        {
                            "site_id": site_id,
                            "source_id": source_id,
                            "start_date": None,
                            "end_date": None,
                        },
                    )
                    .mappings()
                    .first()
                )
        except SQLAlchemyError as exc:
            raise OperationalTextExtractorError(
                f"Could not extract {self.source_type} {source_id}: {exc}"
            ) from exc

        if row is None:
            return None
        return self._row_to_document(row, site_id)

    def _row_to_document(self, row: dict[str, Any], site_id: int) -> RagDocument | None:
        # Build readable labeled text. Labels make retrieved snippets clearer for
        # the LLM than raw concatenated values.
        parts = []
        for field in self.text_fields:
            value = row.get(field)
            if value is not None and str(value).strip():
                label = field.replace("_", " ").title()
                parts.append(f"{label}: {str(value).strip()}")
        if not parts:
            # Semantic search over empty text has no value.
            return None

        event_date = (
            _coerce_date(row.get(self.event_date_field))
            if self.event_date_field
            else None
        )
        entity_type = (
            row.get(self.entity_type_field)
            if self.entity_type_field
            else self.entity_type
        )
        entity_id = row.get(self.entity_id_field) if self.entity_id_field else None
        entity_name = row.get(self.entity_name_field) if self.entity_name_field else None
        text_value = f"{self.title}. " + " ".join(parts)

        return RagDocument(
            source_type=self.source_type,
            source_id=str(row["id"]),
            site_id=site_id,
            text=text_value,
            event_date=event_date,
            entity_type=str(entity_type).strip() if entity_type is not None else None,
            entity_id=str(entity_id) if entity_id is not None else None,
            entity_name=str(entity_name).strip() if entity_name is not None else None,
            source_updated_at=_coerce_datetime(row.get("updated_at")),
            metadata={"title": self.title},
        )


def _coerce_date(value: Any) -> date | None:
    # SQLAlchemy may return date, datetime, or string depending on driver and SQL
    # expression. Normalize before storing as vector metadata.
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _coerce_datetime(value: Any) -> datetime | None:
    # updated_at is optional metadata. Missing values are omitted from the vector DB.
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


DATE_FILTER = """
AND (:start_date IS NULL OR {field} >= :start_date)
AND (:end_date IS NULL OR {field} <= :end_date)
"""


CHART_NOTE_REQUIRED_TABLES = {
    "chart_notes",
    "wells",
    "lacts",
    "flares",
    "knock_outs",
    "tanks",
    "treaters",
    "water_plants",
    "pumps",
    "flow_meters",
    "monitored_objects",
}


CHART_NOTE_KNOWN_OBJECT_TYPES = (
    "'RodPumpOilWell'",
    "'OilWell'",
    "'WaterWell'",
    "'Lact'",
    "'Flare'",
    "'KnockOut'",
    "'Tank'",
    "'Treater'",
    "'WaterPlant'",
    "'Pump'",
    "'FlowMeter'",
    "'Flow_Meter'",
)


CHART_NOTE_KNOWN_OBJECT_TYPES_SQL = ", ".join(CHART_NOTE_KNOWN_OBJECT_TYPES)


CHART_NOTE_QUERY = f"""
    SELECT id, site_id, object_type, entity_type, entity_id, entity_name, chart_name,
        event_date, note, updated_at
    FROM (
        SELECT cn.id, cn.site_id, cn.object_type, 'well' AS entity_type, w.id AS entity_id,
            w.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN wells w
            ON w.id = cn.object_id
            AND w.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type = 'RodPumpOilWell'

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'well' AS entity_type, w.id AS entity_id,
            w.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN wells w
            ON w.id = cn.object_id
            AND w.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type IN ('OilWell', 'WaterWell')

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'lact' AS entity_type, l.id AS entity_id,
            l.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN lacts l
            ON l.id = cn.object_id
            AND l.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type = 'Lact'

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'flare' AS entity_type, f.id AS entity_id,
            f.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN flares f
            ON f.id = cn.object_id
            AND f.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type = 'Flare'

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'knock_out' AS entity_type, ko.id AS entity_id,
            ko.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN knock_outs ko
            ON ko.id = cn.object_id
            AND ko.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type = 'KnockOut'

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'tank' AS entity_type, t.id AS entity_id,
            t.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN tanks t
            ON t.id = cn.object_id
            AND t.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type = 'Tank'

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'treater' AS entity_type, tr.id AS entity_id,
            tr.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN treaters tr
            ON tr.id = cn.object_id
            AND tr.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type = 'Treater'

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'water_plant' AS entity_type, wp.id AS entity_id,
            wp.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN water_plants wp
            ON wp.id = cn.object_id
            AND wp.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type = 'WaterPlant'

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'pump' AS entity_type, p.id AS entity_id,
            p.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN pumps p
            ON p.id = cn.object_id
            AND p.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type = 'Pump'

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'flow_meter' AS entity_type, fm.id AS entity_id,
            fm.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN flow_meters fm
            ON fm.id = cn.object_id
            AND fm.site_id = cn.site_id
        WHERE cn.site_id = :site_id
            AND cn.object_type IN ('FlowMeter', 'Flow_Meter')

        UNION ALL

        SELECT cn.id, cn.site_id, cn.object_type, 'monitored_object' AS entity_type, mo.id AS entity_id,
            mo.name AS entity_name, cn.chart_name,
            cn.x_axis_value AS event_date, cn.note, cn.updated_at
        FROM chart_notes cn
        JOIN monitored_objects mo
            ON mo.id = cn.object_id
            AND mo.site_id = cn.site_id
            AND mo.type = cn.object_type
        WHERE cn.site_id = :site_id
            AND cn.object_type NOT IN ({CHART_NOTE_KNOWN_OBJECT_TYPES_SQL})
    ) chart_note_rows
    WHERE site_id = :site_id
    {DATE_FILTER.format(field="DATE(event_date)")}
"""


# Each source definition below is independent. If a required table is absent,
# OperationalTextExtractor skips the source. If a row has no useful text, it is
# ignored. The resulting RagDocument objects are the only data sent to the vector DB.
SOURCE_DEFINITIONS: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        source_type="onrr_code",
        required_tables={"onrr_codes"},
        query="""
            SELECT id, :site_id AS site_id, name, active_well, injection_well,
                description, updated_at
            FROM onrr_codes
        """,
        text_fields=("name", "active_well", "injection_well", "description"),
        entity_type="onrr_code",
        entity_id_field="id",
        entity_name_field="name",
        title="ONRR code",
    ),
    SourceDefinition(
        source_type="general_note",
        required_tables={"general_notes"},
        query=f"""
            SELECT id, site_id, date AS event_date, comments, updated_at
            FROM general_notes
            WHERE site_id = :site_id
            {DATE_FILTER.format(field="date")}
        """,
        text_fields=("comments",),
        event_date_field="event_date",
        title="General note",
    ),
    SourceDefinition(
        source_type="chart_note",
        required_tables=CHART_NOTE_REQUIRED_TABLES,
        query=CHART_NOTE_QUERY,
        text_fields=("entity_name", "chart_name", "note"),
        event_date_field="event_date",
        entity_type_field="entity_type",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Chart note",
    ),
    SourceDefinition(
        source_type="work_order",
        required_tables={"work_orders"},
        query=f"""
            SELECT id, site_id, subject, status, vendor, comments,
                time AS event_date, updated_at
            FROM work_orders
            WHERE site_id = :site_id
            {DATE_FILTER.format(field="DATE(time)")}
        """,
        text_fields=("subject", "status", "vendor", "comments"),
        event_date_field="event_date",
        entity_type="work_order",
        entity_id_field="id",
        entity_name_field="subject",
        title="Work order",
    ),
    SourceDefinition(
        source_type="work_order_note",
        required_tables={"work_order_notes", "work_orders"},
        query=f"""
            SELECT won.id, wo.site_id, wo.id AS work_order_id, wo.subject,
                won.notes, won.time AS event_date, won.updated_at
            FROM work_order_notes won
            JOIN work_orders wo ON wo.id = won.work_order_id
            WHERE wo.site_id = :site_id
            {DATE_FILTER.format(field="DATE(won.time)")}
        """,
        text_fields=("subject", "notes"),
        event_date_field="event_date",
        entity_type="work_order",
        entity_id_field="work_order_id",
        entity_name_field="subject",
        title="Work order note",
    ),
    SourceDefinition(
        source_type="well_shutdown",
        required_tables={"well_shutdowns", "wells"},
        query=f"""
            SELECT ws.id, w.site_id, ws.well_id, w.name AS well_name,
                DATE(ws.`start`) AS event_date,
                ws.`start` AS start, ws.`end` AS end,
                ws.downtime_code, ws.comments, ws.updated_at
            FROM well_shutdowns ws
            JOIN wells w ON w.id = ws.well_id
            WHERE w.site_id = :site_id
            {DATE_FILTER.format(field="DATE(ws.`start`)")}
        """,
        text_fields=(
            "well_name",
            "downtime_code",
            "comments",
            "start",
            "end",
        ),
        event_date_field="event_date",
        entity_type="well",
        entity_id_field="well_id",
        entity_name_field="well_name",
        title="Well shutdown",
    ),
    SourceDefinition(
        source_type="well_test",
        required_tables={"well_tests", "wells"},
        query=f"""
            SELECT wt.id, w.site_id, wt.well_id, w.name AS well_name,
                wt.time AS event_date, wt.oil, wt.water, wt.gas, wt.comments,
                wt.updated_at
            FROM well_tests wt
            JOIN wells w ON w.id = wt.well_id
            WHERE w.site_id = :site_id
                AND wt.comments IS NOT NULL
                AND wt.comments <> ''
            {DATE_FILTER.format(field="DATE(wt.time)")}
        """,
        text_fields=("well_name", "oil", "water", "gas", "comments"),
        event_date_field="event_date",
        entity_type="well",
        entity_id_field="well_id",
        entity_name_field="well_name",
        title="Well test comment",
    ),
    SourceDefinition(
        source_type="lact_reading_comment",
        required_tables={"lact_readings", "lacts"},
        query=f"""
            SELECT lr.id, l.site_id, lr.lact_id AS entity_id, l.name AS entity_name,
                lr.time AS event_date, lr.comments, lr.updated_at
            FROM lact_readings lr
            JOIN lacts l ON l.id = lr.lact_id
            WHERE l.site_id = :site_id
                AND lr.comments IS NOT NULL
                AND lr.comments <> ''
            {DATE_FILTER.format(field="DATE(lr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="lact",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="LACT reading comment",
    ),
    SourceDefinition(
        source_type="flare_reading_comment",
        required_tables={"flare_readings", "flares"},
        query=f"""
            SELECT fr.id, f.site_id, fr.flare_id AS entity_id, f.name AS entity_name,
                fr.time AS event_date, fr.comments, fr.updated_at
            FROM flare_readings fr
            JOIN flares f ON f.id = fr.flare_id
            WHERE f.site_id = :site_id
                AND fr.comments IS NOT NULL
                AND fr.comments <> ''
            {DATE_FILTER.format(field="DATE(fr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="flare",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Flare reading comment",
    ),
    SourceDefinition(
        source_type="tank_reading_comment",
        required_tables={"linear_tank_readings", "tanks"},
        query=f"""
            SELECT tr.id, t.site_id, tr.tank_id AS entity_id, t.name AS entity_name,
                tr.time AS event_date, tr.comments, tr.updated_at
            FROM linear_tank_readings tr
            JOIN tanks t ON t.id = tr.tank_id
            WHERE t.site_id = :site_id
                AND tr.comments IS NOT NULL
                AND tr.comments <> ''
            {DATE_FILTER.format(field="DATE(tr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="tank",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Tank reading comment",
    ),
    SourceDefinition(
        source_type="flow_meter_reading_comment",
        required_tables={"flow_meter_readings", "flow_meters"},
        query=f"""
            SELECT fmr.id, fm.site_id, fmr.flow_meter_id AS entity_id,
                fm.name AS entity_name, fmr.time AS event_date,
                fmr.comments, fmr.updated_at
            FROM flow_meter_readings fmr
            JOIN flow_meters fm ON fm.id = fmr.flow_meter_id
            WHERE fm.site_id = :site_id
                AND fmr.comments IS NOT NULL
                AND fmr.comments <> ''
            {DATE_FILTER.format(field="DATE(fmr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="flow_meter",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Flow meter reading comment",
    ),
    SourceDefinition(
        source_type="water_plant_reading_comment",
        required_tables={"water_plant_readings", "water_plants"},
        query=f"""
            SELECT wpr.id, wp.site_id, wpr.water_plant_id AS entity_id,
                wp.name AS entity_name, wpr.time AS event_date,
                wpr.comments, wpr.updated_at
            FROM water_plant_readings wpr
            JOIN water_plants wp ON wp.id = wpr.water_plant_id
            WHERE wp.site_id = :site_id
                AND wpr.comments IS NOT NULL
                AND wpr.comments <> ''
            {DATE_FILTER.format(field="DATE(wpr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="water_plant",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Water plant reading comment",
    ),
    SourceDefinition(
        source_type="treater_reading_comment",
        required_tables={"treater_readings", "treaters"},
        query=f"""
            SELECT tr.id, t.site_id, tr.treater_id AS entity_id,
                t.name AS entity_name, tr.time AS event_date,
                tr.comments, tr.updated_at
            FROM treater_readings tr
            JOIN treaters t ON t.id = tr.treater_id
            WHERE t.site_id = :site_id
                AND tr.comments IS NOT NULL
                AND tr.comments <> ''
            {DATE_FILTER.format(field="DATE(tr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="treater",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Treater reading comment",
    ),
    SourceDefinition(
        source_type="knock_out_reading_comment",
        required_tables={"knock_out_readings", "knock_outs"},
        query=f"""
            SELECT kor.id, ko.site_id, kor.knock_out_id AS entity_id,
                ko.name AS entity_name, kor.time AS event_date,
                kor.comments, kor.updated_at
            FROM knock_out_readings kor
            JOIN knock_outs ko ON ko.id = kor.knock_out_id
            WHERE ko.site_id = :site_id
                AND kor.comments IS NOT NULL
                AND kor.comments <> ''
            {DATE_FILTER.format(field="DATE(kor.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="knock_out",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Knock-out reading comment",
    ),
    SourceDefinition(
        source_type="pump_reading_comment",
        required_tables={"pump_readings", "pumps"},
        query=f"""
            SELECT pr.id, p.site_id, pr.pump_id AS entity_id,
                p.name AS entity_name, pr.time AS event_date,
                pr.comments, pr.updated_at
            FROM pump_readings pr
            JOIN pumps p ON p.id = pr.pump_id
            WHERE p.site_id = :site_id
                AND pr.comments IS NOT NULL
                AND pr.comments <> ''
            {DATE_FILTER.format(field="DATE(pr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="pump",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Pump reading comment",
    ),
    SourceDefinition(
        source_type="well_fluid_comment",
        required_tables={"well_fluids", "wells"},
        query=f"""
            SELECT wf.id, w.site_id, wf.well_id AS entity_id,
                w.name AS entity_name, wf.time AS event_date,
                wf.comments, wf.updated_at
            FROM well_fluids wf
            JOIN wells w ON w.id = wf.well_id
            WHERE w.site_id = :site_id
                AND wf.comments IS NOT NULL
                AND wf.comments <> ''
            {DATE_FILTER.format(field="DATE(wf.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="well",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Well fluid comment",
    ),
    SourceDefinition(
        source_type="well_injection_comment",
        required_tables={"well_injections", "wells"},
        query=f"""
            SELECT wi.id, w.site_id, wi.well_id AS entity_id,
                w.name AS entity_name, wi.time AS event_date,
                wi.comments, wi.updated_at
            FROM well_injections wi
            JOIN wells w ON w.id = wi.well_id
            WHERE w.site_id = :site_id
                AND wi.comments IS NOT NULL
                AND wi.comments <> ''
            {DATE_FILTER.format(field="DATE(wi.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="well",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Well injection comment",
    ),
    SourceDefinition(
        source_type="run_ticket_comment",
        required_tables={"run_tickets", "tanks"},
        query=f"""
            SELECT rt.id, t.site_id, rt.tank_id AS entity_id,
                t.name AS entity_name, rt.time AS event_date,
                rt.number, rt.comments, rt.updated_at
            FROM run_tickets rt
            JOIN tanks t ON t.id = rt.tank_id
            WHERE t.site_id = :site_id
                AND rt.comments IS NOT NULL
                AND rt.comments <> ''
            {DATE_FILTER.format(field="DATE(rt.time)")}
        """,
        text_fields=("entity_name", "number", "comments"),
        event_date_field="event_date",
        entity_type="tank",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Run ticket comment",
    ),
    SourceDefinition(
        source_type="water_draw_comment",
        required_tables={"water_draws", "tanks"},
        query=f"""
            SELECT wd.id, t.site_id, wd.tank_id AS entity_id,
                t.name AS entity_name, wd.time AS event_date,
                wd.comments, wd.updated_at
            FROM water_draws wd
            JOIN tanks t ON t.id = wd.tank_id
            WHERE t.site_id = :site_id
                AND wd.comments IS NOT NULL
                AND wd.comments <> ''
            {DATE_FILTER.format(field="DATE(wd.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="tank",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Water draw comment",
    ),
    SourceDefinition(
        source_type="mixed_tank_reading_comment",
        required_tables={"mixed_tank_readings", "tanks"},
        query=f"""
            SELECT mtr.id, t.site_id, mtr.tank_id AS entity_id,
                t.name AS entity_name, mtr.time AS event_date,
                mtr.comments, mtr.updated_at
            FROM mixed_tank_readings mtr
            JOIN tanks t ON t.id = mtr.tank_id
            WHERE t.site_id = :site_id
                AND mtr.comments IS NOT NULL
                AND mtr.comments <> ''
            {DATE_FILTER.format(field="DATE(mtr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="tank",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Mixed tank reading comment",
    ),
    SourceDefinition(
        source_type="non_linear_tank_reading_comment",
        required_tables={"non_linear_tank_readings", "tanks"},
        query=f"""
            SELECT nltr.id, t.site_id, nltr.tank_id AS entity_id,
                t.name AS entity_name, nltr.time AS event_date,
                nltr.comments, nltr.updated_at
            FROM non_linear_tank_readings nltr
            JOIN tanks t ON t.id = nltr.tank_id
            WHERE t.site_id = :site_id
                AND nltr.comments IS NOT NULL
                AND nltr.comments <> ''
            {DATE_FILTER.format(field="DATE(nltr.time)")}
        """,
        text_fields=("entity_name", "comments"),
        event_date_field="event_date",
        entity_type="tank",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Non-linear tank reading comment",
    ),
    SourceDefinition(
        source_type="well_history",
        required_tables={"well_histories", "wells"},
        query=f"""
            SELECT wh.id, w.site_id, wh.well_id, w.name AS well_name,
                wh.property, wh.old_value, wh.new_value,
                wh.changed_at AS event_date, wh.updated_at
            FROM well_histories wh
            JOIN wells w ON w.id = wh.well_id
            WHERE w.site_id = :site_id
            {DATE_FILTER.format(field="DATE(wh.changed_at)")}
        """,
        text_fields=("well_name", "property", "old_value", "new_value"),
        event_date_field="event_date",
        entity_type="well",
        entity_id_field="well_id",
        entity_name_field="well_name",
        title="Well history",
    ),
    SourceDefinition(
        source_type="alarm_log",
        required_tables={"alarm_logs", "data_points"},
        query=f"""
            SELECT al.id, dp.site_id, dp.id AS entity_id,
                CONCAT_WS(' - ', dp.facility_name, dp.device_name, dp.data_point_name) AS entity_name,
                al.event, al.comments,
                FROM_UNIXTIME(al.unix_timestamp) AS event_date,
                al.created_at AS updated_at
            FROM alarm_logs al
            JOIN data_points dp ON dp.id = al.data_point_id
            WHERE dp.site_id = :site_id
                AND (al.comments IS NOT NULL OR al.event IS NOT NULL)
            {DATE_FILTER.format(field="DATE(FROM_UNIXTIME(al.unix_timestamp))")}
        """,
        text_fields=("entity_name", "event", "comments"),
        event_date_field="event_date",
        entity_type="data_point",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Alarm log",
    ),
    SourceDefinition(
        source_type="legacy_alarm_log",
        required_tables={"alarm_logs", "alarm_data_points"},
        query=f"""
            SELECT al.id, dp.site_id, dp.id AS entity_id,
                CONCAT_WS(' - ', dp.facility_name, dp.device_name, dp.data_point_name) AS entity_name,
                al.event, al.comments,
                FROM_UNIXTIME(al.unix_timestamp) AS event_date,
                al.created_at AS updated_at
            FROM alarm_logs al
            JOIN alarm_data_points dp ON dp.id = al.alarm_data_point_id
            WHERE dp.site_id = :site_id
                AND (al.comments IS NOT NULL OR al.event IS NOT NULL)
            {DATE_FILTER.format(field="DATE(FROM_UNIXTIME(al.unix_timestamp))")}
        """,
        text_fields=("entity_name", "event", "comments"),
        event_date_field="event_date",
        entity_type="data_point",
        entity_id_field="entity_id",
        entity_name_field="entity_name",
        title="Alarm log",
    ),
    SourceDefinition(
        source_type="rod_pump_note",
        required_tables={"rod_pump_notes", "wells"},
        query=f"""
                SELECT rpn.id, w.site_id, rpn.well_id, w.name AS well_name,
                    rpn.note, DATE(rpn.updated_at) AS event_date, rpn.updated_at
                FROM rod_pump_notes rpn
                JOIN wells w ON w.id = rpn.well_id
                WHERE w.site_id = :site_id
                {DATE_FILTER.format(field="DATE(rpn.updated_at)")}
            """,
        text_fields=("well_name", "note"),
        event_date_field="event_date",
        entity_type="well",
        entity_id_field="well_id",
        entity_name_field="well_name",
        title="Rod pump note",
    ),
)


def source_definition(source_type: str) -> SourceDefinition:
    for source in SOURCE_DEFINITIONS:
        if source.source_type == source_type:
            return source
    raise OperationalTextExtractorError(f"Unsupported source type: {source_type}")


def supported_source_types() -> set[str]:
    return {source.source_type for source in SOURCE_DEFINITIONS}
