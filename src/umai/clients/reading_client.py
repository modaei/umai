from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import bindparam, create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings


class ReadingClientError(RuntimeError):
    """Raised when a reading query is invalid or cannot be completed."""


@dataclass(frozen=True)
class ReadingDefinition:
    label: str
    table: str
    base_table: str
    foreign_key: str
    base_key: str
    fields: tuple[str, ...]
    base_filter: str = ""
    supports_missing: bool = True


@dataclass(frozen=True)
class EquipmentDefinition:
    label: str
    table: str
    key: str
    metadata_fields: tuple[str, ...]


READING_DEFINITIONS: dict[str, ReadingDefinition] = {
    "lact": ReadingDefinition(
        "LACT readings",
        "lact_readings",
        "lacts",
        "lact_id",
        "id",
        ("reading", "temperature", "bs_w", "comments"),
    ),
    "flare": ReadingDefinition(
        "Flare readings",
        "flare_readings",
        "flares",
        "flare_id",
        "id",
        ("pressure", "volume", "comments"),
    ),
    "linear_tank": ReadingDefinition(
        "Linear tank readings",
        "linear_tank_readings",
        "tanks",
        "tank_id",
        "id",
        ("level", "feet", "inches", "temperature", "comments"),
        "AND b.type = 'linear-volume'",
    ),
    "mixed_tank": ReadingDefinition(
        "Mixed tank readings",
        "mixed_tank_readings",
        "tanks",
        "tank_id",
        "id",
        (
            "top_level_feet",
            "top_level_inches",
            "water_level_feet",
            "water_level_inches",
            "comments",
        ),
        "AND b.type = 'mixed-water-oil'",
    ),
    "non_linear_tank": ReadingDefinition(
        "Non-linear tank readings",
        "non_linear_tank_readings",
        "tanks",
        "tank_id",
        "id",
        ("initial_feet", "initial_inches", "final_feet", "final_inches", "comments"),
        "AND b.type = 'non-linear-volume'",
    ),
    "water_plant": ReadingDefinition(
        "Water plant readings",
        "water_plant_readings",
        "water_plants",
        "water_plant_id",
        "id",
        (
            "meter_reading",
            "flow_rate",
            "suction_pressure",
            "discharge_pressure",
            "comments",
        ),
    ),
    "flow_meter": ReadingDefinition(
        "Flow meter readings",
        "flow_meter_readings",
        "flow_meters",
        "flow_meter_id",
        "id",
        ("total", "flow", "odometer", "comments"),
    ),
    "well_test": ReadingDefinition(
        "Well tests",
        "well_tests",
        "wells",
        "well_id",
        "id",
        (
            "oil",
            "water",
            "gas",
            "pip",
            "m_temp",
            "amps",
            "tbgp",
            "csgp",
            "runtime",
            "comments",
        ),
        supports_missing=False,
    ),
    "well_fluid": ReadingDefinition(
        "Well fluid readings",
        "well_fluids",
        "wells",
        "well_id",
        "id",
        ("level", "comments"),
        supports_missing=False,
    ),
    "well_injection": ReadingDefinition(
        "Well injection readings",
        "well_injections",
        "wells",
        "well_id",
        "id",
        ("flow_rate", "total", "type", "tbg", "csg", "comments"),
        supports_missing=False,
    ),
    "run_ticket": ReadingDefinition(
        "Run tickets",
        "run_tickets",
        "tanks",
        "tank_id",
        "id",
        (
            "number",
            "measurement_method",
            "obs_grav",
            "obs_temp",
            "b_s_w",
            "corr_grav",
            "gov",
            "nsv",
            "initial_feet",
            "initial_inches",
            "initial_qtr",
            "final_feet",
            "final_inches",
            "final_qtr",
            "initial_volume",
            "final_volume",
            "initial_meter_reading",
            "final_meter_reading",
            "comments",
        ),
        supports_missing=False,
    ),
    "water_draw": ReadingDefinition(
        "Water draws",
        "water_draws",
        "tanks",
        "tank_id",
        "id",
        ("initial_feet", "initial_inches", "final_feet", "final_inches", "comments"),
        supports_missing=False,
    ),
    "treater": ReadingDefinition(
        "Treater readings",
        "treater_readings",
        "treaters",
        "treater_id",
        "id",
        ("oil_intake", "pressure", "temperature", "comments"),
    ),
    "knock_out": ReadingDefinition(
        "Knock-out readings",
        "knock_out_readings",
        "knock_outs",
        "knock_out_id",
        "id",
        ("inlet", "oil_off", "comments"),
    ),
    "pump": ReadingDefinition(
        "Pump readings",
        "pump_readings",
        "pumps",
        "pump_id",
        "id",
        ("suction_pressure", "discharge_pressure", "comments"),
    ),
}

MISSING_READINGS_REPORT_FUNCTION_NAME = "hartzog_daily_missing"


BASE_ENTITY_LABELS = {
    "flares": "Flare",
    "flow_meters": "Flow Meter",
    "knock_outs": "Knock Out",
    "lacts": "LACT",
    "pumps": "Pump",
    "tanks": "Tank",
    "treaters": "Treater",
    "water_plants": "Water Plant",
    "wells": "Well",
}

TANK_VOLUME_FIELDS = (
    "bbl_foot",
    "volume",
    "oil_volume",
    "water_volume",
    "total_volume",
    "initial_water_volume",
    "final_water_volume",
    "initial_oil_volume",
    "final_oil_volume",
    "recoverable_oil_volume",
    "initial_recoverable_oil_volume",
    "final_recoverable_oil_volume",
    "recoverable_oil_difference",
    "volume_status",
    "content_type",
)

EQUIPMENT_METADATA_FIELDS: dict[str, tuple[str, ...]] = {
    "lact": ("monitored", "disable_reading"),
    "flare": ("monitored", "disable_reading"),
    "linear_tank": (
        "monitored",
        "disable_reading",
        "type",
        "contents",
        "unusable_height",
        "bbl_foot",
        "non_linear_volume_mapping_id",
    ),
    "mixed_tank": (
        "monitored",
        "disable_reading",
        "type",
        "contents",
        "unusable_height",
        "bbl_foot",
        "non_linear_volume_mapping_id",
    ),
    "non_linear_tank": (
        "monitored",
        "disable_reading",
        "type",
        "contents",
        "unusable_height",
        "bbl_foot",
        "non_linear_volume_mapping_id",
    ),
    "water_plant": ("monitored", "disable_reading"),
    "flow_meter": (
        "monitored",
        "disable_reading",
        "type",
        "measurement_method",
        "water_plant_id",
        "flared_gas_report",
        "water_transfer_report",
    ),
    "treater": ("monitored", "disable_reading"),
    "knock_out": ("monitored", "disable_reading"),
    "pump": ("monitored", "disable_reading", "type", "water_plant_id"),
}

EQUIPMENT_SEARCH_READING_TYPES = tuple(EQUIPMENT_METADATA_FIELDS)

EQUIPMENT_DEFINITIONS: dict[str, EquipmentDefinition] = {
    "lact": EquipmentDefinition("LACTs", "lacts", "id", EQUIPMENT_METADATA_FIELDS["lact"]),
    "flare": EquipmentDefinition("Flares", "flares", "id", EQUIPMENT_METADATA_FIELDS["flare"]),
    "tank": EquipmentDefinition("Tanks", "tanks", "id", EQUIPMENT_METADATA_FIELDS["linear_tank"]),
    "water_plant": EquipmentDefinition(
        "Water plants",
        "water_plants",
        "id",
        EQUIPMENT_METADATA_FIELDS["water_plant"],
    ),
    "flow_meter": EquipmentDefinition(
        "Flow meters",
        "flow_meters",
        "id",
        EQUIPMENT_METADATA_FIELDS["flow_meter"],
    ),
    "treater": EquipmentDefinition(
        "Treaters",
        "treaters",
        "id",
        EQUIPMENT_METADATA_FIELDS["treater"],
    ),
    "knock_out": EquipmentDefinition(
        "Knock-outs",
        "knock_outs",
        "id",
        EQUIPMENT_METADATA_FIELDS["knock_out"],
    ),
    "pump": EquipmentDefinition("Pumps", "pumps", "id", EQUIPMENT_METADATA_FIELDS["pump"]),
}


def list_supported_reading_types() -> list[dict[str, Any]]:
    return [
        {
            "reading_type": "tank",
            "label": "All tank readings",
            "supports_missing": True,
        },
        *[
            {
                "reading_type": key,
                "label": definition.label,
                "supports_missing": definition.supports_missing,
            }
            for key, definition in READING_DEFINITIONS.items()
        ],
    ]


class ReadingClient:
    def __init__(self, engine: Engine, max_rows: int = 500):
        self.engine = engine
        self.max_rows = max_rows
        self._missing_reading_exclusion_cache: dict[int, dict[str, tuple[int, ...]]] = {}
        self._table_columns_cache: dict[str, set[str]] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "ReadingClient":
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

    def list_reading_types(self) -> list[dict[str, Any]]:
        return list_supported_reading_types()

    def get_readings_for_date(
        self, site_id: int, reading_type: str, reading_date: str
    ) -> dict[str, Any]:
        day = self._parse_date(reading_date)
        if reading_type == "tank":
            return self._get_all_tank_readings(site_id, day)
        definition = self._definition(reading_type)
        rows = self._readings_for_day(site_id, definition, day)
        return {
            "reading_type": reading_type,
            "label": definition.label,
            "site_id": site_id,
            "date": day.isoformat(),
            "count": len(rows),
            "readings": self._compact_reading_rows(definition, rows),
        }

    def compare_readings_between_dates(
        self,
        site_id: int,
        reading_type: str,
        first_date: str,
        second_date: str,
    ) -> dict[str, Any]:
        first_day = self._parse_date(first_date)
        second_day = self._parse_date(second_date)
        if reading_type == "tank":
            return {
                "reading_type": "tank",
                "label": "All tank readings",
                "site_id": site_id,
                "first_date": first_day.isoformat(),
                "second_date": second_day.isoformat(),
                "groups": [
                    self.compare_readings_between_dates(
                        site_id, tank_type, first_date, second_date
                    )
                    for tank_type in (
                        "linear_tank",
                        "mixed_tank",
                        "non_linear_tank",
                    )
                ],
            }
        definition = self._definition(reading_type)
        first_rows = self._readings_for_day(site_id, definition, first_day)
        second_rows = self._readings_for_day(site_id, definition, second_day)
        return {
            "reading_type": reading_type,
            "label": definition.label,
            "site_id": site_id,
            "first_date": first_day.isoformat(),
            "second_date": second_day.isoformat(),
            "first_count": len(first_rows),
            "second_count": len(second_rows),
            "comparison_rows": self._comparison_rows(
                reading_type, definition, first_rows, second_rows
            ),
            "entity_summary": self._entity_summary(first_rows, second_rows),
        }

    def missing_readings_for_date(
        self, site_id: int, reading_type: str, reading_date: str
    ) -> dict[str, Any]:
        day = self._parse_date(reading_date)
        if reading_type == "tank":
            groups = [
                self.missing_readings_for_date(site_id, tank_type, reading_date)
                for tank_type in ("linear_tank", "mixed_tank", "non_linear_tank")
            ]
            return {
                "reading_type": "tank",
                "label": "All tank readings",
                "site_id": site_id,
                "date": day.isoformat(),
                "missing_count": sum(group["missing_count"] for group in groups),
                "groups": groups,
            }
        definition = self._definition(reading_type)
        if not definition.supports_missing:
            raise ReadingClientError(
                f"Missing-reading checks are not supported for {reading_type}."
            )

        start, end = self._day_bounds(day)
        ignored_ids = self._missing_reading_ignored_ids(site_id, reading_type)
        query = text(
            f"""
            SELECT b.name AS entity_name
            FROM {definition.base_table} b
            LEFT JOIN {definition.table} r
                ON b.{definition.base_key} = r.{definition.foreign_key}
                AND r.time >= :start_time
                AND r.time < :end_time
            WHERE r.{definition.foreign_key} IS NULL
                AND b.site_id = :site_id
                AND COALESCE(b.disable_reading, 0) = 0
                {("AND b.id NOT IN :ignored_ids" if ignored_ids else "")}
                {definition.base_filter}
            ORDER BY b.name
            LIMIT :limit
            """
        )
        params: dict[str, Any] = {
            "site_id": site_id,
            "start_time": start,
            "end_time": end,
        }
        if ignored_ids:
            params["ignored_ids"] = list(ignored_ids)
            query = query.bindparams(bindparam("ignored_ids", expanding=True))
        rows = self._execute(query, **params)
        return {
            "reading_type": reading_type,
            "label": definition.label,
            "site_id": site_id,
            "date": day.isoformat(),
            "missing_count": len(rows),
            "missing_entities": self._with_entity_display(definition, rows),
        }

    def all_missing_readings_for_date(
        self, site_id: int, reading_date: str
    ) -> dict[str, Any]:
        day = self._parse_date(reading_date)
        groups = []
        skipped = []
        for reading_type, definition in READING_DEFINITIONS.items():
            if not definition.supports_missing:
                continue
            try:
                groups.append(
                    self.missing_readings_for_date(site_id, reading_type, reading_date)
                )
            except ReadingClientError as exc:
                skipped.append({"reading_type": reading_type, "error": str(exc)})
        groups = [group for group in groups if group["missing_count"] > 0]
        return {
            "reading_type": "all_supported_missing_readings",
            "label": "All supported missing readings",
            "site_id": site_id,
            "date": day.isoformat(),
            "missing_count": sum(group["missing_count"] for group in groups),
            "groups": groups,
            "skipped": skipped,
        }

    def all_missing_readings_for_range(
        self, site_id: int, start_date: str, end_date: str
    ) -> dict[str, Any]:
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if end_day < start_day:
            raise ReadingClientError("end_date must be on or after start_date.")
        if (end_day - start_day).days > 31:
            raise ReadingClientError(
                "Missing-reading range checks are limited to 32 days."
            )

        dates = []
        total_missing_count = 0
        current_day = start_day
        while current_day <= end_day:
            day_result = self.all_missing_readings_for_date(
                site_id, current_day.isoformat()
            )
            total_missing_count += int(day_result["missing_count"])
            dates.append(
                {
                    "date": current_day.isoformat(),
                    "missing_count": day_result["missing_count"],
                    "groups": day_result["groups"],
                    "skipped": day_result["skipped"],
                }
            )
            current_day += timedelta(days=1)

        return {
            "reading_type": "all_supported_missing_readings",
            "label": "All supported missing readings",
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "day_count": len(dates),
            "missing_count": total_missing_count,
            "dates": dates,
        }

    def search_well_tests(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
        well_name: str | None = None,
        min_oil: float | None = None,
        max_oil: float | None = None,
        min_water: float | None = None,
        max_water: float | None = None,
        min_gas: float | None = None,
        max_gas: float | None = None,
        min_runtime: float | None = None,
        max_runtime: float | None = None,
    ) -> dict[str, Any]:
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if end_day < start_day:
            raise ReadingClientError("end_date must be on or after start_date.")

        start_time = datetime.combine(start_day, time.min)
        end_time = datetime.combine(end_day + timedelta(days=1), time.min)
        conditions = [
            "w.site_id = :site_id",
            "wt.time >= :start_time",
            "wt.time < :end_time",
        ]
        params: dict[str, Any] = {
            "site_id": site_id,
            "start_time": start_time,
            "end_time": end_time,
        }

        if well_name:
            conditions.append("w.name LIKE :well_name")
            params["well_name"] = f"%{well_name}%"

        numeric_filters = {
            "min_oil": ("wt.oil", ">=", min_oil),
            "max_oil": ("wt.oil", "<=", max_oil),
            "min_water": ("wt.water", ">=", min_water),
            "max_water": ("wt.water", "<=", max_water),
            "min_gas": ("wt.gas", ">=", min_gas),
            "max_gas": ("wt.gas", "<=", max_gas),
            "min_runtime": ("wt.runtime", ">=", min_runtime),
            "max_runtime": ("wt.runtime", "<=", max_runtime),
        }
        applied_filters: dict[str, Any] = {}
        for name, (column, operator, value) in numeric_filters.items():
            if value is None:
                continue
            conditions.append(f"{column} {operator} :{name}")
            params[name] = value
            applied_filters[name] = value

        where_sql = "\n                AND ".join(conditions)
        query = text(
            f"""
            SELECT wt.time,
                w.name AS entity_name,
                wt.oil,
                wt.water,
                wt.gas,
                wt.pip,
                wt.m_temp,
                wt.amps,
                wt.tbgp,
                wt.csgp,
                wt.runtime,
                wt.comments
            FROM well_tests wt
            JOIN wells w
                ON wt.well_id = w.id
            WHERE {where_sql}
            ORDER BY wt.time, w.name
            LIMIT :limit
            """
        )
        rows = self._execute(query, **params)
        definition = self._definition("well_test")
        return {
            "reading_type": "well_test",
            "label": "Well tests",
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "filters": {
                **({"well_name": well_name} if well_name else {}),
                **applied_filters,
            },
            "count": len(rows),
            "well_tests": self._compact_range_reading_rows(
                definition, self._with_entity_display(definition, rows)
            ),
        }

    def analyze_well_tests(
        self,
        site_id: int,
        analysis_mode: str,
        group_by: str,
        start_date: str | None = None,
        end_date: str | None = None,
        well_name: str | None = None,
        battery_name: str | None = None,
        test_count: int | None = None,
    ) -> dict[str, Any]:
        """Analyze well tests deterministically by well, battery, or selected site."""
        if analysis_mode not in {
            "latest_previous",
            "range_summary",
            "recent_tests",
            "range_sequence",
        }:
            raise ReadingClientError("Unsupported well-test analysis mode.")
        if group_by not in {"well", "battery", "site"}:
            raise ReadingClientError("group_by must be well, battery, or site.")
        if (start_date is None) != (end_date is None):
            raise ReadingClientError("start_date and end_date must be provided together.")
        start_day = self._parse_date(start_date) if start_date else None
        end_day = self._parse_date(end_date) if end_date else None
        if start_day and end_day and end_day < start_day:
            raise ReadingClientError("end_date must be on or after start_date.")
        if analysis_mode == "range_summary" and start_day is None:
            raise ReadingClientError("range_summary requires start_date and end_date.")
        if analysis_mode == "range_sequence" and start_day is None:
            raise ReadingClientError("range_sequence requires start_date and end_date.")
        if analysis_mode in {"recent_tests", "range_sequence"} and group_by != "well":
            raise ReadingClientError(
                f"{analysis_mode} compares tests per well and requires group_by=well."
            )
        if analysis_mode == "recent_tests" and (
            test_count is None or not 2 <= test_count <= 10
        ):
            raise ReadingClientError("recent_tests requires test_count between 2 and 10.")
        if analysis_mode != "recent_tests" and test_count is not None:
            raise ReadingClientError("test_count is only supported by recent_tests.")

        conditions = ["w.site_id = :site_id"]
        params: dict[str, Any] = {"site_id": site_id}
        if start_day is not None and end_day is not None:
            conditions.extend(["wt.time >= :start_time", "wt.time < :end_time"])
            params["start_time"] = datetime.combine(start_day, time.min)
            params["end_time"] = datetime.combine(end_day + timedelta(days=1), time.min)
        if well_name:
            conditions.append("LOWER(w.name) LIKE LOWER(:well_name)")
            params["well_name"] = f"%{well_name}%"
        if battery_name:
            conditions.append("LOWER(b.name) LIKE LOWER(:battery_name)")
            params["battery_name"] = f"%{battery_name}%"
        where_sql = " AND ".join(conditions)

        if analysis_mode == "range_summary":
            return self._summarize_well_tests_by_group(
                site_id=site_id,
                group_by=group_by,
                where_sql=where_sql,
                params=params,
                start_day=start_day,
                end_day=end_day,
            )

        if analysis_mode in {"recent_tests", "range_sequence"}:
            return self._sequence_well_tests(
                site_id=site_id,
                analysis_mode=analysis_mode,
                where_sql=where_sql,
                params=params,
                start_day=start_day,
                end_day=end_day,
                well_name=well_name,
                battery_name=battery_name,
                test_count=test_count,
            )

        query = text(
            f"""
            SELECT ranked.well_name,
                ranked.battery_name,
                ranked.test_time,
                ranked.oil,
                ranked.water,
                ranked.gas,
                ranked.runtime,
                ranked.rn
            FROM (
                SELECT w.name AS well_name,
                    COALESCE(b.name, 'No Battery') AS battery_name,
                    wt.time AS test_time,
                    wt.oil,
                    wt.water,
                    wt.gas,
                    wt.runtime,
                    ROW_NUMBER() OVER (
                        PARTITION BY w.id ORDER BY wt.time DESC, wt.id DESC
                    ) AS rn
                FROM well_tests wt
                JOIN wells w ON w.id = wt.well_id
                LEFT JOIN batteries b ON b.id = w.battery_id
                WHERE {where_sql}
            ) ranked
            WHERE ranked.rn <= 2
            ORDER BY ranked.well_name, ranked.rn
            LIMIT :limit
            """
        )
        rows = self._execute(query, **params)
        comparisons = self._well_test_comparisons(rows)
        groups = self._roll_up_well_test_comparisons(comparisons, group_by)
        return {
            "analysis_mode": analysis_mode,
            "group_by": group_by,
            "site_id": site_id,
            "start_date": start_day.isoformat() if start_day else None,
            "end_date": end_day.isoformat() if end_day else None,
            "filters": {
                **({"well_name": well_name} if well_name else {}),
                **({"battery_name": battery_name} if battery_name else {}),
            },
            "well_count": len(comparisons),
            "group_count": len(groups),
            "groups": groups,
        }

    def _sequence_well_tests(
        self,
        *,
        site_id: int,
        analysis_mode: str,
        where_sql: str,
        params: dict[str, Any],
        start_day: date | None,
        end_day: date | None,
        well_name: str | None,
        battery_name: str | None,
        test_count: int | None,
    ) -> dict[str, Any]:
        """Return chronological per-well tests and deterministic adjacent deltas."""
        if analysis_mode == "recent_tests":
            query = text(
                f"""
                SELECT ranked.test_id,
                    ranked.well_name,
                    ranked.battery_name,
                    ranked.test_time,
                    ranked.oil,
                    ranked.water,
                    ranked.gas,
                    ranked.runtime
                FROM (
                    SELECT wt.id AS test_id,
                        w.name AS well_name,
                        COALESCE(b.name, 'No Battery') AS battery_name,
                        wt.time AS test_time,
                        wt.oil,
                        wt.water,
                        wt.gas,
                        wt.runtime,
                        ROW_NUMBER() OVER (
                            PARTITION BY w.id ORDER BY wt.time DESC, wt.id DESC
                        ) AS rn
                    FROM well_tests wt
                    JOIN wells w ON w.id = wt.well_id
                    LEFT JOIN batteries b ON b.id = w.battery_id
                    WHERE {where_sql}
                ) ranked
                WHERE ranked.rn <= :sequence_test_count
                ORDER BY ranked.well_name, ranked.test_time, ranked.test_id
                LIMIT :fetch_limit
                """
            )
            params = {
                **params,
                "sequence_test_count": test_count,
                "fetch_limit": self.max_rows + 1,
            }
        else:
            query = text(
                f"""
                SELECT wt.id AS test_id,
                    w.name AS well_name,
                    COALESCE(b.name, 'No Battery') AS battery_name,
                    wt.time AS test_time,
                    wt.oil,
                    wt.water,
                    wt.gas,
                    wt.runtime
                FROM well_tests wt
                JOIN wells w ON w.id = wt.well_id
                LEFT JOIN batteries b ON b.id = w.battery_id
                WHERE {where_sql}
                ORDER BY w.name, wt.time, wt.id
                LIMIT :fetch_limit
                """
            )
            params = {**params, "fetch_limit": self.max_rows + 1}

        rows = self._execute(query, **params)
        truncated = len(rows) > self.max_rows
        selected_rows = rows[: self.max_rows]
        groups = self._well_test_sequences(selected_rows)
        return {
            "analysis_mode": analysis_mode,
            "group_by": "well",
            "site_id": site_id,
            "start_date": start_day.isoformat() if start_day else None,
            "end_date": end_day.isoformat() if end_day else None,
            "filters": {
                **({"well_name": well_name} if well_name else {}),
                **({"battery_name": battery_name} if battery_name else {}),
            },
            **({"requested_test_count": test_count} if test_count else {}),
            "well_count": len(groups),
            "test_count": sum(group["test_count"] for group in groups),
            "truncated": truncated,
            "groups": groups,
        }

    @classmethod
    def _well_test_sequences(
        cls, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Group tests by well and compare each test with its selected predecessor."""
        by_well: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_well.setdefault(str(row["well_name"]), []).append(row)

        groups = []
        for well_name, tests in sorted(by_well.items()):
            ordered = sorted(
                tests,
                key=lambda item: (str(item["test_time"]), int(item["test_id"])),
            )
            sequence = []
            previous: dict[str, Any] | None = None
            for test in ordered:
                item: dict[str, Any] = {"date": str(test["test_time"])[:10]}
                for metric in ("oil", "water", "gas", "runtime"):
                    value = test.get(metric)
                    item[metric] = value
                    previous_value = previous.get(metric) if previous else None
                    item[f"{metric}_change"] = (
                        round(float(value) - float(previous_value), 3)
                        if value is not None and previous_value is not None
                        else None
                    )
                sequence.append(cls._round_numeric_values(item))
                previous = test
            groups.append(
                {
                    "group_name": well_name,
                    "well_name": well_name,
                    "battery_name": ordered[-1]["battery_name"],
                    "test_count": len(sequence),
                    "tests": sequence,
                }
            )
        return groups

    def _summarize_well_tests_by_group(
        self,
        *,
        site_id: int,
        group_by: str,
        where_sql: str,
        params: dict[str, Any],
        start_day: date,
        end_day: date,
    ) -> dict[str, Any]:
        """Aggregate a bounded well-test range using fixed, trusted SQL."""
        group_expression = {
            "well": "w.name",
            "battery": "COALESCE(b.name, 'No Battery')",
            "site": "'Selected site'",
        }[group_by]
        query = text(
            f"""
            SELECT {group_expression} AS group_name,
                COUNT(*) AS test_count,
                COUNT(DISTINCT w.id) AS well_count,
                SUM(wt.oil) AS oil_sum,
                AVG(wt.oil) AS oil_average,
                MIN(wt.oil) AS oil_minimum,
                MAX(wt.oil) AS oil_maximum,
                SUM(wt.water) AS water_sum,
                AVG(wt.water) AS water_average,
                MIN(wt.water) AS water_minimum,
                MAX(wt.water) AS water_maximum,
                SUM(wt.gas) AS gas_sum,
                AVG(wt.gas) AS gas_average,
                MIN(wt.gas) AS gas_minimum,
                MAX(wt.gas) AS gas_maximum,
                AVG(wt.runtime) AS runtime_average
            FROM well_tests wt
            JOIN wells w ON w.id = wt.well_id
            LEFT JOIN batteries b ON b.id = w.battery_id
            WHERE {where_sql}
            GROUP BY {group_expression}
            ORDER BY group_name
            LIMIT :limit
            """
        )
        groups = [self._round_numeric_values(row) for row in self._execute(query, **params)]
        return {
            "analysis_mode": "range_summary",
            "group_by": group_by,
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "group_count": len(groups),
            "groups": groups,
        }

    @classmethod
    def _well_test_comparisons(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pair each well's latest test with its immediately previous test."""
        by_well: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_well.setdefault(str(row["well_name"]), []).append(row)
        comparisons = []
        for well_name, tests in sorted(by_well.items()):
            ordered = sorted(tests, key=lambda item: int(item["rn"]))
            if len(ordered) < 2:
                continue
            latest, previous = ordered[0], ordered[1]
            item: dict[str, Any] = {
                "group_name": well_name,
                "well_name": well_name,
                "battery_name": latest["battery_name"],
                "latest_date": str(latest["test_time"])[:10],
                "previous_date": str(previous["test_time"])[:10],
            }
            for metric in ("oil", "water", "gas", "runtime"):
                latest_value = latest.get(metric)
                previous_value = previous.get(metric)
                item[f"latest_{metric}"] = latest_value
                item[f"previous_{metric}"] = previous_value
                item[f"{metric}_change"] = (
                    round(float(latest_value) - float(previous_value), 3)
                    if latest_value is not None and previous_value is not None
                    else None
                )
            comparisons.append(cls._round_numeric_values(item))
        return comparisons

    @classmethod
    def _roll_up_well_test_comparisons(
        cls, comparisons: list[dict[str, Any]], group_by: str
    ) -> list[dict[str, Any]]:
        """Return per-well rows or aggregate paired tests by battery/site."""
        if group_by == "well":
            return comparisons
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in comparisons:
            key = item["battery_name"] if group_by == "battery" else "Selected site"
            grouped.setdefault(str(key), []).append(item)
        results = []
        for group_name, items in sorted(grouped.items()):
            result: dict[str, Any] = {
                "group_name": group_name,
                "well_count": len(items),
            }
            for metric in ("oil", "water", "gas", "runtime"):
                latest_values = [item[f"latest_{metric}"] for item in items if item[f"latest_{metric}"] is not None]
                previous_values = [item[f"previous_{metric}"] for item in items if item[f"previous_{metric}"] is not None]
                result[f"latest_{metric}_sum"] = sum(latest_values)
                result[f"previous_{metric}_sum"] = sum(previous_values)
                result[f"{metric}_sum_change"] = sum(latest_values) - sum(previous_values)
                result[f"latest_{metric}_average"] = sum(latest_values) / len(latest_values) if latest_values else None
                result[f"previous_{metric}_average"] = sum(previous_values) / len(previous_values) if previous_values else None
            results.append(cls._round_numeric_values(result))
        return results

    @staticmethod
    def _round_numeric_values(row: dict[str, Any]) -> dict[str, Any]:
        """Normalize database numeric types for compact JSON tool output."""
        return {
            key: round(float(value), 3) if isinstance(value, (float, Decimal)) else value
            for key, value in row.items()
        }

    def search_readings(
        self,
        site_id: int,
        reading_type: str,
        start_date: str,
        end_date: str,
        entity_name: str | None = None,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if end_day < start_day:
            raise ReadingClientError("end_date must be on or after start_date.")
        if reading_type == "tank":
            groups = [
                self.search_readings(
                    site_id, tank_type, start_date, end_date, entity_name, filters
                )
                for tank_type in ("linear_tank", "mixed_tank", "non_linear_tank")
            ]
            return {
                "reading_type": "tank",
                "label": "All tank readings",
                "site_id": site_id,
                "start_date": start_day.isoformat(),
                "end_date": end_day.isoformat(),
                "count": sum(group["count"] for group in groups),
                "groups": groups,
            }

        definition = self._definition(reading_type)
        start_time = datetime.combine(start_day, time.min)
        end_time = datetime.combine(end_day + timedelta(days=1), time.min)
        conditions = [
            "b.site_id = :site_id",
            "r.time >= :start_time",
            "r.time < :end_time",
        ]
        if definition.base_filter:
            conditions.append(definition.base_filter.removeprefix("AND "))

        params: dict[str, Any] = {
            "site_id": site_id,
            "start_time": start_time,
            "end_time": end_time,
        }
        if entity_name:
            conditions.append("b.name LIKE :entity_name")
            params["entity_name"] = f"%{entity_name}%"

        applied_filters = self._filter_conditions(definition, filters or [], conditions, params)
        where_sql = "\n                AND ".join(conditions)
        field_sql = _reading_select_fields(definition)
        query = text(
            f"""
            SELECT r.time,
                b.name AS entity_name,
                {field_sql}
            FROM {definition.table} r
            JOIN {definition.base_table} b
                ON r.{definition.foreign_key} = b.{definition.base_key}
            WHERE {where_sql}
            ORDER BY r.time, b.name
            LIMIT :limit
            """
        )
        rows = self._execute(query, **params)
        rows = self._with_tank_volumes(reading_type, rows)
        return {
            "reading_type": reading_type,
            "label": definition.label,
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "filters": {
                **({"entity_name": entity_name} if entity_name else {}),
                **({"field_filters": applied_filters} if applied_filters else {}),
            },
            "count": len(rows),
            "readings": self._compact_range_reading_rows(
                definition, self._with_entity_display(definition, rows)
            ),
        }

    def search_tank_readings(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
        battery_name: str | None = None,
        tank_name: str | None = None,
        tank_key: str | None = None,
        tank_type: str | None = None,
        contains: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Search tank readings with tank/battery metadata and computed volumes.

        This is intentionally separate from the generic `search_readings` method.
        Tank questions commonly filter by battery relation or computed volume
        fields, neither of which maps cleanly to raw reading-table columns.
        """
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if end_day < start_day:
            raise ReadingClientError("end_date must be on or after start_date.")

        normalized_type = _normalize_tank_type(tank_type)
        reading_types = (
            [normalized_type]
            if normalized_type
            else ["linear_tank", "mixed_tank", "non_linear_tank"]
        )
        normalized_contains = _normalize_contains(contains)
        filters = filters or []
        rows: list[dict[str, Any]] = []
        for reading_type in reading_types:
            rows.extend(
                self._search_tank_readings_for_type(
                    site_id=site_id,
                    reading_type=reading_type,
                    start_day=start_day,
                    end_day=end_day,
                    battery_name=battery_name,
                    tank_name=tank_name,
                    tank_key=tank_key,
                    monitored=monitored,
                    disable_reading=disable_reading,
                )
            )

        filtered_rows = [
            row
            for row in rows
            if _matches_contains(row, normalized_contains)
            and _matches_computed_filters(row, filters)
        ]
        return {
            "reading_type": "tank",
            "label": "Tank readings",
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "filters": {
                **({"battery_name": battery_name} if battery_name else {}),
                **({"tank_name": tank_name} if tank_name else {}),
                **({"tank_key": tank_key} if tank_key else {}),
                **({"tank_type": normalized_type} if normalized_type else {}),
                **({"contains": normalized_contains} if normalized_contains else {}),
                **({"monitored": monitored} if monitored is not None else {}),
                **(
                    {"disable_reading": disable_reading}
                    if disable_reading is not None
                    else {}
                ),
                **({"computed_filters": filters} if filters else {}),
            },
            "count": len(filtered_rows),
            "readings": filtered_rows[: self.max_rows],
        }

    def search_equipment_readings(
        self,
        site_id: int,
        reading_type: str,
        start_date: str,
        end_date: str,
        battery_name: str | None = None,
        entity_name: str | None = None,
        entity_key: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        equipment_filters: list[dict[str, Any]] | None = None,
        reading_filters: list[dict[str, Any]] | None = None,
        computed_filters: list[dict[str, Any]] | None = None,
        contains: str | None = None,
    ) -> dict[str, Any]:
        """Search readings with equipment metadata and battery relation filters.

        This is the relation-aware counterpart to `search_readings`. It uses
        whitelisted reading/base-table fields and joins batteries by the actual
        database relation instead of relying on equipment names.
        """
        start_day = self._parse_date(start_date)
        end_day = self._parse_date(end_date)
        if end_day < start_day:
            raise ReadingClientError("end_date must be on or after start_date.")

        reading_types = (
            ["linear_tank", "mixed_tank", "non_linear_tank"]
            if reading_type == "tank"
            else [_normalize_equipment_reading_type(reading_type)]
        )
        normalized_contains = _normalize_contains(contains)
        equipment_filters = equipment_filters or []
        reading_filters = reading_filters or []
        computed_filters = computed_filters or []
        rows: list[dict[str, Any]] = []
        applied_equipment_filters: list[dict[str, Any]] = []
        applied_reading_filters: list[dict[str, Any]] = []

        for resolved_type in reading_types:
            if resolved_type not in EQUIPMENT_SEARCH_READING_TYPES:
                raise ReadingClientError(
                    f"Equipment search is not supported for {reading_type}."
                )
            result = self._search_equipment_readings_for_type(
                site_id=site_id,
                reading_type=resolved_type,
                start_day=start_day,
                end_day=end_day,
                battery_name=battery_name,
                entity_name=entity_name,
                entity_key=entity_key,
                monitored=monitored,
                disable_reading=disable_reading,
                equipment_filters=equipment_filters,
                reading_filters=reading_filters,
            )
            rows.extend(result["rows"])
            applied_equipment_filters.extend(result["equipment_filters"])
            applied_reading_filters.extend(result["reading_filters"])

        filtered_rows = [
            row
            for row in rows
            if _matches_contains(row, normalized_contains)
            and _matches_computed_filters(row, computed_filters)
        ]
        return {
            "reading_type": reading_type,
            "label": "Equipment readings",
            "site_id": site_id,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "filters": {
                **({"battery_name": battery_name} if battery_name else {}),
                **({"entity_name": entity_name} if entity_name else {}),
                **({"entity_key": entity_key} if entity_key else {}),
                **({"monitored": monitored} if monitored is not None else {}),
                **(
                    {"disable_reading": disable_reading}
                    if disable_reading is not None
                    else {}
                ),
                **({"contains": normalized_contains} if normalized_contains else {}),
                **(
                    {"equipment_filters": applied_equipment_filters}
                    if applied_equipment_filters
                    else {}
                ),
                **(
                    {"reading_filters": applied_reading_filters}
                    if applied_reading_filters
                    else {}
                ),
                **(
                    {"computed_filters": computed_filters}
                    if computed_filters
                    else {}
                ),
            },
            "count": len(filtered_rows),
            "readings": filtered_rows[: self.max_rows],
        }

    def list_equipment(
        self,
        site_id: int,
        equipment_type: str,
        battery_name: str | None = None,
        entity_name: str | None = None,
        entity_key: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        equipment_filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """List configured equipment entities without requiring readings."""
        definition = _equipment_definition(equipment_type)
        conditions = ["b.site_id = :site_id"]
        params: dict[str, Any] = {"site_id": site_id}
        if battery_name:
            _append_battery_condition(conditions, params, battery_name)
        if entity_name:
            conditions.append("LOWER(b.name) LIKE LOWER(:entity_name)")
            params["entity_name"] = f"%{entity_name}%"
        if entity_key:
            if not self._table_has_column(definition.table, "key"):
                raise ReadingClientError(f"{definition.label} do not have a key field.")
            conditions.append("LOWER(b.`key`) LIKE LOWER(:entity_key)")
            params["entity_key"] = f"%{entity_key}%"
        if monitored is not None:
            if self._table_has_column(definition.table, "monitored"):
                conditions.append("COALESCE(b.`monitored`, 0) = :monitored")
                params["monitored"] = 1 if monitored else 0
        if disable_reading is not None:
            if self._table_has_column(definition.table, "disable_reading"):
                conditions.append("COALESCE(b.`disable_reading`, 0) = :disable_reading")
                params["disable_reading"] = 1 if disable_reading else 0

        applied_filters = self._equipment_list_filter_conditions(
            definition, equipment_filters or [], conditions, params
        )
        metadata_fields = [
            field
            for field in definition.metadata_fields
            if self._table_has_column(definition.table, field)
        ]
        metadata_sql = "".join(
            f",\n                b.`{field}` AS equipment_{field}"
            for field in metadata_fields
        )
        key_sql = (
            ",\n                b.`key` AS entity_key"
            if self._table_has_column(definition.table, "key")
            else ""
        )
        battery_join_sql = _equipment_list_battery_join_sql(equipment_type)
        query = text(
            f"""
            SELECT b.{definition.key} AS entity_id,
                b.name AS entity_name
                {key_sql},
                bat.id AS battery_id,
                bat.name AS battery_name,
                bat.`key` AS battery_key
                {metadata_sql}
            FROM {definition.table} b
            {battery_join_sql}
            WHERE {" AND ".join(conditions)}
            ORDER BY bat.name, b.name
            LIMIT :limit
            """
        )
        rows = self._execute(query, **params)
        entities = [
            _compact_equipment_entity_row(equipment_type, definition, row, metadata_fields)
            for row in rows
        ]
        return {
            "equipment_type": equipment_type,
            "label": definition.label,
            "site_id": site_id,
            "filters": {
                **({"battery_name": battery_name} if battery_name else {}),
                **({"entity_name": entity_name} if entity_name else {}),
                **({"entity_key": entity_key} if entity_key else {}),
                **({"monitored": monitored} if monitored is not None else {}),
                **(
                    {"disable_reading": disable_reading}
                    if disable_reading is not None
                    else {}
                ),
                **({"equipment_filters": applied_filters} if applied_filters else {}),
            },
            "count": len(entities),
            "entities": entities,
        }

    def resolve_reading_entity(
        self,
        site_id: int,
        entity_name: str,
        reading_date: str | None = None,
        reading_type: str | None = None,
    ) -> dict[str, Any]:
        """Find reading-capable entities matching a user-supplied object name.

        This method exists because an object name alone is not always enough to
        select the correct reading table. For example, a phrase such as
        "Battery 2 Vent" may be guessed as a flare by the model even when the
        actual reading is stored on a flow meter. The resolver searches the
        configured reading base tables first and reports ambiguity explicitly.
        """
        normalized_name = entity_name.strip()
        if not normalized_name:
            raise ReadingClientError("entity_name is required.")

        day = self._parse_date(reading_date) if reading_date else None
        reading_types = self._resolution_reading_types(reading_type)
        candidates: list[dict[str, Any]] = []
        entity_name_variants = _entity_name_lookup_variants(
            normalized_name, reading_type
        )
        for candidate_reading_type in reading_types:
            definition = self._definition(candidate_reading_type)
            for lookup_name in entity_name_variants:
                for row in self._matching_entities(
                    site_id, definition, lookup_name
                ):
                    candidate = self._entity_candidate(
                        candidate_reading_type, definition, row
                    )
                    if day is not None:
                        candidate["has_reading"] = self._entity_has_reading(
                            site_id, definition, int(row["entity_id"]), day
                        )
                    candidates.append(candidate)

        candidates = self._deduplicate_candidates(candidates)
        candidates.sort(
            key=lambda candidate: (
                candidate["match_rank"],
                candidate["reading_type"],
                candidate["entity_display_name"],
            )
        )
        return {
            "entity_name": normalized_name,
            "reading_type": reading_type,
            "site_id": site_id,
            "date": day.isoformat() if day else None,
            "count": len(candidates),
            "candidates": candidates[: self.max_rows],
        }

    def get_reading_for_entity(
        self,
        site_id: int,
        entity_name: str,
        reading_date: str,
        reading_type: str | None = None,
    ) -> dict[str, Any]:
        """Resolve an entity name, then return that entity's reading for one day."""
        day = self._parse_date(reading_date)
        resolution = self.resolve_reading_entity(
            site_id, entity_name, day.isoformat(), reading_type
        )
        candidates = resolution["candidates"]
        if len(candidates) != 1:
            return {
                "reading_type": reading_type,
                "label": "Resolved entity reading",
                "site_id": site_id,
                "date": day.isoformat(),
                "entity_name": entity_name,
                "needs_clarification": len(candidates) > 1,
                "count": 0,
                "candidates": [
                    {
                        "reading_type": candidate["reading_type"],
                        "entity_type": candidate["entity_type"],
                        "entity_display_name": candidate["entity_display_name"],
                        **(
                            {"has_reading": candidate["has_reading"]}
                            if "has_reading" in candidate
                            else {}
                        ),
                    }
                    for candidate in candidates
                ],
                "readings": [],
            }

        candidate = candidates[0]
        definition = self._definition(candidate["reading_type"])
        rows = self._readings_for_day_entity(
            site_id, definition, int(candidate["entity_id"]), day
        )
        return {
            "reading_type": candidate["reading_type"],
            "label": definition.label,
            "site_id": site_id,
            "date": day.isoformat(),
            "entity": {
                "reading_type": candidate["reading_type"],
                "entity_type": candidate["entity_type"],
                "entity_display_name": candidate["entity_display_name"],
            },
            "count": len(rows),
            "readings": self._compact_reading_rows(definition, rows),
        }

    def _readings_for_day(
        self, site_id: int, definition: ReadingDefinition, day: date
    ) -> list[dict[str, Any]]:
        start, end = self._day_bounds(day)
        field_sql = _reading_select_fields(definition)
        query = text(
            f"""
            SELECT r.id AS reading_id,
                r.time,
                b.name AS entity_name,
                {field_sql}
            FROM {definition.table} r
            JOIN {definition.base_table} b
                ON r.{definition.foreign_key} = b.{definition.base_key}
            WHERE b.site_id = :site_id
                AND r.time >= :start_time
                AND r.time < :end_time
                {definition.base_filter}
            ORDER BY r.time, b.name
            LIMIT :limit
            """
        )
        rows = self._execute(query, site_id=site_id, start_time=start, end_time=end)
        return self._with_entity_display(
            definition,
            self._with_tank_volumes(_reading_type_for_definition(definition), rows),
        )

    def _readings_for_day_entity(
        self,
        site_id: int,
        definition: ReadingDefinition,
        entity_id: int,
        day: date,
    ) -> list[dict[str, Any]]:
        start, end = self._day_bounds(day)
        field_sql = _reading_select_fields(definition)
        query = text(
            f"""
            SELECT r.id AS reading_id,
                r.time,
                b.name AS entity_name,
                {field_sql}
            FROM {definition.table} r
            JOIN {definition.base_table} b
                ON r.{definition.foreign_key} = b.{definition.base_key}
            WHERE b.site_id = :site_id
                AND b.{definition.base_key} = :entity_id
                AND r.time >= :start_time
                AND r.time < :end_time
                {definition.base_filter}
            ORDER BY r.time, b.name
            LIMIT :limit
            """
        )
        rows = self._execute(
            query,
            site_id=site_id,
            entity_id=entity_id,
            start_time=start,
            end_time=end,
        )
        return self._with_entity_display(
            definition,
            self._with_tank_volumes(_reading_type_for_definition(definition), rows),
        )

    def _matching_entities(
        self, site_id: int, definition: ReadingDefinition, entity_name: str
    ) -> list[dict[str, Any]]:
        has_key = self._table_has_column(definition.base_table, "key")
        key_select = ", b.`key` AS entity_key" if has_key else ""
        key_condition = " OR LOWER(b.`key`) LIKE LOWER(:entity_name)" if has_key else ""
        query = text(
            f"""
            SELECT b.{definition.base_key} AS entity_id,
                b.name AS entity_name
                {key_select}
            FROM {definition.base_table} b
            WHERE b.site_id = :site_id
                AND (
                    LOWER(b.name) LIKE LOWER(:entity_name)
                    {key_condition}
                )
                {definition.base_filter}
            ORDER BY b.name
            LIMIT :limit
            """
        )
        rows = self._execute(
            query,
            site_id=site_id,
            entity_name=f"%{entity_name}%",
        )
        normalized_query = _normalize_entity_match_value(entity_name)
        for row in rows:
            row["match_rank"] = _entity_match_rank(row, normalized_query)
        return rows

    def _entity_has_reading(
        self, site_id: int, definition: ReadingDefinition, entity_id: int, day: date
    ) -> bool:
        start, end = self._day_bounds(day)
        query = text(
            f"""
            SELECT 1
            FROM {definition.table} r
            JOIN {definition.base_table} b
                ON r.{definition.foreign_key} = b.{definition.base_key}
            WHERE b.site_id = :site_id
                AND b.{definition.base_key} = :entity_id
                AND r.time >= :start_time
                AND r.time < :end_time
                {definition.base_filter}
            LIMIT 1
            """
        )
        return bool(
            self._execute(
                query,
                site_id=site_id,
                entity_id=entity_id,
                start_time=start,
                end_time=end,
            )
        )

    @staticmethod
    def _entity_candidate(
        reading_type: str, definition: ReadingDefinition, row: dict[str, Any]
    ) -> dict[str, Any]:
        entity_type = BASE_ENTITY_LABELS.get(definition.base_table, "Entity")
        return {
            "reading_type": reading_type,
            "entity_id": row["entity_id"],
            "entity_type": entity_type,
            "entity_name": row["entity_name"],
            "entity_display_name": f"{entity_type} - {row['entity_name']}",
            "match_rank": row["match_rank"],
        }

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
        for candidate in candidates:
            key = (
                candidate["reading_type"],
                candidate["entity_type"],
                str(candidate["entity_id"]),
            )
            existing = deduplicated.get(key)
            if existing is None or candidate["match_rank"] < existing["match_rank"]:
                deduplicated[key] = candidate
        return list(deduplicated.values())

    def _resolution_reading_types(self, reading_type: str | None) -> list[str]:
        if reading_type is None:
            return list(READING_DEFINITIONS)
        if reading_type == "tank":
            return ["linear_tank", "mixed_tank", "non_linear_tank"]
        self._definition(reading_type)
        return [reading_type]

    def _table_has_column(self, table_name: str, column_name: str) -> bool:
        columns = self._table_columns_cache.get(table_name)
        if columns is None:
            try:
                columns = {
                    str(column["name"])
                    for column in inspect(self.engine).get_columns(table_name)
                }
            except SQLAlchemyError:
                columns = set()
            self._table_columns_cache[table_name] = columns
        return column_name in columns

    def _get_all_tank_readings(self, site_id: int, day: date) -> dict[str, Any]:
        groups = [
            self.get_readings_for_date(site_id, tank_type, day.isoformat())
            for tank_type in ("linear_tank", "mixed_tank", "non_linear_tank")
        ]
        return {
            "reading_type": "tank",
            "label": "All tank readings",
            "site_id": site_id,
            "date": day.isoformat(),
            "count": sum(group["count"] for group in groups),
            "groups": groups,
        }

    def _search_tank_readings_for_type(
        self,
        site_id: int,
        reading_type: str,
        start_day: date,
        end_day: date,
        battery_name: str | None = None,
        tank_name: str | None = None,
        tank_key: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
    ) -> list[dict[str, Any]]:
        definition = self._definition(reading_type)
        start_time = datetime.combine(start_day, time.min)
        end_time = datetime.combine(end_day + timedelta(days=1), time.min)
        conditions = [
            "t.site_id = :site_id",
            "r.time >= :start_time",
            "r.time < :end_time",
        ]
        if definition.base_filter:
            conditions.append(
                definition.base_filter.removeprefix("AND ").replace("b.", "t.")
            )

        params: dict[str, Any] = {
            "site_id": site_id,
            "start_time": start_time,
            "end_time": end_time,
        }
        if battery_name:
            battery_reference = _battery_match_reference(battery_name)
            if battery_reference["is_number"]:
                conditions.append(
                    "("
                    "LOWER(REPLACE(bat.name, ' ', '')) IN (:battery_compact, :battery_number) "
                    "OR LOWER(REPLACE(REPLACE(bat.`key`, '_', ''), ' ', '')) IN (:battery_compact, :battery_number)"
                    ")"
                )
                params["battery_compact"] = battery_reference["compact"]
                params["battery_number"] = battery_reference["number"]
            else:
                conditions.append(
                    "(LOWER(bat.name) LIKE LOWER(:battery_name) OR LOWER(bat.`key`) LIKE LOWER(:battery_name))"
                )
                params["battery_name"] = battery_reference["pattern"]
        if tank_name:
            name_conditions = []
            for index, lookup_name in enumerate(
                _entity_name_lookup_variants(tank_name, "tank")
            ):
                param_name = f"tank_name_{index}"
                name_conditions.append(f"LOWER(t.name) LIKE LOWER(:{param_name})")
                params[param_name] = f"%{lookup_name}%"
            conditions.append(f"({' OR '.join(name_conditions)})")
        if tank_key:
            conditions.append("LOWER(t.`key`) LIKE LOWER(:tank_key)")
            params["tank_key"] = f"%{tank_key}%"
        if monitored is not None:
            conditions.append("COALESCE(t.monitored, 0) = :monitored")
            params["monitored"] = 1 if monitored else 0
        if disable_reading is not None:
            conditions.append("COALESCE(t.disable_reading, 0) = :disable_reading")
            params["disable_reading"] = 1 if disable_reading else 0

        field_sql = _reading_select_fields(definition, table_alias="r", base_alias="t")
        query = text(
            f"""
            SELECT r.time,
                t.id AS tank_id,
                t.name AS tank_name,
                t.`key` AS tank_key,
                t.type AS tank_type,
                t.bbl_foot,
                t.non_linear_volume_mapping_id,
                t.monitored,
                t.disable_reading,
                bat.id AS battery_id,
                bat.name AS battery_name,
                bat.`key` AS battery_key,
                {field_sql}
            FROM {definition.table} r
            JOIN tanks t ON r.tank_id = t.id
            LEFT JOIN batteries bat ON bat.id = t.battery_id
            WHERE {" AND ".join(conditions)}
            ORDER BY r.time, bat.name, t.name
            LIMIT :limit
            """
        )
        rows = self._execute(query, **params)
        if reading_type == "non_linear_tank":
            rows = self._attach_non_linear_mapping_details(rows)
        rows = self._with_tank_volumes(reading_type, rows)
        return [
            _compact_tank_reading_row(reading_type, definition, row)
            for row in rows
        ]

    def _search_equipment_readings_for_type(
        self,
        site_id: int,
        reading_type: str,
        start_day: date,
        end_day: date,
        battery_name: str | None = None,
        entity_name: str | None = None,
        entity_key: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        equipment_filters: list[dict[str, Any]] | None = None,
        reading_filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        definition = self._definition(reading_type)
        start_time = datetime.combine(start_day, time.min)
        end_time = datetime.combine(end_day + timedelta(days=1), time.min)
        conditions = [
            "b.site_id = :site_id",
            "r.time >= :start_time",
            "r.time < :end_time",
        ]
        if definition.base_filter:
            conditions.append(definition.base_filter.removeprefix("AND "))

        params: dict[str, Any] = {
            "site_id": site_id,
            "start_time": start_time,
            "end_time": end_time,
        }
        if battery_name:
            _append_battery_condition(conditions, params, battery_name)
        if entity_name:
            name_conditions = []
            for index, lookup_name in enumerate(
                _entity_name_lookup_variants(entity_name, reading_type)
            ):
                param_name = f"entity_name_{index}"
                name_conditions.append(f"LOWER(b.name) LIKE LOWER(:{param_name})")
                params[param_name] = f"%{lookup_name}%"
            conditions.append(f"({' OR '.join(name_conditions)})")
        if entity_key:
            if not self._table_has_column(definition.base_table, "key"):
                raise ReadingClientError(
                    f"{definition.label} entities do not have a key field."
                )
            conditions.append("LOWER(b.`key`) LIKE LOWER(:entity_key)")
            params["entity_key"] = f"%{entity_key}%"
        if monitored is not None:
            _append_boolean_metadata_condition(
                self,
                definition,
                conditions,
                params,
                "monitored",
                monitored,
            )
        if disable_reading is not None:
            _append_boolean_metadata_condition(
                self,
                definition,
                conditions,
                params,
                "disable_reading",
                disable_reading,
            )

        applied_equipment_filters = self._equipment_filter_conditions(
            definition, equipment_filters or [], conditions, params
        )
        applied_reading_filters = self._filter_conditions(
            definition, reading_filters or [], conditions, params
        )
        metadata_fields = [
            field
            for field in EQUIPMENT_METADATA_FIELDS.get(reading_type, ())
            if self._table_has_column(definition.base_table, field)
        ]
        metadata_sql = "".join(
            f",\n                b.`{field}` AS equipment_{field}"
            for field in metadata_fields
        )
        key_sql = (
            ",\n                b.`key` AS entity_key"
            if self._table_has_column(definition.base_table, "key")
            else ""
        )
        field_sql = _reading_select_fields(definition)
        battery_join_sql = _equipment_battery_join_sql(reading_type)
        query = text(
            f"""
            SELECT r.time,
                b.{definition.base_key} AS entity_id,
                b.name AS entity_name
                {key_sql},
                bat.id AS battery_id,
                bat.name AS battery_name,
                bat.`key` AS battery_key
                {metadata_sql},
                {field_sql}
            FROM {definition.table} r
            JOIN {definition.base_table} b
                ON r.{definition.foreign_key} = b.{definition.base_key}
            {battery_join_sql}
            WHERE {" AND ".join(conditions)}
            ORDER BY r.time, bat.name, b.name
            LIMIT :limit
            """
        )
        rows = self._execute(query, **params)
        if reading_type == "non_linear_tank":
            rows = self._attach_non_linear_mapping_details_for_base_rows(rows)
        rows = self._with_tank_volumes(reading_type, rows)
        return {
            "equipment_filters": applied_equipment_filters,
            "reading_filters": applied_reading_filters,
            "rows": [
                _compact_equipment_reading_row(
                    reading_type,
                    definition,
                    row,
                    metadata_fields,
                )
                for row in rows
            ],
        }

    def _attach_non_linear_mapping_details(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        mapping_ids = sorted(
            {
                int(row["non_linear_volume_mapping_id"])
                for row in rows
                if _is_number(row.get("non_linear_volume_mapping_id"))
            }
        )
        if not mapping_ids:
            return rows
        query = text(
            """
            SELECT tank_volume_mapping_id, feet, inches, volume
            FROM non_linear_tank_volume_mapping_details
            WHERE tank_volume_mapping_id IN :mapping_ids
            ORDER BY tank_volume_mapping_id, feet, inches
            """
        ).bindparams(bindparam("mapping_ids", expanding=True))
        detail_rows = self._execute(query, mapping_ids=mapping_ids)
        details_by_mapping: dict[int, dict[int, list[tuple[float, float]]]] = {}
        for detail in detail_rows:
            mapping_id = int(detail["tank_volume_mapping_id"])
            feet = int(detail["feet"])
            details_by_mapping.setdefault(mapping_id, {}).setdefault(feet, []).append(
                (float(detail["inches"]), float(detail["volume"]))
            )
        enriched = []
        for row in rows:
            row = dict(row)
            mapping_id = row.get("non_linear_volume_mapping_id")
            if _is_number(mapping_id):
                row["_mapping_details"] = details_by_mapping.get(int(mapping_id), {})
            enriched.append(row)
        return enriched

    def _attach_non_linear_mapping_details_for_base_rows(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        remapped = []
        for row in rows:
            row = dict(row)
            row["non_linear_volume_mapping_id"] = row.get(
                "equipment_non_linear_volume_mapping_id"
            )
            remapped.append(row)
        return self._attach_non_linear_mapping_details(remapped)

    def _execute(self, query, **params: Any) -> list[dict[str, Any]]:
        params.setdefault("limit", self.max_rows)
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, params)
                return [self._serialize_row(dict(row._mapping)) for row in result]
        except SQLAlchemyError as exc:
            raise ReadingClientError(f"Database query failed: {exc}") from exc

    @staticmethod
    def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
        serialized = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat(sep=" ")
            elif isinstance(value, date):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _entity_summary(
        first_rows: list[dict[str, Any]], second_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        entity_names = {
            row["entity_name"] for row in first_rows
        } | {row["entity_name"] for row in second_rows}
        summary = []
        for entity_name in sorted(entity_names):
            first = [row for row in first_rows if row["entity_name"] == entity_name]
            second = [row for row in second_rows if row["entity_name"] == entity_name]
            display_name = (first or second)[0]["entity_display_name"]
            summary.append(
                {
                    "entity_display_name": display_name,
                    "first_count": len(first),
                    "second_count": len(second),
                    "status": ReadingClient._comparison_status(len(first), len(second)),
                }
            )
        return summary

    @staticmethod
    def _comparison_status(first_count: int, second_count: int) -> str:
        if first_count and second_count:
            return "present_on_both_dates"
        if first_count:
            return "present_only_on_first_date"
        return "present_only_on_second_date"

    @staticmethod
    def _comparison_rows(
        reading_type: str,
        definition: ReadingDefinition,
        first_rows: list[dict[str, Any]],
        second_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        first_by_entity = ReadingClient._latest_by_entity(first_rows)
        second_by_entity = ReadingClient._latest_by_entity(second_rows)
        entity_names = set(first_by_entity) | set(second_by_entity)
        rows = []

        for entity_name in sorted(entity_names):
            first = first_by_entity.get(entity_name)
            second = second_by_entity.get(entity_name)
            display_name = (first or second)["entity_display_name"]

            if first is None or second is None:
                rows.append(
                    {
                        "entity_display_name": display_name,
                        "status": "missing_on_first_date"
                        if first is None
                        else "missing_on_second_date",
                    }
                )
                continue

            changes = ReadingClient._reading_changes(reading_type, definition, first, second)
            rows.append(
                {
                    "entity_display_name": display_name,
                    "status": "compared",
                    "changes": changes,
                }
            )

        return rows

    @staticmethod
    def _latest_by_entity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        latest = {}
        for row in rows:
            current = latest.get(row["entity_name"])
            if current is None or str(row.get("time", "")) > str(current.get("time", "")):
                latest[row["entity_name"]] = row
        return latest

    @staticmethod
    def _reading_changes(
        reading_type: str,
        definition: ReadingDefinition,
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if reading_type == "mixed_tank":
            changes = [
                ReadingClient._feet_inches_change(
                    "top_level",
                    "Top level",
                    first.get("top_level_feet"),
                    first.get("top_level_inches"),
                    second.get("top_level_feet"),
                    second.get("top_level_inches"),
                ),
                ReadingClient._feet_inches_change(
                    "water_level",
                    "Water level",
                    first.get("water_level_feet"),
                    first.get("water_level_inches"),
                    second.get("water_level_feet"),
                    second.get("water_level_inches"),
                ),
            ]
            return [
                change
                for change in changes
                if change["direction"] != "unchanged"
            ]

        changes = []
        for field in definition.fields:
            if field == "comments":
                continue
            first_value = first.get(field)
            second_value = second.get(field)
            if first_value == second_value:
                direction = "unchanged"
                delta = None
            elif _is_number(first_value) and _is_number(second_value):
                numeric_delta = float(second_value) - float(first_value)
                direction = _direction(numeric_delta)
                delta = round(numeric_delta, 3)
            else:
                direction = "changed"
                delta = None

            if direction != "unchanged":
                changes.append(
                    {
                        "field": field,
                        "first_value": first_value,
                        "second_value": second_value,
                        "direction": direction,
                        "delta": delta,
                    }
                )
        return changes

    @staticmethod
    def _feet_inches_change(
        field: str,
        label: str,
        first_feet: Any,
        first_inches: Any,
        second_feet: Any,
        second_inches: Any,
    ) -> dict[str, Any]:
        first_total = _total_inches(first_feet, first_inches)
        second_total = _total_inches(second_feet, second_inches)
        if first_total is None or second_total is None:
            return {
                "field": field,
                "label": label,
                "first_value": _format_feet_inches(first_feet, first_inches),
                "second_value": _format_feet_inches(second_feet, second_inches),
                "direction": "unknown",
                "delta": None,
            }

        delta_inches = second_total - first_total
        return {
            "field": field,
            "label": label,
            "first_value": _format_feet_inches(first_feet, first_inches),
            "second_value": _format_feet_inches(second_feet, second_inches),
            "direction": _direction(delta_inches),
            "delta_inches": delta_inches,
            "delta": _format_inches_delta(delta_inches),
        }

    @staticmethod
    def _with_entity_display(
        definition: ReadingDefinition, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        entity_type = BASE_ENTITY_LABELS.get(definition.base_table, "Entity")
        enriched = []
        for row in rows:
            row = dict(row)
            row["entity_type"] = entity_type
            row["entity_display_name"] = f"{entity_type} - {row['entity_name']}"
            enriched.append(row)
        return enriched

    @staticmethod
    def _with_tank_volumes(
        reading_type: str,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if reading_type not in {"linear_tank", "mixed_tank", "non_linear_tank"}:
            return rows

        enriched = []
        for row in rows:
            row = dict(row)
            if reading_type == "linear_tank":
                row.update(_linear_tank_volume(row))
            elif reading_type == "mixed_tank":
                row.update(_mixed_tank_volumes(row))
            elif reading_type == "non_linear_tank":
                row.update(_non_linear_tank_volumes(row))
            enriched.append(row)
        return enriched

    @staticmethod
    def _compact_reading_rows(
        definition: ReadingDefinition, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            ReadingClient._compact_reading_row(definition, row)
            for row in rows
        ]

    @staticmethod
    def _compact_reading_row(
        definition: ReadingDefinition, row: dict[str, Any]
    ) -> dict[str, Any]:
        compact = {
            "entity_display_name": row["entity_display_name"],
            "entity_type": row["entity_type"],
        }
        for field in definition.fields:
            value = row.get(field)
            if value is not None:
                compact[field] = value
        for field in TANK_VOLUME_FIELDS:
            value = row.get(field)
            if value is not None:
                compact[field] = value
        return compact

    @staticmethod
    def _compact_range_reading_rows(
        definition: ReadingDefinition, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "date": str(row.get("time", ""))[:10],
                **ReadingClient._compact_reading_row(definition, row),
            }
            for row in rows
        ]

    @staticmethod
    def _filter_conditions(
        definition: ReadingDefinition,
        filters: list[dict[str, Any]],
        conditions: list[str],
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        operators = {
            ">": ">",
            ">=": ">=",
            "<": "<",
            "<=": "<=",
            "=": "=",
            "==": "=",
        }
        applied = []
        numeric_fields = {
            field for field in definition.fields if field not in {"comments", "type"}
        }
        for index, filter_item in enumerate(filters):
            field = str(filter_item.get("field", "")).strip()
            operator = operators.get(str(filter_item.get("operator", "")).strip())
            value = filter_item.get("value")
            if field not in numeric_fields:
                raise ReadingClientError(
                    f"Cannot filter {definition.label} by unsupported field: {field}."
                )
            if operator is None:
                raise ReadingClientError(
                    "Filter operator must be one of >, >=, <, <=, =, ==."
                )
            if not _is_number(value):
                raise ReadingClientError("Filter values must be numbers.")

            param_name = f"filter_{index}"
            conditions.append(f"r.{field} {operator} :{param_name}")
            params[param_name] = value
            applied.append({"field": field, "operator": operator, "value": value})
        return applied

    def _equipment_filter_conditions(
        self,
        definition: ReadingDefinition,
        filters: list[dict[str, Any]],
        conditions: list[str],
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        reading_type = _reading_type_for_definition(definition)
        allowed_fields = set(EQUIPMENT_METADATA_FIELDS.get(reading_type, ()))
        operators = {
            ">": ">",
            ">=": ">=",
            "<": "<",
            "<=": "<=",
            "=": "=",
            "==": "=",
            "!=": "!=",
            "contains": "contains",
        }
        applied = []
        for index, filter_item in enumerate(filters):
            field = str(filter_item.get("field", "")).strip()
            operator = operators.get(str(filter_item.get("operator", "")).strip())
            value = filter_item.get("value")
            if field not in allowed_fields:
                raise ReadingClientError(
                    f"Cannot filter {definition.label} equipment by unsupported field: {field}."
                )
            if not self._table_has_column(definition.base_table, field):
                raise ReadingClientError(
                    f"{definition.label} entities do not have field: {field}."
                )
            if operator is None:
                raise ReadingClientError(
                    "Equipment filter operator must be one of >, >=, <, <=, =, ==, !=, contains."
                )

            param_name = f"equipment_filter_{index}"
            if operator == "contains":
                conditions.append(f"LOWER(b.`{field}`) LIKE LOWER(:{param_name})")
                params[param_name] = f"%{value}%"
                applied.append({"field": field, "operator": operator, "value": value})
                continue
            if operator in {">", ">=", "<", "<="} and not _is_number(value):
                raise ReadingClientError("Range equipment filter values must be numbers.")

            conditions.append(f"b.`{field}` {operator} :{param_name}")
            params[param_name] = _coerce_bool_to_int(value)
            applied.append({"field": field, "operator": operator, "value": value})
        return applied

    def _equipment_list_filter_conditions(
        self,
        definition: EquipmentDefinition,
        filters: list[dict[str, Any]],
        conditions: list[str],
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        allowed_fields = set(definition.metadata_fields)
        operators = {
            ">": ">",
            ">=": ">=",
            "<": "<",
            "<=": "<=",
            "=": "=",
            "==": "=",
            "!=": "!=",
            "contains": "contains",
        }
        applied = []
        for index, filter_item in enumerate(filters):
            field = str(filter_item.get("field", "")).strip()
            operator = operators.get(str(filter_item.get("operator", "")).strip())
            value = filter_item.get("value")
            if field not in allowed_fields:
                raise ReadingClientError(
                    f"Cannot filter {definition.label} by unsupported field: {field}."
                )
            if not self._table_has_column(definition.table, field):
                raise ReadingClientError(f"{definition.label} do not have field: {field}.")
            if operator is None:
                raise ReadingClientError(
                    "Equipment filter operator must be one of >, >=, <, <=, =, ==, !=, contains."
                )
            param_name = f"equipment_list_filter_{index}"
            if operator == "contains":
                conditions.append(f"LOWER(b.`{field}`) LIKE LOWER(:{param_name})")
                params[param_name] = f"%{value}%"
            else:
                if operator in {">", ">=", "<", "<="} and not _is_number(value):
                    raise ReadingClientError("Range equipment filter values must be numbers.")
                conditions.append(f"b.`{field}` {operator} :{param_name}")
                params[param_name] = _coerce_bool_to_int(value)
            applied.append({"field": field, "operator": operator, "value": value})
        return applied

    @staticmethod
    def _parse_date(value: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ReadingClientError("Dates must use YYYY-MM-DD format.") from exc

    @staticmethod
    def _day_bounds(day: date) -> tuple[datetime, datetime]:
        start = datetime.combine(day, time.min)
        return start, start + timedelta(days=1)

    @staticmethod
    def _definition(reading_type: str) -> ReadingDefinition:
        try:
            return READING_DEFINITIONS[reading_type]
        except KeyError as exc:
            raise ReadingClientError(f"Unsupported reading type: {reading_type}") from exc

    def _missing_reading_ignored_ids(
        self, site_id: int, reading_type: str
    ) -> tuple[int, ...]:
        site_cache = self._missing_reading_exclusion_cache.get(site_id)
        if site_cache is None:
            site_cache = self._load_missing_reading_exclusions(site_id)
            self._missing_reading_exclusion_cache[site_id] = site_cache
        return site_cache.get(reading_type, ())

    def _load_missing_reading_exclusions(
        self, site_id: int
    ) -> dict[str, tuple[int, ...]]:
        query = text(
            """
            SELECT related_entities
            FROM report_configurations
            WHERE site_id = :site_id
                AND function_name = :function_name
            ORDER BY id DESC
            LIMIT 1
            """
        )
        try:
            with self.engine.connect() as connection:
                row = (
                    connection.execute(
                        query,
                        {
                            "site_id": site_id,
                            "function_name": MISSING_READINGS_REPORT_FUNCTION_NAME,
                        },
                    )
                    .mappings()
                    .first()
                )
        except SQLAlchemyError:
            return {}

        if row is None:
            return {}

        related_entities = row.get("related_entities")
        if isinstance(related_entities, str):
            try:
                related_entities = json.loads(related_entities)
            except json.JSONDecodeError:
                return {}
        if not isinstance(related_entities, dict):
            return {}

        exclusions: dict[str, tuple[int, ...]] = {}
        if lact_ids := _coerce_id_list(related_entities.get("lact_ids")):
            exclusions["lact"] = lact_ids
        if flare_ids := _coerce_id_list(related_entities.get("flare_ids")):
            exclusions["flare"] = flare_ids
        if knock_out_ids := _coerce_id_list(related_entities.get("knock_out_ids")):
            exclusions["knock_out"] = knock_out_ids
        if tank_ids := related_entities.get("tank_ids"):
            if isinstance(tank_ids, dict):
                linear_ids = _coerce_id_list(tank_ids.get("linear"))
                mixed_ids = _coerce_id_list(tank_ids.get("mixed"))
                non_linear_ids = _coerce_id_list(tank_ids.get("nonLinear"))
                if linear_ids:
                    exclusions["linear_tank"] = linear_ids
                if mixed_ids:
                    exclusions["mixed_tank"] = mixed_ids
                if non_linear_ids:
                    exclusions["non_linear_tank"] = non_linear_ids
        if treater_ids := _coerce_id_list(related_entities.get("treater_ids")):
            exclusions["treater"] = treater_ids
        if water_plant_ids := _coerce_id_list(related_entities.get("water_plant_ids")):
            exclusions["water_plant"] = water_plant_ids
        if pump_ids := _coerce_id_list(related_entities.get("pump_ids")):
            exclusions["pump"] = pump_ids
        if flow_meter_ids := related_entities.get("flow_meter_ids"):
            if isinstance(flow_meter_ids, dict):
                combined = (
                    _coerce_id_list(flow_meter_ids.get("water"))
                    + _coerce_id_list(flow_meter_ids.get("gas"))
                    + _coerce_id_list(flow_meter_ids.get("oil"))
                )
                if combined:
                    exclusions["flow_meter"] = tuple(dict.fromkeys(combined))
        return exclusions


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _normalize_entity_match_value(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _entity_name_lookup_variants(entity_name: str, reading_type: str | None) -> list[str]:
    """Return safe lookup variants for entity names copied from assistant output.

    Reading responses use display names such as "Tank - 10-1 Oil", but the
    underlying equipment table stores only "10-1 Oil". Users often reuse the
    displayed text in a follow-up command, so resolution should try the stored
    form as well as the literal user input.
    """
    variants = [entity_name.strip()]
    stripped = _strip_entity_display_prefix(entity_name, reading_type)
    if stripped and stripped not in variants:
        variants.append(stripped)
    return variants


def _strip_entity_display_prefix(entity_name: str, reading_type: str | None) -> str | None:
    """Remove known equipment display prefixes from a user-supplied name."""
    prefixes = _entity_display_prefixes_for_reading_type(reading_type)
    if not prefixes:
        return None

    prefix_pattern = "|".join(
        re.escape(prefix) for prefix in sorted(prefixes, key=len, reverse=True)
    )
    match = re.match(
        rf"^\s*(?:{prefix_pattern})\s*(?:-\s*)?(.+?)\s*$",
        entity_name,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    stripped = match.group(1).strip()
    return stripped if stripped and stripped != entity_name.strip() else None


def _entity_display_prefixes_for_reading_type(reading_type: str | None) -> set[str]:
    """Map reading type hints to the UI prefixes that may appear in answers."""
    if reading_type is None:
        return {
            "Flare",
            "Flow Meter",
            "Knock Out",
            "LACT",
            "Pump",
            "Tank",
            "Treater",
            "Water Plant",
            "Well",
        }
    if reading_type in {"tank", "linear_tank", "mixed_tank", "non_linear_tank"}:
        return {"Tank"}

    return {
        "flare": {"Flare"},
        "flow_meter": {"Flow Meter"},
        "knock_out": {"Knock Out"},
        "lact": {"LACT"},
        "pump": {"Pump"},
        "treater": {"Treater"},
        "water_plant": {"Water Plant"},
        "well_fluid": {"Well"},
        "well_injection": {"Well"},
        "well_test": {"Well"},
    }.get(reading_type, set())


def _entity_match_rank(row: dict[str, Any], normalized_query: str) -> int:
    """Rank exact entity-name/key matches ahead of broader partial matches."""
    values = [
        _normalize_entity_match_value(row.get("entity_name")),
        _normalize_entity_match_value(row.get("entity_key")),
    ]
    if normalized_query in values:
        return 0
    if any(value.endswith(normalized_query) for value in values if value):
        return 1
    return 2


def _reading_select_fields(
    definition: ReadingDefinition,
    table_alias: str = "r",
    base_alias: str = "b",
) -> str:
    fields = [f"{table_alias}.{field}" for field in definition.fields]
    if definition.base_table == "tanks" and definition.table in {
        "linear_tank_readings",
        "mixed_tank_readings",
        "non_linear_tank_readings",
    }:
        fields.append(f"{base_alias}.bbl_foot")
        fields.append(f"{base_alias}.contents AS equipment_contents")
        fields.append(f"{base_alias}.unusable_height AS equipment_unusable_height")
    return ",\n                ".join(fields)


def _reading_type_for_definition(definition: ReadingDefinition) -> str:
    for reading_type, candidate in READING_DEFINITIONS.items():
        if candidate is definition:
            return reading_type
    return ""


def _linear_tank_volume(row: dict[str, Any]) -> dict[str, Any]:
    bbl_foot = row.get("bbl_foot")
    contents = row.get("equipment_contents") or row.get("contents")
    if not _is_number(bbl_foot):
        return {"volume_status": "missing_bbl_foot", "content_type": contents}

    if _is_number(row.get("level")):
        level = float(row["level"])
    elif _is_number(row.get("feet")) and _is_number(row.get("inches")):
        level = float(row["feet"]) + float(row["inches"]) / 12
    else:
        return {"bbl_foot": float(bbl_foot), "volume_status": "missing_level"}

    volume = round(level * float(bbl_foot), 2)
    result = {
        "bbl_foot": float(bbl_foot),
        "volume": volume,
        "content_type": contents,
    }
    if contents == "oil":
        result["oil_volume"] = volume
        result["recoverable_oil_volume"] = round(
            max(level - float(row.get("equipment_unusable_height") or 0), 0)
            * float(bbl_foot),
            2,
        )
    elif contents == "water":
        result["water_volume"] = volume
        result["recoverable_oil_volume"] = 0.0
    return result


def _non_linear_tank_volumes(row: dict[str, Any]) -> dict[str, Any]:
    initial_volume = _non_linear_volume_from_mapping(
        row, row.get("initial_feet"), row.get("initial_inches")
    )
    final_volume = _non_linear_volume_from_mapping(
        row, row.get("final_feet"), row.get("final_inches")
    )
    contents = row.get("equipment_contents") or row.get("contents")
    volume_prefix = "oil" if contents == "oil" else "water"
    result: dict[str, Any] = {"content_type": contents}
    unusable_volume = _non_linear_unusable_volume(row)
    if initial_volume is not None:
        result[f"initial_{volume_prefix}_volume"] = round(initial_volume, 2)
        result["initial_recoverable_oil_volume"] = (
            round(max(initial_volume - unusable_volume, 0), 2)
            if contents == "oil" and unusable_volume is not None
            else 0.0
        )
    if final_volume is not None:
        result[f"final_{volume_prefix}_volume"] = round(final_volume, 2)
        result["final_recoverable_oil_volume"] = (
            round(max(final_volume - unusable_volume, 0), 2)
            if contents == "oil" and unusable_volume is not None
            else 0.0
        )
        result[f"{volume_prefix}_volume"] = round(final_volume, 2)
        result["volume"] = round(final_volume, 2)
    elif initial_volume is not None:
        result[f"{volume_prefix}_volume"] = round(initial_volume, 2)
        result["volume"] = round(initial_volume, 2)
    if initial_volume is not None and final_volume is not None:
        result["total_volume"] = round(initial_volume - final_volume, 2)
        result["recoverable_oil_difference"] = round(
            result["initial_recoverable_oil_volume"]
            - result["final_recoverable_oil_volume"],
            2,
        )
    if "volume" not in result:
        result["volume_status"] = "missing_non_linear_mapping"
    return result


def _mixed_tank_volumes(row: dict[str, Any]) -> dict[str, Any]:
    bbl_foot = row.get("bbl_foot")
    contents = row.get("equipment_contents") or row.get("contents") or "water-oil"
    if not _is_number(bbl_foot):
        return {"volume_status": "missing_bbl_foot", "content_type": contents}

    required = (
        row.get("top_level_feet"),
        row.get("top_level_inches"),
        row.get("water_level_feet"),
        row.get("water_level_inches"),
    )
    if not all(_is_number(value) for value in required):
        return {"bbl_foot": float(bbl_foot), "volume_status": "missing_levels"}

    top_level = float(row["top_level_feet"]) + float(row["top_level_inches"]) / 12
    water_level = float(row["water_level_feet"]) + float(row["water_level_inches"]) / 12
    total_volume = top_level * float(bbl_foot)
    water_volume = water_level * float(bbl_foot)
    oil_volume = total_volume - water_volume
    unusable_height = float(row.get("equipment_unusable_height") or 0)
    recoverable_oil_height = max(top_level - max(water_level, unusable_height), 0)
    return {
        "bbl_foot": float(bbl_foot),
        "oil_volume": round(oil_volume, 2),
        "water_volume": round(water_volume, 2),
        "total_volume": round(total_volume, 2),
        "recoverable_oil_volume": round(recoverable_oil_height * float(bbl_foot), 2),
        "content_type": contents,
    }


def _normalize_tank_type(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "linear": "linear_tank",
        "linear_tank": "linear_tank",
        "linear_volume": "linear_tank",
        "mixed": "mixed_tank",
        "mixed_tank": "mixed_tank",
        "mixed_water_oil": "mixed_tank",
        "oil_water": "mixed_tank",
        "non_linear": "non_linear_tank",
        "non_linear_tank": "non_linear_tank",
        "non_linear_volume": "non_linear_tank",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ReadingClientError(f"Unsupported tank_type: {value}") from exc


def _normalize_equipment_reading_type(value: str) -> str:
    normalized = str(value).strip()
    tank_aliases = {
        "linear",
        "linear_tank",
        "linear-volume",
        "linear_volume",
        "mixed",
        "mixed_tank",
        "mixed-water-oil",
        "mixed_water_oil",
        "oil_water",
        "non_linear",
        "non-linear",
        "non_linear_tank",
        "non-linear_tank",
        "non-linear-volume",
        "non_linear_volume",
    }
    normalized_key = normalized.lower().replace("-", "_").replace(" ", "_")
    if normalized_key in {item.replace("-", "_") for item in tank_aliases}:
        resolved = _normalize_tank_type(normalized)
        if resolved is not None:
            return resolved
    return normalized


def _equipment_definition(equipment_type: str) -> EquipmentDefinition:
    normalized = str(equipment_type).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lacts": "lact",
        "flares": "flare",
        "tanks": "tank",
        "water_plants": "water_plant",
        "waterplant": "water_plant",
        "flow_meters": "flow_meter",
        "flowmeter": "flow_meter",
        "treaters": "treater",
        "knockouts": "knock_out",
        "knock_outs": "knock_out",
        "knockout": "knock_out",
        "pumps": "pump",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return EQUIPMENT_DEFINITIONS[normalized]
    except KeyError as exc:
        raise ReadingClientError(f"Unsupported equipment type: {equipment_type}") from exc


def _normalize_contains(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().lower()
    aliases = {
        "oil": "oil",
        "water": "water",
        "any": "any",
        "volume": "any",
        "fluid": "any",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ReadingClientError("contains must be oil, water, or any.") from exc


def _matches_contains(row: dict[str, Any], contains: str | None) -> bool:
    if contains is None:
        return True
    if contains == "oil":
        return row.get("content_type") in {"oil", "water-oil"} and _positive(
            row.get("oil_volume") or row.get("volume")
        )
    if contains == "water":
        return _positive(row.get("water_volume")) or (
            row.get("content_type") in {"water", "water-oil"} and _positive(row.get("volume"))
        )
    return any(
        _positive(row.get(field))
        for field in ("oil_volume", "water_volume", "total_volume", "volume")
    )


def _matches_computed_filters(
    row: dict[str, Any], filters: list[dict[str, Any]]
) -> bool:
    for filter_item in filters:
        field = str(filter_item.get("field", "")).strip()
        operator = str(filter_item.get("operator", "")).strip()
        value = filter_item.get("value")
        if field not in {
            "oil_volume",
            "water_volume",
            "total_volume",
            "volume",
            "bbl_foot",
            "initial_water_volume",
            "final_water_volume",
            "recoverable_oil_volume",
            "initial_recoverable_oil_volume",
            "final_recoverable_oil_volume",
            "recoverable_oil_difference",
        }:
            raise ReadingClientError(f"Unsupported tank computed filter field: {field}.")
        if operator not in {">", ">=", "<", "<=", "=", "=="}:
            raise ReadingClientError("Filter operator must be one of >, >=, <, <=, =, ==.")
        if not _is_number(value):
            raise ReadingClientError("Filter values must be numbers.")
        current = row.get(field)
        if not _is_number(current) or not _compare(float(current), operator, float(value)):
            return False
    return True


def _compare(left: float, operator: str, right: float) -> bool:
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    return left == right


def _positive(value: Any) -> bool:
    return _is_number(value) and float(value) > 0


def _battery_match_reference(value: str) -> dict[str, Any]:
    normalized = str(value).strip()
    digits = "".join(char for char in normalized if char.isdigit())
    if digits and normalized.lower().replace(" ", "") in {digits, f"battery{digits}"}:
        return {
            "is_number": True,
            "compact": f"battery{digits}",
            "number": digits,
        }
    return {
        "is_number": False,
        "pattern": f"%{normalized}%",
    }


def _append_battery_condition(
    conditions: list[str], params: dict[str, Any], battery_name: str
) -> None:
    battery_reference = _battery_match_reference(battery_name)
    if battery_reference["is_number"]:
        conditions.append(
            "("
            "LOWER(REPLACE(bat.name, ' ', '')) IN (:battery_compact, :battery_number) "
            "OR LOWER(REPLACE(REPLACE(bat.`key`, '_', ''), ' ', '')) IN (:battery_compact, :battery_number)"
            ")"
        )
        params["battery_compact"] = battery_reference["compact"]
        params["battery_number"] = battery_reference["number"]
        return

    conditions.append(
        "(LOWER(bat.name) LIKE LOWER(:battery_name) OR LOWER(bat.`key`) LIKE LOWER(:battery_name))"
    )
    params["battery_name"] = battery_reference["pattern"]


def _append_boolean_metadata_condition(
    client: ReadingClient,
    definition: ReadingDefinition,
    conditions: list[str],
    params: dict[str, Any],
    field: str,
    value: bool,
) -> None:
    if not client._table_has_column(definition.base_table, field):
        return
    conditions.append(f"COALESCE(b.`{field}`, 0) = :{field}")
    params[field] = 1 if value else 0


def _equipment_battery_join_sql(reading_type: str) -> str:
    if reading_type == "pump":
        return """
            LEFT JOIN water_plants wp ON wp.id = b.water_plant_id
            LEFT JOIN batteries bat ON bat.id = wp.battery_id
            """
    return "LEFT JOIN batteries bat ON bat.id = b.battery_id"


def _equipment_list_battery_join_sql(equipment_type: str) -> str:
    normalized = str(equipment_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"pump", "pumps"}:
        return """
            LEFT JOIN water_plants wp ON wp.id = b.water_plant_id
            LEFT JOIN batteries bat ON bat.id = wp.battery_id
            """
    return "LEFT JOIN batteries bat ON bat.id = b.battery_id"


def _coerce_bool_to_int(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    return value


def _compact_tank_reading_row(
    reading_type: str,
    definition: ReadingDefinition,
    row: dict[str, Any],
) -> dict[str, Any]:
    compact = {
        "date": str(row.get("time", ""))[:10],
        "reading_type": reading_type,
        "tank_id": row.get("tank_id"),
        "tank_name": row.get("tank_name"),
        "tank_key": row.get("tank_key"),
        "tank_type": row.get("tank_type"),
        "contents": row.get("equipment_contents"),
        "entity_display_name": f"Tank - {row.get('tank_name')}",
        "battery_id": row.get("battery_id"),
        "battery_name": row.get("battery_name"),
        "battery_key": row.get("battery_key"),
        "monitored": row.get("monitored"),
        "disable_reading": row.get("disable_reading"),
    }
    for field in definition.fields:
        value = row.get(field)
        if value is not None:
            compact[field] = value
    for field in TANK_VOLUME_FIELDS:
        value = row.get(field)
        if value is not None:
            compact[field] = value
    return compact


def _compact_equipment_reading_row(
    reading_type: str,
    definition: ReadingDefinition,
    row: dict[str, Any],
    metadata_fields: list[str],
) -> dict[str, Any]:
    entity_type = BASE_ENTITY_LABELS.get(definition.base_table, "Entity")
    compact = {
        "date": str(row.get("time", ""))[:10],
        "reading_type": reading_type,
        "entity_id": row.get("entity_id"),
        "entity_name": row.get("entity_name"),
        "entity_key": row.get("entity_key"),
        "entity_type": entity_type,
        "entity_display_name": f"{entity_type} - {row.get('entity_name')}",
        "battery_id": row.get("battery_id"),
        "battery_name": row.get("battery_name"),
        "battery_key": row.get("battery_key"),
    }
    if definition.base_table == "tanks":
        compact.update(
            {
                "tank_id": row.get("entity_id"),
                "tank_name": row.get("entity_name"),
                "tank_key": row.get("entity_key"),
                "tank_type": row.get("equipment_type"),
            }
        )
    for field in metadata_fields:
        value = row.get(f"equipment_{field}")
        if value is not None:
            compact[field] = value
    for field in definition.fields:
        value = row.get(field)
        if value is not None:
            compact[field] = value
    for field in TANK_VOLUME_FIELDS:
        value = row.get(field)
        if value is not None:
            compact[field] = value
    return compact


def _compact_equipment_entity_row(
    equipment_type: str,
    definition: EquipmentDefinition,
    row: dict[str, Any],
    metadata_fields: list[str],
) -> dict[str, Any]:
    entity_type = BASE_ENTITY_LABELS.get(definition.table, "Entity")
    compact = {
        "equipment_type": equipment_type,
        "entity_id": row.get("entity_id"),
        "entity_name": row.get("entity_name"),
        "entity_key": row.get("entity_key"),
        "entity_type": entity_type,
        "entity_display_name": f"{entity_type} - {row.get('entity_name')}",
        "battery_id": row.get("battery_id"),
        "battery_name": row.get("battery_name"),
        "battery_key": row.get("battery_key"),
    }
    if definition.table == "tanks":
        compact.update(
            {
                "tank_id": row.get("entity_id"),
                "tank_name": row.get("entity_name"),
                "tank_key": row.get("entity_key"),
                "tank_type": row.get("equipment_type"),
            }
        )
    for field in metadata_fields:
        value = row.get(f"equipment_{field}")
        if value is not None:
            compact[field] = value
    return compact


def _non_linear_volume_from_mapping(
    row: dict[str, Any], feet: Any, inches: Any
) -> float | None:
    details = row.get("_mapping_details")
    if not isinstance(details, dict):
        return None
    if not _is_number(feet) or not _is_number(inches):
        return None
    candidates = details.get(int(feet))
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(float(item[0]) - float(inches)))
    return float(nearest[1])


def _non_linear_unusable_volume(row: dict[str, Any]) -> float | None:
    height = float(row.get("equipment_unusable_height") or 0)
    if height <= 0:
        return 0.0
    feet = int(height)
    inches = (height - feet) * 12
    return _non_linear_volume_from_mapping(row, feet, inches)


def _direction(delta: float) -> str:
    if delta > 0:
        return "increase"
    if delta < 0:
        return "decrease"
    return "unchanged"


def _total_inches(feet: Any, inches: Any) -> float | None:
    if not _is_number(feet) or not _is_number(inches):
        return None
    return float(feet) * 12 + float(inches)


def _format_feet_inches(feet: Any, inches: Any) -> str | None:
    if not _is_number(feet) or not _is_number(inches):
        return None
    inches_value = float(inches)
    if inches_value.is_integer():
        inches_text = str(int(inches_value))
    else:
        inches_text = str(round(inches_value, 2)).rstrip("0").rstrip(".")
    return f"{int(feet)}'{inches_text}\""


def _format_inches_delta(delta_inches: float) -> str:
    signless = abs(delta_inches)
    feet = int(signless // 12)
    inches = signless - feet * 12
    if inches.is_integer():
        inches_text = str(int(inches))
    else:
        inches_text = str(round(inches, 2)).rstrip("0").rstrip(".")

    if feet:
        return f"{feet}'{inches_text}\""
    return f"{inches_text}\""


def _coerce_id_list(value: Any) -> tuple[int, ...]:
    if not value:
        return ()
    if isinstance(value, (str, bytes)):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    ids = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(dict.fromkeys(ids))


class UnavailableReadingClient:
    def __init__(self, reason: str):
        self.reason = reason

    def list_reading_types(self) -> list[dict[str, Any]]:
        return list_supported_reading_types()

    def get_readings_for_date(
        self, site_id: int, reading_type: str, reading_date: str
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def compare_readings_between_dates(
        self,
        site_id: int,
        reading_type: str,
        first_date: str,
        second_date: str,
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def missing_readings_for_date(
        self, site_id: int, reading_type: str, reading_date: str
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def all_missing_readings_for_date(
        self, site_id: int, reading_date: str
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def all_missing_readings_for_range(
        self, site_id: int, start_date: str, end_date: str
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def search_well_tests(
        self,
        site_id: int,
        start_date: str,
        end_date: str,
        well_name: str | None = None,
        min_oil: float | None = None,
        max_oil: float | None = None,
        min_water: float | None = None,
        max_water: float | None = None,
        min_gas: float | None = None,
        max_gas: float | None = None,
        min_runtime: float | None = None,
        max_runtime: float | None = None,
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def analyze_well_tests(
        self,
        site_id: int,
        analysis_mode: str,
        group_by: str,
        start_date: str | None = None,
        end_date: str | None = None,
        well_name: str | None = None,
        battery_name: str | None = None,
        test_count: int | None = None,
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def search_readings(
        self,
        site_id: int,
        reading_type: str,
        start_date: str,
        end_date: str,
        entity_name: str | None = None,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def search_equipment_readings(
        self,
        site_id: int,
        reading_type: str,
        start_date: str,
        end_date: str,
        battery_name: str | None = None,
        entity_name: str | None = None,
        entity_key: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        equipment_filters: list[dict[str, Any]] | None = None,
        reading_filters: list[dict[str, Any]] | None = None,
        computed_filters: list[dict[str, Any]] | None = None,
        contains: str | None = None,
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def list_equipment(
        self,
        site_id: int,
        equipment_type: str,
        battery_name: str | None = None,
        entity_name: str | None = None,
        entity_key: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        equipment_filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def resolve_reading_entity(
        self,
        site_id: int,
        entity_name: str,
        reading_date: str | None = None,
        reading_type: str | None = None,
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)

    def get_reading_for_entity(
        self,
        site_id: int,
        entity_name: str,
        reading_date: str,
        reading_type: str | None = None,
    ) -> dict[str, Any]:
        raise ReadingClientError(self.reason)
