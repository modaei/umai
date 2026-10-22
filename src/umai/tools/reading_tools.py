from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.clients.reading_client import (
    READING_DEFINITIONS,
    ReadingClient,
    ReadingClientError,
)
from umai.tools.rag_enrichment import (
    OperationalContextSearchStore,
    add_operational_context,
)


logger = logging.getLogger(__name__)

ReadingType = Literal[
    "tank",
    "lact",
    "flare",
    "linear_tank",
    "mixed_tank",
    "non_linear_tank",
    "water_plant",
    "flow_meter",
    "well_test",
    "well_fluid",
    "well_injection",
    "run_ticket",
    "water_draw",
    "treater",
    "knock_out",
    "pump",
]


class GetReadingsInput(BaseModel):
    reading_type: ReadingType = Field(description="The type of reading to retrieve.")
    reading_date: str = Field(description="Date in YYYY-MM-DD format.")


class GetReadingForEntityInput(BaseModel):
    entity_name: str = Field(
        description="Object name or partial object name, such as Battery 2 Vent.",
    )
    reading_date: str = Field(description="Date in YYYY-MM-DD format.")
    reading_type: ReadingType | None = Field(
        default=None,
        description=(
            "Optional reading type when the user explicitly named it. Leave empty "
            "when the user gave only the object name."
        ),
    )


class CompareReadingsInput(BaseModel):
    reading_type: ReadingType = Field(description="The type of reading to compare.")
    first_date: str = Field(description="First date in YYYY-MM-DD format.")
    second_date: str = Field(description="Second date in YYYY-MM-DD format.")


class MissingReadingsInput(BaseModel):
    reading_type: ReadingType = Field(description="The type of reading to check.")
    reading_date: str = Field(description="Date in YYYY-MM-DD format.")


class AllMissingReadingsInput(BaseModel):
    reading_date: str = Field(description="Date in YYYY-MM-DD format.")


class AllMissingReadingsRangeInput(BaseModel):
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")


class SearchWellTestsInput(BaseModel):
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    well_name: str | None = Field(
        default=None,
        description="Optional well name or partial well name to filter by.",
    )
    min_oil: float | None = Field(
        default=None,
        description="Optional minimum oil value, inclusive.",
    )
    max_oil: float | None = Field(
        default=None,
        description="Optional maximum oil value, inclusive.",
    )
    min_water: float | None = Field(
        default=None,
        description="Optional minimum water value, inclusive.",
    )
    max_water: float | None = Field(
        default=None,
        description="Optional maximum water value, inclusive.",
    )
    min_gas: float | None = Field(
        default=None,
        description="Optional minimum gas value, inclusive.",
    )
    max_gas: float | None = Field(
        default=None,
        description="Optional maximum gas value, inclusive.",
    )
    min_runtime: float | None = Field(
        default=None,
        description="Optional minimum runtime value, inclusive.",
    )
    max_runtime: float | None = Field(
        default=None,
        description="Optional maximum runtime value, inclusive.",
    )


class AnalyzeWellTestsInput(BaseModel):
    analysis_mode: Literal[
        "latest_previous", "range_summary", "recent_tests", "range_sequence"
    ] = Field(
        description=(
            "latest_previous compares each well's latest test with its immediately "
            "previous test; range_summary aggregates tests in a date range; "
            "recent_tests compares the latest test_count tests within each well; "
            "range_sequence compares every test in a date range with that well's "
            "immediately preceding test in the same range."
        )
    )
    group_by: Literal["well", "battery", "site"] = Field(
        description="Return analysis per well, per battery, or for the selected site."
    )
    start_date: str | None = Field(
        default=None,
        description=(
            "Optional YYYY-MM-DD lower bound for latest_previous; required for "
            "range_summary."
        ),
    )
    end_date: str | None = Field(
        default=None,
        description=(
            "Optional YYYY-MM-DD upper bound for latest_previous; required for "
            "range_summary."
        ),
    )
    well_name: str | None = Field(
        default=None, description="Optional case-insensitive partial well name."
    )
    battery_name: str | None = Field(
        default=None, description="Optional case-insensitive partial battery name."
    )
    test_count: int | None = Field(
        default=None,
        ge=2,
        le=10,
        description="Number of latest tests per well; required for recent_tests only.",
    )


class ReadingNumericFilter(BaseModel):
    field: str = Field(
        description="Reading field to filter, such as oil, water, gas, reading, pressure, level, or total.",
    )
    operator: Literal[">", ">=", "<", "<=", "=", "=="] = Field(
        description="Numeric comparison operator.",
    )
    value: float = Field(description="Numeric comparison value.")


class TankComputedFilter(BaseModel):
    field: str = Field(
        description=(
            "Computed tank field to filter: oil_volume, water_volume, total_volume, "
            "volume, recoverable_oil_volume, initial_recoverable_oil_volume, "
            "final_recoverable_oil_volume, recoverable_oil_difference, bbl_foot, "
            "initial_water_volume, or final_water_volume."
        ),
    )
    operator: Literal[">", ">=", "<", "<=", "=", "=="] = Field(
        description="Numeric comparison operator.",
    )
    value: float = Field(description="Numeric comparison value.")


class EquipmentFilter(BaseModel):
    field: str = Field(
        description=(
            "Equipment metadata field to filter, such as type, measurement_method, "
            "monitored, disable_reading, bbl_foot, or water_plant_id."
        ),
    )
    operator: Literal[">", ">=", "<", "<=", "=", "==", "!=", "contains"] = Field(
        description="Comparison operator for the equipment metadata field.",
    )
    value: str | float | bool = Field(description="Comparison value.")


class ListEquipmentInput(BaseModel):
    equipment_type: Literal[
        "lact",
        "flare",
        "tank",
        "water_plant",
        "flow_meter",
        "treater",
        "knock_out",
        "pump",
    ] = Field(description="Equipment type to list.")
    battery_name: str | None = Field(
        default=None,
        description="Optional battery name or number, such as Battery 6 or 6.",
    )
    entity_name: str | None = Field(
        default=None,
        description="Optional equipment name or partial name.",
    )
    entity_key: str | None = Field(
        default=None,
        description="Optional equipment key or partial key.",
    )
    monitored: bool | None = Field(default=None)
    disable_reading: bool | None = Field(default=None)
    equipment_filters: list[EquipmentFilter] = Field(
        default_factory=list,
        description=(
            "Optional filters on whitelisted equipment metadata fields. Examples: "
            "type = gas for flow meters, type = water for pumps, type = mixed-water-oil for tanks."
        ),
    )


class SearchReadingsInput(BaseModel):
    reading_type: ReadingType = Field(description="The type of reading to search.")
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    entity_name: str | None = Field(
        default=None,
        description="Optional entity name or partial name to filter by.",
    )
    filters: list[ReadingNumericFilter] = Field(
        default_factory=list,
        description=(
            "Optional numeric filters on fields supported by the selected reading type. "
            "Examples: oil > 20 for well_test, reading > 100 for lact, pressure > 50 for flare."
        ),
    )


class SearchTankReadingsInput(BaseModel):
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    battery_name: str | None = Field(
        default=None,
        description="Optional battery name or number, such as Battery 6 or 6.",
    )
    tank_name: str | None = Field(
        default=None,
        description="Optional tank name or partial tank name.",
    )
    tank_key: str | None = Field(
        default=None,
        description="Optional tank key or partial tank key.",
    )
    tank_type: str | None = Field(
        default=None,
        description="Optional tank type: linear_tank, mixed_tank, or non_linear_tank.",
    )
    contains: Literal["oil", "water", "any"] | None = Field(
        default=None,
        description=(
            "Optional content filter based on persisted tanks.contents and positive "
            "computed volume. Oil includes only oil or water-oil tanks with positive "
            "gross oil_volume. Water includes water or water-oil tanks with positive "
            "water volume. For recoverable oil qualification, also filter "
            "recoverable_oil_volume > 0."
        ),
    )
    monitored: bool | None = Field(default=None)
    disable_reading: bool | None = Field(default=None)
    filters: list[TankComputedFilter] = Field(
        default_factory=list,
        description="Optional filters on computed tank volume fields.",
    )


class SearchEquipmentReadingsInput(BaseModel):
    reading_type: ReadingType = Field(
        description=(
            "Equipment reading type. Use tank for all tank types, or a specific "
            "type such as flow_meter, lact, flare, knock_out, treater, "
            "water_plant, pump, linear_tank, mixed_tank, or non_linear_tank."
        ),
    )
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    battery_name: str | None = Field(
        default=None,
        description="Optional battery name or number, such as Battery 6 or 6.",
    )
    entity_name: str | None = Field(
        default=None,
        description="Optional equipment name or partial name.",
    )
    entity_key: str | None = Field(
        default=None,
        description="Optional equipment key or partial key.",
    )
    monitored: bool | None = Field(default=None)
    disable_reading: bool | None = Field(default=None)
    equipment_filters: list[EquipmentFilter] = Field(
        default_factory=list,
        description=(
            "Optional filters on whitelisted equipment metadata fields. Examples: "
            "type = gas for flow meters, type = water for pumps, "
            "measurement_method = total for flow meters."
        ),
    )
    reading_filters: list[ReadingNumericFilter] = Field(
        default_factory=list,
        description=(
            "Optional numeric filters on raw reading fields. Examples: total > 100, "
            "pressure > 50, inlet > 0, suction_pressure >= 20."
        ),
    )
    computed_filters: list[TankComputedFilter] = Field(
        default_factory=list,
        description="Optional filters on computed tank volume fields.",
    )
    contains: Literal["oil", "water", "any"] | None = Field(
        default=None,
        description="Optional tank content filter for tank reading types.",
    )


def _json_result(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def build_reading_tools(
    client: ReadingClient,
    site_id: int,
    operational_context_store: OperationalContextSearchStore | None = None,
) -> list[StructuredTool]:
    def list_reading_types() -> str:
        """List reading types that can be queried by the assistant."""
        return _json_result(
            {
                "site_id": site_id,
                "reading_types": client.list_reading_types(),
            }
        )

    def get_readings_for_date(reading_type: str, reading_date: str) -> str:
        """Retrieve readings of one type for the selected site and one date."""
        logger.info(
            "Getting readings site_id=%s type=%s date=%s",
            site_id,
            reading_type,
            reading_date,
        )
        try:
            result = {
                "ok": True,
                **client.get_readings_for_date(site_id, reading_type, reading_date),
            }
            enriched = add_operational_context(
                result,
                operational_context_store,
                site_id=site_id,
                query=f"{reading_type} reading context",
                start_date=reading_date,
                end_date=reading_date,
                limit=8,
            )
            return _json_result(enriched)
        except ReadingClientError as exc:
            logger.warning("Reading lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def get_reading_for_entity(
        entity_name: str, reading_date: str, reading_type: str | None = None
    ) -> str:
        """Resolve a reading entity by name, then retrieve its reading for one date."""
        logger.info(
            "Getting reading for entity site_id=%s entity=%s type=%s date=%s",
            site_id,
            entity_name,
            reading_type,
            reading_date,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.get_reading_for_entity(
                        site_id, entity_name, reading_date, reading_type
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("Entity reading lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def compare_readings_between_dates(
        reading_type: str, first_date: str, second_date: str
    ) -> str:
        """Retrieve one reading type for two dates and summarize presence changes."""
        logger.info(
            "Comparing readings site_id=%s type=%s first=%s second=%s",
            site_id,
            reading_type,
            first_date,
            second_date,
        )
        try:
            result = {
                "ok": True,
                **client.compare_readings_between_dates(
                    site_id, reading_type, first_date, second_date
                ),
            }
            start_date, end_date = sorted((first_date, second_date))
            enriched = add_operational_context(
                result,
                operational_context_store,
                site_id=site_id,
                query=f"{reading_type} reading comparison context",
                start_date=start_date,
                end_date=end_date,
                limit=8,
            )
            return _json_result(enriched)
        except ReadingClientError as exc:
            logger.warning("Reading comparison failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def find_missing_readings(reading_type: str, reading_date: str) -> str:
        """Find configured entities missing a daily reading of the selected type."""
        logger.info(
            "Finding missing readings site_id=%s type=%s date=%s",
            site_id,
            reading_type,
            reading_date,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.missing_readings_for_date(
                        site_id, reading_type, reading_date
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("Missing reading lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def find_all_missing_readings(reading_date: str) -> str:
        """Find all configured entities missing supported daily readings on one date."""
        logger.info(
            "Finding all missing readings site_id=%s date=%s",
            site_id,
            reading_date,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.all_missing_readings_for_date(site_id, reading_date),
                }
            )
        except ReadingClientError as exc:
            logger.warning("All missing readings lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def find_all_missing_readings_for_range(start_date: str, end_date: str) -> str:
        """Find all supported missing readings for every date in a bounded range."""
        logger.info(
            "Finding all missing readings site_id=%s start=%s end=%s",
            site_id,
            start_date,
            end_date,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.all_missing_readings_for_range(
                        site_id, start_date, end_date
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("All missing readings range lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def search_well_tests(
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
    ) -> str:
        """Search well tests over a date range with optional numeric filters."""
        logger.info(
            "Searching well tests site_id=%s start=%s end=%s well=%s",
            site_id,
            start_date,
            end_date,
            well_name,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.search_well_tests(
                        site_id=site_id,
                        start_date=start_date,
                        end_date=end_date,
                        well_name=well_name,
                        min_oil=min_oil,
                        max_oil=max_oil,
                        min_water=min_water,
                        max_water=max_water,
                        min_gas=min_gas,
                        max_gas=max_gas,
                        min_runtime=min_runtime,
                        max_runtime=max_runtime,
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("Well test search failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def analyze_well_tests(
        analysis_mode: str,
        group_by: str,
        start_date: str | None = None,
        end_date: str | None = None,
        well_name: str | None = None,
        battery_name: str | None = None,
        test_count: int | None = None,
    ) -> str:
        """Compare or summarize well tests by well, battery, or selected site."""
        logger.info(
            "Analyzing well tests site_id=%s mode=%s group_by=%s start=%s end=%s",
            site_id,
            analysis_mode,
            group_by,
            start_date,
            end_date,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.analyze_well_tests(
                        site_id=site_id,
                        analysis_mode=analysis_mode,
                        group_by=group_by,
                        start_date=start_date,
                        end_date=end_date,
                        well_name=well_name,
                        battery_name=battery_name,
                        test_count=test_count,
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("Well-test analysis failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def search_readings(
        reading_type: str,
        start_date: str,
        end_date: str,
        entity_name: str | None = None,
        filters: list[ReadingNumericFilter] | None = None,
    ) -> str:
        """Search readings over a date range with optional field filters."""
        filter_values = [
            filter_item.model_dump() if hasattr(filter_item, "model_dump") else filter_item
            for filter_item in (filters or [])
        ]
        logger.info(
            "Searching readings site_id=%s type=%s start=%s end=%s entity=%s filters=%s",
            site_id,
            reading_type,
            start_date,
            end_date,
            entity_name,
            filter_values,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.search_readings(
                        site_id=site_id,
                        reading_type=reading_type,
                        start_date=start_date,
                        end_date=end_date,
                        entity_name=entity_name,
                        filters=filter_values,
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("Reading search failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def list_equipment(
        equipment_type: str,
        battery_name: str | None = None,
        entity_name: str | None = None,
        entity_key: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        equipment_filters: list[EquipmentFilter] | None = None,
    ) -> str:
        """List configured equipment without requiring a reading date."""
        equipment_filter_values = [
            filter_item.model_dump() if hasattr(filter_item, "model_dump") else filter_item
            for filter_item in (equipment_filters or [])
        ]
        logger.info(
            "Listing equipment site_id=%s type=%s battery=%s entity=%s filters=%s",
            site_id,
            equipment_type,
            battery_name,
            entity_name or entity_key,
            equipment_filter_values,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.list_equipment(
                        site_id=site_id,
                        equipment_type=equipment_type,
                        battery_name=battery_name,
                        entity_name=entity_name,
                        entity_key=entity_key,
                        monitored=monitored,
                        disable_reading=disable_reading,
                        equipment_filters=equipment_filter_values,
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("Equipment listing failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def search_tank_readings(
        start_date: str,
        end_date: str,
        battery_name: str | None = None,
        tank_name: str | None = None,
        tank_key: str | None = None,
        tank_type: str | None = None,
        contains: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        filters: list[TankComputedFilter] | None = None,
    ) -> str:
        """Search tank readings by tank/battery metadata and computed volumes."""
        filter_values = [
            filter_item.model_dump() if hasattr(filter_item, "model_dump") else filter_item
            for filter_item in (filters or [])
        ]
        logger.info(
            "Searching tank readings site_id=%s start=%s end=%s battery=%s tank=%s contains=%s filters=%s",
            site_id,
            start_date,
            end_date,
            battery_name,
            tank_name or tank_key,
            contains,
            filter_values,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.search_tank_readings(
                        site_id=site_id,
                        start_date=start_date,
                        end_date=end_date,
                        battery_name=battery_name,
                        tank_name=tank_name,
                        tank_key=tank_key,
                        tank_type=tank_type,
                        contains=contains,
                        monitored=monitored,
                        disable_reading=disable_reading,
                        filters=filter_values,
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("Tank reading search failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def search_equipment_readings(
        reading_type: str,
        start_date: str,
        end_date: str,
        battery_name: str | None = None,
        entity_name: str | None = None,
        entity_key: str | None = None,
        monitored: bool | None = None,
        disable_reading: bool | None = None,
        equipment_filters: list[EquipmentFilter] | None = None,
        reading_filters: list[ReadingNumericFilter] | None = None,
        computed_filters: list[TankComputedFilter] | None = None,
        contains: str | None = None,
    ) -> str:
        """Search equipment readings by battery relation, metadata, and conditions."""
        equipment_filter_values = [
            filter_item.model_dump() if hasattr(filter_item, "model_dump") else filter_item
            for filter_item in (equipment_filters or [])
        ]
        reading_filter_values = [
            filter_item.model_dump() if hasattr(filter_item, "model_dump") else filter_item
            for filter_item in (reading_filters or [])
        ]
        computed_filter_values = [
            filter_item.model_dump() if hasattr(filter_item, "model_dump") else filter_item
            for filter_item in (computed_filters or [])
        ]
        logger.info(
            "Searching equipment readings site_id=%s type=%s start=%s end=%s battery=%s entity=%s equipment_filters=%s reading_filters=%s",
            site_id,
            reading_type,
            start_date,
            end_date,
            battery_name,
            entity_name or entity_key,
            equipment_filter_values,
            reading_filter_values,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.search_equipment_readings(
                        site_id=site_id,
                        reading_type=reading_type,
                        start_date=start_date,
                        end_date=end_date,
                        battery_name=battery_name,
                        entity_name=entity_name,
                        entity_key=entity_key,
                        monitored=monitored,
                        disable_reading=disable_reading,
                        equipment_filters=equipment_filter_values,
                        reading_filters=reading_filter_values,
                        computed_filters=computed_filter_values,
                        contains=contains,
                    ),
                }
            )
        except ReadingClientError as exc:
            logger.warning("Equipment reading search failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    return [
        StructuredTool.from_function(
            func=list_reading_types,
            name="list_reading_types",
            description=(
                "List raw daily reading types available from the Ometrics database."
            ),
        ),
        StructuredTool.from_function(
            func=get_readings_for_date,
            name="get_readings_for_date",
            description=(
                "Get raw readings such as LACT, tank, water plant, well test, "
                "or flow meter readings for one date."
            ),
            args_schema=GetReadingsInput,
        ),
        StructuredTool.from_function(
            func=get_reading_for_entity,
            name="get_reading_for_entity",
            description=(
                "Resolve an object name to the correct reading entity and get its "
                "reading for one date. Use this when the user asks whether an "
                "object has a reading but does not clearly name the reading type, "
                "or when names may overlap across flow meters, flares, tanks, "
                "LACTs, pumps, treaters, water plants, knock-outs, or wells. "
                "Returns clarification candidates when the object name is ambiguous."
            ),
            args_schema=GetReadingForEntityInput,
        ),
        StructuredTool.from_function(
            func=compare_readings_between_dates,
            name="compare_readings_between_dates",
            description=(
                "Compare raw readings of one type between two dates. Returns "
                "compact precomputed comparison rows; use those rows directly "
                "instead of recalculating differences."
            ),
            args_schema=CompareReadingsInput,
        ),
        StructuredTool.from_function(
            func=find_missing_readings,
            name="find_missing_readings",
            description=(
                "Find configured entities that are missing a reading on a date. "
                f"Supported types: {', '.join(k for k, v in READING_DEFINITIONS.items() if v.supports_missing)}."
            ),
            args_schema=MissingReadingsInput,
        ),
        StructuredTool.from_function(
            func=find_all_missing_readings,
            name="find_all_missing_readings",
            description=(
                "Find all configured entities missing any supported daily reading "
                "on a date. Use this when the user asks which readings are missing "
                "without naming a specific reading type."
            ),
            args_schema=AllMissingReadingsInput,
        ),
        StructuredTool.from_function(
            func=find_all_missing_readings_for_range,
            name="find_all_missing_readings_for_range",
            description=(
                "Find all configured entities missing any supported daily reading "
                "for each date in a date range. Use this instead of repeated "
                "find_all_missing_readings calls when the user asks about missing "
                "readings or missing data entries for a week, month, last calendar "
                "week, or any multi-day range."
            ),
            args_schema=AllMissingReadingsRangeInput,
        ),
        StructuredTool.from_function(
            func=list_equipment,
            name="list_equipment",
            description=(
                "List configured equipment entities without requiring readings or "
                "a date. Use this for inventory questions such as what tanks are "
                "in Battery 6, list flow meters for a battery, which pumps belong "
                "to a battery, or equipment filtered by metadata. Do not use "
                "reading search tools for simple equipment-list questions."
            ),
            args_schema=ListEquipmentInput,
        ),
        StructuredTool.from_function(
            func=search_readings,
            name="search_readings",
            description=(
                "Search any raw reading type across a date range, optionally filtered "
                "by entity name and numeric field conditions. Use for LACT, flare, "
                "tank, flow meter, well test, well fluid, well injection, run ticket, "
                "water draw, treater, knock-out, pump, and similar range queries."
            ),
            args_schema=SearchReadingsInput,
        ),
        StructuredTool.from_function(
            func=search_equipment_readings,
            name="search_equipment_readings",
            description=(
                "Search equipment readings using actual equipment-to-battery "
                "relations, equipment metadata, raw reading filters, and tank "
                "computed-volume filters. Use this for questions like Battery 6 "
                "flow meters with total > 100, Battery 13 knock-outs with inlet > 0, "
                "water pumps in a battery, or tanks in a battery containing oil/water. "
                "Do not infer battery relation from equipment names."
            ),
            args_schema=SearchEquipmentReadingsInput,
        ),
        StructuredTool.from_function(
            func=search_tank_readings,
            name="search_tank_readings",
            description=(
                "Search tank readings using tank metadata, battery relation, and "
                "computed volumes. Use this for tank oil/water stock, tanks in a "
                "battery, tanks containing oil/water, bottom-feet volume questions, "
                "or computed volume filters. Tank contents come from tanks.contents; "
                "only oil and water-oil contents are oil-capable. Mixed tanks provide "
                "oil_volume, recoverable_oil_volume, and water_volume. Use contains=oil "
                "for gross oil qualification; combine it with recoverable_oil_volume > 0 "
                "for recoverable oil qualification."
            ),
            args_schema=SearchTankReadingsInput,
        ),
        StructuredTool.from_function(
            func=search_well_tests,
            name="search_well_tests",
            description=(
                "Search well tests across a date range, optionally filtered by well "
                "name and numeric oil, water, gas, or runtime thresholds. Use this "
                "for questions like 'show all well tests between two dates with oil > 20'. "
                "For producing-well test coverage, search the complete range without "
                "well_name so the caller can compare all tested wells."
            ),
            args_schema=SearchWellTestsInput,
        ),
        StructuredTool.from_function(
            func=analyze_well_tests,
            name="analyze_well_tests",
            description=(
                "Analyze and compare well tests by well, battery, or selected site. "
                "Use latest_previous for each well's latest test versus its prior "
                "test, optionally rolled up by battery/site. Use range_summary for "
                "grouped test counts, sums, averages, minima, and maxima over a date "
                "range. Use recent_tests to compare the latest requested number of "
                "tests chronologically within each well. Use range_sequence to "
                "compare every test in a date range with that well's preceding test "
                "inside that range. Sequence modes require group_by=well. "
                "Prefer this tool over operational SQL for well-test analysis."
            ),
            args_schema=AnalyzeWellTestsInput,
        ),
    ]
