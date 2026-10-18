import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text

from omai.clients.data_point_client import DataPointClient


def make_client() -> DataPointClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE data_points (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    facility_id INTEGER,
                    facility_name TEXT,
                    device_name TEXT,
                    data_point_name TEXT,
                    tag TEXT,
                    type TEXT,
                    active INTEGER
                )
                """
            )
        )
        connection.execute(text("CREATE TABLE sites (id INTEGER PRIMARY KEY, `key` TEXT)"))
        connection.execute(
            text(
                """
                CREATE TABLE wells (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT,
                    `key` TEXT,
                    pump_type TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE tanks (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT,
                    `key` TEXT,
                    data_point_facility_id INTEGER,
                    data_point_device_name TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE data_point_data (
                    id INTEGER PRIMARY KEY,
                    data_point_id INTEGER UNIQUE,
                    data TEXT,
                    last_update INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO sites (id, `key`) VALUES (1, 'HARTZOG'), (2, 'OTHER')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO wells (id, site_id, name, `key`, pump_type)
                VALUES (1, 1, 'HARTZOG DRAW UNIT 2510', 'HDU_2510', 'rod')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO data_points
                    (id, site_id, facility_id, facility_name, device_name, data_point_name, tag, type, active)
                VALUES
                    (1, 1, NULL, 'HDU_2510', NULL, NULL, NULL, 'facility', 1),
                    (2, 1, 1, NULL, 'HDU_2510', 'stroke_min', 'stroke_min', 'number', 1),
                    (3, 1, 1, NULL, 'HDU_2510', 'min_load_ls', 'min_load_ls', 'number', 1),
                    (4, 1, 1, NULL, 'HDU_2510', 'min_load_sp', 'min_load_sp', 'number', 0),
                    (5, 1, 1, NULL, 'HDU_2510', 'min_load_consecutive_sp', 'min_load_consecutive_sp', 'number', 1),
                    (6, 1, 1, NULL, 'HDU_2510', 'yesterday_min_load', 'yesterday_min_load', 'number', 1),
                    (7, 1, 1, NULL, 'Controller A', 'pressure', 'pressure', 'number', 1),
                    (8, 2, NULL, 'HDU_2510', 'HDU_2510', 'stroke_min', 'stroke_min', 'number', 1),
                    (9, 1, 1, NULL, 'HDU_2510', 'bad_payload', 'bad_payload', 'number', 1),
                    (10, 1, NULL, 'ABC_9999', 'ABC_9999', 'value', 'value', 'number', 1),
                    (11, 1, NULL, 'XYZ_9999', 'XYZ_9999', 'value', 'value', 'number', 1),
                    (12, 1, 1, NULL, 'HDU_2510', 'Pump Fillage', 'pump_fillage', 'number', 1),
                    (20, 1, NULL, 'TankFacility', NULL, NULL, NULL, 'facility', 1),
                    (21, 1, 20, NULL, 'Tank Device', 'Level', 'level', 'number', 1)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO tanks
                    (id, site_id, name, `key`, data_point_facility_id, data_point_device_name)
                VALUES
                    (1, 1, 'Tank - 10-1 Oil', 'Tank_10_1_Oil', 20, 'Tank Device')
                """
            )
        )
        rows = [
            (2, {"value": 4.84, "type": "Real", "timestamp": 1_717_241_400}),
            (3, {"value": 15449, "type": "Real", "timestamp": 1_717_241_400}),
            (4, {"value": 13800, "type": "Real", "timestamp": 1_717_241_400}),
            (5, {"value": 5, "type": "Real", "timestamp": 1_717_241_400}),
            (6, {"value": 15297, "type": "Real", "timestamp": 1_717_241_400}),
            (7, {"value": 75, "type": "Real"}),
            (8, {"value": 999, "type": "Real", "timestamp": 1_717_241_400}),
        ]
        for index, (data_point_id, payload) in enumerate(rows, start=1):
            connection.execute(
                text(
                    "INSERT INTO data_point_data (id, data_point_id, data, last_update) "
                    "VALUES (:id, :data_point_id, :data, :last_update)"
                ),
                {
                    "id": index,
                    "data_point_id": data_point_id,
                    "data": json.dumps(payload),
                    "last_update": 1_717_241_500,
                },
            )
        connection.execute(
            text(
                "INSERT INTO data_point_data (id, data_point_id, data, last_update) "
                "VALUES (20, 9, 'not-json', 1717241500)"
            )
        )
    return DataPointClient(engine, "UTC", http_get=fake_graphite_get)


class fake_graphite_get:
    """Tiny callable response double for Graphite render requests."""

    last_params = None

    def __new__(cls, url, params=None, timeout=None):
        cls.last_params = params
        return cls

    @staticmethod
    def raise_for_status():
        return None

    @staticmethod
    def json():
        base = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp())
        return [
            {
                "target": "test",
                "datapoints": [
                    [10, base],
                    [12, base + 3600],
                    [14, base + 7200],
                    [16, base + 10800],
                    [18, base + 14400],
                    [20, base + 18000],
                ],
            }
        ]


def test_resolves_facility_shorthand_and_exact_normalized_data_point():
    result = make_client().get_values(1, "2510", "stroke min")

    assert result["status"] == "found"
    assert result["facility_name"] == "HDU_2510"
    assert result["device_name"] == "HDU_2510"
    assert result["device_defaulted"] is True
    assert result["values"][0]["data_point_name"] == "stroke_min"
    assert result["values"][0]["value"] == 4.84
    assert result["values"][0]["received_at"].endswith("UTC")


def test_returns_only_closest_prefix_data_point_matches_and_includes_inactive():
    result = make_client().get_values(1, "HDU 2510", "min load")

    assert [item["data_point_name"] for item in result["values"]] == [
        "min_load_ls",
        "min_load_sp",
    ]
    assert result["values"][1]["active"] is False


def test_explicit_device_is_matched_with_normalization():
    result = make_client().get_values(
        1, "HDU-2510", "pressure", device_name="Controller-A"
    )

    assert result["status"] == "found"
    assert result["device_defaulted"] is False
    assert result["device_name"] == "Controller A"
    assert result["values"][0]["value"] == 75
    assert result["values"][0]["received_at"].endswith("UTC")


def test_facility_ambiguity_returns_candidates_without_values():
    result = make_client().get_values(1, "9999", "value")

    assert result["status"] == "ambiguous"
    assert result["selector"] == "facility"
    assert result["candidates"] == ["ABC_9999", "XYZ_9999"]
    assert "values" not in result


def test_malformed_payload_is_reported_without_crashing():
    result = make_client().get_values(1, "2510", "bad payload")

    value = result["values"][0]
    assert value["value_available"] is False
    assert value["received_at"].endswith("UTC")
    assert value["data_error"] == "Current value payload is invalid JSON."


def test_lookup_is_strictly_site_scoped():
    result = make_client().get_values(1, "2510", "stroke min")

    assert result["values"][0]["value"] == 4.84


def test_analyzes_rod_pump_trend_with_fuzzy_data_point_name():
    result = make_client().analyze_trend(1, "2510", "pump filage", 7)

    assert result["status"] == "found"
    assert result["data_point_name"] == "Pump Fillage"
    assert result["summary"]["trend"] == "rising"
    assert "MI3.HARTZOG.HDU_2510.pump_fillage" in fake_graphite_get.last_params["target"]


def test_analyzes_equipment_trend_through_configured_facility_and_device():
    result = make_client().analyze_trend(
        1,
        "10-1 oil",
        "leve",
        7,
        equipment_type="tank",
    )

    assert result["status"] == "found"
    assert result["equipment"]["name"] == "Tank - 10-1 Oil"
    assert result["data_point_name"] == "Level"
    assert "MI3.HARTZOG.TankFacility.level" in fake_graphite_get.last_params["target"]


def test_summary_describes_typical_band_and_zero_dropout():
    samples = [
        {
            "time": "07/01/2026 00:00:00 UTC",
            "timestamp": "2026-07-01T00:00:00+00:00",
            "value": value,
        }
        for value in [
            20_500,
            21_000,
            19_800,
            20_900,
            18_700,
            0,
            20_200,
            19_500,
            21_200,
            18_900,
            18_200,
        ]
    ]

    summary = make_client()._summarize_samples(samples)

    assert summary["typical_band"]["low"] > 0
    assert summary["typical_band"]["high"] > summary["typical_band"]["low"]
    assert summary["trend"] == "falling"
    assert summary["variability"] in {"moderate", "high"}
    assert summary["overall_shape"] in {"gradual_decrease", "highly_variable", "oscillating"}
    assert summary["segments"][0]["label"] == "beginning"
    assert any(change["direction"] == "drop" for change in summary["significant_changes"])
    assert any("Drop to zero" in anomaly for anomaly in summary["anomalies"])


def test_summary_classifies_flat_series_with_high_spike_plateau():
    normal_values = [
        15_520,
        15_640,
        15_780,
        15_850,
        15_590,
        15_710,
        15_820,
        15_760,
    ] * 20
    values = normal_values[:60] + [45_626.4] + ([65_535] * 8) + normal_values[60:]
    start = datetime(2026, 6, 15, tzinfo=timezone.utc)
    samples = [
        {
            "time": (start + timedelta(hours=index)).strftime("%m/%d/%Y %H:%M:%S UTC"),
            "timestamp": (start + timedelta(hours=index)).isoformat(),
            "value": value,
        }
        for index, value in enumerate(values)
    ]

    summary = make_client()._summarize_samples(samples)

    assert summary["pattern"] == "mostly_flat"
    assert summary["variability"] == "low"
    assert summary["overall_shape"] == "mostly_flat"
    assert any(change["direction"] == "spike" for change in summary["significant_changes"])
    assert summary["average_distorted_by_outliers"] is True
    assert any("Abnormal high spike/plateau" in anomaly for anomaly in summary["anomalies"])
    assert not any("point-to-point drop" in anomaly for anomaly in summary["anomalies"])


def test_all_points_for_facility_includes_all_devices_and_unavailable_values():
    result = make_client().get_values(1, facility_name='2510')

    assert result['status'] == 'found'
    assert result['match_count'] == 8
    assert {item['device_name'] for item in result['values']} == {'HDU_2510', 'Controller A'}
    assert all(item['facility_name'] == 'HDU_2510' for item in result['values'])
    bad = next(item for item in result['values'] if item['data_point_name'] == 'bad_payload')
    assert bad['value_available'] is False
    assert any(item['active'] is False for item in result['values'])


def test_named_point_across_all_facilities_is_site_scoped_and_reports_missing():
    result = make_client().get_values(1, data_point_name='stroke min')

    assert result['scope'] == 'all_facilities'
    assert result['status'] == 'partial'
    assert result['match_count'] == 1
    assert result['values'][0]['value'] == 4.84
    assert result['values'][0]['facility_name'] == 'HDU_2510'
    assert any(item['status'] == 'not_found' for item in result['facility_results'])


def test_multiple_explicit_facilities_are_not_ambiguous_and_are_deduplicated():
    result = make_client().get_values(
        1, facility_names=['ABC_9999', 'XYZ_9999', 'ABC 9999'], data_point_name='value'
    )

    assert result['status'] == 'found'
    assert result['match_count'] == 2
    assert {item['facility_name'] for item in result['values']} == {'ABC_9999', 'XYZ_9999'}
    assert all(item['value_available'] is False for item in result['values'])


def test_ambiguous_or_missing_explicit_facility_does_not_return_partial_scope():
    client = make_client()
    ambiguous = client.get_values(1, facility_names=['2510', '9999'], data_point_name='value')
    missing = client.get_values(1, facility_names=['2510', 'missing'], data_point_name='value')

    assert ambiguous['status'] == 'ambiguous'
    assert ambiguous['selector'] == 'facility'
    assert 'values' not in ambiguous
    assert missing['status'] == 'not_found'
    assert missing['selector'] == 'facility'
    assert 'values' not in missing


def test_multiple_facilities_without_point_returns_all_points():
    result = make_client().get_values(1, facility_names=['2510', 'TankFacility'])

    assert result['status'] == 'found'
    assert result['match_count'] == 9
    assert any(item['device_name'] == 'Tank Device' for item in result['values'])


def test_broad_queries_filter_devices_and_keep_per_device_point_matching():
    client = make_client()
    with client.engine.begin() as connection:
        connection.execute(text("INSERT INTO data_points (id, site_id, facility_name, device_name, data_point_name) VALUES (30, 1, 'ABC_9999', 'Controller A', 'min_load')"))
        connection.execute(text("INSERT INTO data_points (id, site_id, facility_name, device_name, data_point_name) VALUES (31, 1, 'ABC_9999', 'Controller B', 'min_load_ls')"))
    result = client.get_values(1, facility_names=['ABC_9999'], data_point_name='min load')
    assert {item['data_point_name'] for item in result['values']} == {'min_load', 'min_load_ls'}
    filtered = client.get_values(1, data_point_name='pressure', device_name='Controller-A')
    assert filtered['match_count'] == 1
    assert filtered['values'][0]['value'] == 75
    assert filtered['values'][0]['device_name'] == 'Controller A'


def test_value_scope_validation_rejects_empty_or_conflicting_selectors():
    import pytest
    from omai.clients.data_point_client import DataPointClientError

    client = make_client()
    for arguments in (
        {}, {'device_name': 'Controller A'}, {'facility_names': []},
        {'facility_name': ' '}, {'facility_names': ['2510', '']},
        {'data_point_name': ''}, {'data_point_name': '---'},
        {'facility_name': '2510', 'data_point_name': ''},
        {'facility_name': '2510', 'facility_names': ['2510']},
        {'data_point_name': 'pressure', 'device_name': ''},
    ):
        with pytest.raises(DataPointClientError):
            client.get_values(1, **arguments)


def test_no_match_and_empty_site_return_no_values():
    client = make_client()
    for site_id, point in ((1, 'missing point'), (999, 'stroke min')):
        result = client.get_values(site_id, data_point_name=point)
        assert result['status'] == 'not_found'
        assert result['values'] == []


def test_value_tool_exposes_new_scopes_and_returns_structured_errors():
    from omai.tools.data_point_tools import build_data_point_tools
    from omai.clients.data_point_client import UnavailableDataPointClient

    tool = build_data_point_tools(make_client(), site_id=1)[0]
    for arguments, count in (
        ({'facility_name': '2510'}, 8),
        ({'facility_names': ['ABC_9999', 'XYZ_9999'], 'data_point_name': 'value'}, 2),
        ({'data_point_name': 'stroke min'}, 1),
    ):
        result = json.loads(tool.invoke(arguments))
        assert result['ok'] is True
        assert result['match_count'] == count
    invalid = json.loads(tool.invoke({}))
    assert invalid['ok'] is False
    unavailable = build_data_point_tools(UnavailableDataPointClient('Unavailable'), site_id=1)[0]
    assert json.loads(unavailable.invoke({'facility_names': ['2510']})) == {'ok': False, 'error': 'Unavailable'}
