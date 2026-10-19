from datetime import datetime, timedelta, timezone
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx
import pytest
from PIL import Image
from sqlalchemy import create_engine, text

from omai.rod_pump_health.parameter_trends import (
    ParameterTrendAnalyzer, SYSTEM_PROMPT,
    PARAMETERS, monitoring_samples, parameter_analyzer_from_settings,
)
from omai.rod_pump_health.reader import RodPumpHealthAnalyzer


NOW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)


@pytest.fixture
def analyzer():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        for statement in [
            "CREATE TABLE wells(id INTEGER, site_id INTEGER, pump_type TEXT, `key` TEXT)",
            "CREATE TABLE sites(id INTEGER, `key` TEXT)",
            "CREATE TABLE report_configurations(site_id INTEGER, function_name TEXT, "
            "related_entities TEXT)",
            "CREATE TABLE data_points(id INTEGER, site_id INTEGER, facility_id INTEGER, "
            "facility_name TEXT, data_point_name TEXT, tag TEXT, active INTEGER)",
            "CREATE TABLE rod_pump_health_analyses(well_id INTEGER, findings TEXT)",
            "INSERT INTO wells VALUES(23,4,'rod','HDU_5125'),(24,5,'rod','OTHER')",
            "INSERT INTO sites VALUES(4,'HARTZOG'),(5,'OTHER')",
            "INSERT INTO data_points VALUES(1,4,NULL,'HDU_5125',NULL,NULL,1)",
            "INSERT INTO data_points VALUES(2,4,1,NULL,'Pump Fillage','pump_fillage',0)",
            "INSERT INTO data_points VALUES(3,4,1,NULL,'Max Load LS','peak_load_ls',1)",
            "INSERT INTO data_points VALUES(4,5,NULL,'OTHER','Pump Fillage','pump_fillage',1)",
            "INSERT INTO data_points VALUES(5,4,1,NULL,'Stroke Seconds','stroke_seconds',0)",
        ]:
            connection.execute(text(statement))
        connection.execute(text(
            "INSERT INTO report_configurations VALUES(4,'rod_pump_monitoring',:ids)"
        ), {"ids": json.dumps({"well_ids": [23]})})
    calls = {"http": [], "llm": []}

    def http_get(url, params, timeout):
        calls["http"].append(params)
        targets = [value for name, value in params if name == "target"]
        series = [
            {"target": target, "datapoints": [
                [80 - index, (NOW - timedelta(days=29 - index)).timestamp()]
                for index in range(29)
            ]} for target in targets
        ]
        return httpx.Response(200, json=series, request=httpx.Request("GET", url))

    def reviewer(content):
        calls["llm"].append(content)
        return {
            "trends": [{"parameter": "Pump Fillage", "title": "Sustained decline"}],
            "interpretation": "Together these changes suggest insufficient inflow in SAM1 operation.",
            "token_usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
        }

    health = RodPumpHealthAnalyzer(engine, engine, "UTC", "unused", None)
    yield ParameterTrendAnalyzer(health, "http://monitoring/render", reviewer, http_get), calls
    engine.dispose()


def test_monthly_review_sends_parameter_graphs_without_numeric_json_or_cards(analyzer):
    service, calls = analyzer
    result = service.analyze(23, NOW)
    assert result["status"] == "complete"
    assert result["window_start_at"] == (NOW - timedelta(days=30)).isoformat()
    assert result["trends"] == [{"parameter": "Pump Fillage", "title": "Sustained decline"}]
    assert result["token_usage"]["total_tokens"] == 130
    targets = [value for key, value in calls["http"][0] if key == "target"]
    assert targets == [
        "MI3.HARTZOG.HDU_5125.pump_fillage", "MI3.HARTZOG.HDU_5125.peak_load_last_stroke",
    ]
    assert len(calls["llm"]) == 1
    content = calls["llm"][0]
    assert [part["type"] for part in content] == ["text", "image_url", "image_url"]
    assert "median" not in content[0]["text"]
    assert "80" not in content[0]["text"]
    for part in content[1:]:
        url = part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert image.size == (1100, 450)
    assert len(result["parameters"]["Pump Fillage"]) == 29
    assert "Yesterday Cycles" in result["missing_parameters"]
    assert len(PARAMETERS) == 6
    assert "Stroke Seconds" not in result["parameters"]
    with pytest.raises(ValueError, match="scope"):
        service.analyze(24, NOW)


def test_cached_monthly_review_skips_only_its_own_llm_call(analyzer):
    service, calls = analyzer
    first = service.analyze(23, NOW)
    first["last_refresh_error"] = "A later refresh failed."
    with service.engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO rod_pump_health_analyses VALUES(23,:findings)"
        ), {"findings": json.dumps({"parameter_trends": first})})
    cached = service.analyze(23, NOW)
    assert cached["unchanged"] is True
    assert cached["last_refresh_error"] is None
    assert len(calls["llm"]) == 1
    service.analyze(23, NOW + timedelta(days=1))
    assert len(calls["llm"]) == 2


def test_missing_history_is_not_a_healthy_verdict(analyzer):
    service, calls = analyzer
    service.http_get = lambda url, **kwargs: httpx.Response(
        200, json=[], request=httpx.Request("GET", url),
    )
    result = service.analyze(23, NOW)
    assert result["status"] == "insufficient_data"
    assert result["parameters"] == {}
    assert calls["llm"] == []


def test_monitoring_samples_reject_outside_values_and_preserve_null_gaps():
    start = NOW - timedelta(days=30)
    samples = [
        [0, start.timestamp()], [10, start.timestamp() + 3600],
        [None, start.timestamp() + 7200], [float("nan"), start.timestamp() + 7200],
        [100, start.timestamp() - 1], [100, NOW.timestamp()],
    ]
    assert monitoring_samples(samples, start, NOW) == [
        {"timestamp": start.timestamp(), "value": 0},
        {"timestamp": start.timestamp() + 3600, "value": 10},
        {"timestamp": start.timestamp() + 7200, "value": None},
    ]


def test_parameter_factory_uses_its_own_prompt_and_separate_reviewer():
    settings = SimpleNamespace(
        monitoring_data_api_url="http://metrics", data_point_trend_timeout_seconds=20,
    )
    health = SimpleNamespace(engine=None, timezone=ZoneInfo("UTC"))
    with patch("omai.rod_pump_health.parameter_trends.analyzer_from_settings", return_value=health):
        with patch("omai.rod_pump_health.parameter_trends.llm_reviewer") as reviewer:
            service = parameter_analyzer_from_settings(settings)
            service.reviewer([{"type": "text", "text": "parameters"}])
    reviewer.assert_called_once_with(settings, system_prompt=SYSTEM_PROMPT)


def test_no_unusual_trends_retains_combined_interpretation(analyzer):
    service, _ = analyzer
    service.reviewer = lambda _: {"trends": [], "interpretation": "No unusual monthly trends."}
    result = service.analyze(23, NOW)
    assert result["status"] == "complete"
    assert result["trends"] == []
    assert result["interpretation"] == "No unusual monthly trends."


@pytest.mark.parametrize("cached", [False, True])
def test_health_endpoint_skips_all_optional_calls_by_default(monkeypatch, cached):
    import importlib
    from omai.rod_pump_health.reader import WellAnalysisRequest

    module = importlib.import_module("omai.api.app")
    assert module.EXTENDED_HEALTH_ANALYSIS_ENABLED is False
    health = SimpleNamespace(
        _well=lambda well_id: {"id": well_id, "site_id": 4},
        analyze=lambda *args: {"well_id": 23, "review_status": "complete", "unchanged": cached},
    )
    monkeypatch.setattr(module, "analyzer_from_settings", lambda _: health)
    with patch.object(module, "parameter_analyzer_from_settings") as parameters:
        with patch.object(module, "conclusion_analyzer_from_settings") as conclusion:
            app = module.create_app()
            route = next(r for r in app.routes if r.path == "/rod-pump-health-analysis")
            result = route.endpoint(WellAnalysisRequest(well_id=23))
    assert result["review_status"] == "complete"
    assert result["parameter_trends"]["status"] == "disabled"
    assert result["conclusion"]["status"] == "disabled"
    parameters.assert_not_called()
    conclusion.assert_not_called()


@pytest.mark.parametrize("card_fails,trend_fails,cached,conclusion_fails", [
    (False, False, False, False), (False, False, True, False),
    (True, False, False, False), (False, True, False, False), (False, False, False, True),
])
def test_health_endpoint_reviews_in_order(
    monkeypatch, card_fails, trend_fails, cached, conclusion_fails,
):
    import importlib
    from omai.rod_pump_health.reader import WellAnalysisRequest

    module = importlib.import_module("omai.api.app")
    monkeypatch.setattr(module, "EXTENDED_HEALTH_ANALYSIS_ENABLED", True)
    calls = []

    class CardAnalyzer:
        def _well(self, well_id):
            return {"id": well_id, "site_id": 4}

        def analyze(self, well_id, recent_card=None):
            calls.append(("cards", well_id))
            if card_fails:
                raise RuntimeError("Card calculation failed.")
            return {"well_id": well_id, "site_id": 4,
                    "review_status": "complete", "unchanged": cached}

    class ParameterAnalyzer:
        def analyze(self, well_id):
            calls.append(("parameters", well_id))
            if trend_fails:
                raise RuntimeError("Monitoring request failed.")
            return {"well_id": well_id, "status": "complete", "trends": []}

    class FinalAnalyzer:
        def analyze(self, well_id, cards, parameters):
            calls.append(("conclusion", well_id))
            if conclusion_fails:
                raise RuntimeError("Conclusion LLM failed.")
            return {"well_id": well_id, "status": "complete", "title": "Assessment", "body": "Done."}

    monkeypatch.setattr(module, "analyzer_from_settings", lambda _: CardAnalyzer())
    monkeypatch.setattr(module, "parameter_analyzer_from_settings", lambda _: ParameterAnalyzer())
    monkeypatch.setattr(module, "conclusion_analyzer_from_settings", lambda _: FinalAnalyzer())
    app = module.create_app()
    route = next(r for r in app.routes if r.path == "/rod-pump-health-analysis")
    result = route.endpoint(WellAnalysisRequest(well_id=23))
    assert calls == [("cards", 23), ("parameters", 23), ("conclusion", 23)]
    assert result["review_status"] == ("failed" if card_fails else "complete")
    assert result["parameter_trends"]["status"] == ("failed" if trend_fails else "complete")
    assert result["conclusion"]["status"] == ("failed" if conclusion_fails else "complete")
    assert not any(r.path in {
        "/rod-pump-dynograph-analysis", "/rod-pump-parameter-trends-analysis",
    } for r in app.routes)
    with pytest.raises(ValueError):
        WellAnalysisRequest(well_id=23, cards=[])
