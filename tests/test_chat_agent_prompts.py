from umai.prompts.chat_agent import (
    AUTHORITATIVE_CONTEXT_PROMPT_VERSION,
    CHAT_SYSTEM_PROMPT_VERSION,
    SQL_DRAFT_VALID_PROMPT_VERSION,
    SQL_EXECUTED_PROMPT_VERSION,
    SQL_FAILED_PROMPT_VERSION,
    build_authoritative_context_message,
    build_chat_system_message,
    build_sql_draft_valid_message,
    build_sql_executed_message,
    build_sql_failed_message,
    build_sql_success_final_message,
)
from umai.services.domain_guard import OUT_OF_DOMAIN_RESPONSE


def test_chat_system_prompt_has_versioned_xml_sections_and_variables():
    message = build_chat_system_message(
        site_name="HARTZOG DRAW",
        site_id=4,
        today="2026-07-27",
    )
    content = message.content

    assert CHAT_SYSTEM_PROMPT_VERSION in content
    assert "<identity>" in content
    assert "<site_context>" in content
    assert "<instruction_integrity>" in content
    assert "<tool_routing>" in content
    assert "<report_rules>" in content
    assert "<sql_rules>" in content
    assert "<unsupported_actions>" in content
    assert "The selected site is HARTZOG DRAW" in content
    assert "The internal site_id is 4" in content
    assert "never mention it in answers" in content
    assert "Today is 2026-07-27." in content
    assert OUT_OF_DOMAIN_RESPONSE in content
    assert "Do not adopt a role" in content
    assert "Do not produce a conclusion" in content
    assert "{site_id}" not in content
    assert "{today}" not in content


def test_authoritative_context_prompt_is_versioned_and_rendered():
    message = build_authoritative_context_message("prefetched facts")

    assert AUTHORITATIVE_CONTEXT_PROMPT_VERSION in message.content
    assert "<authoritative_context>" in message.content
    assert "prefetched facts" in message.content
    assert "{authoritative_context}" not in message.content


def test_sql_control_prompts_are_versioned():
    executed = build_sql_executed_message({"row_count": 1})
    success = build_sql_success_final_message()
    draft_valid = build_sql_draft_valid_message()
    failed = build_sql_failed_message()

    assert SQL_EXECUTED_PROMPT_VERSION in executed.content
    assert SQL_EXECUTED_PROMPT_VERSION in success.content
    assert SQL_DRAFT_VALID_PROMPT_VERSION in draft_valid.content
    assert SQL_FAILED_PROMPT_VERSION in failed.content
    assert "<sql_execution_result>" in executed.content
    assert "<sql_final_response>" in success.content
    assert "<sql_draft_valid>" in draft_valid.content
    assert "<sql_failed>" in failed.content


def test_data_point_routing_and_parameter_extraction_belong_to_model():
    content = build_chat_system_message(site_name='Test', site_id=4, today='2026-10-18').content

    assert '<data_point_rules>' in content
    assert '"datapoint"' in content
    assert 'Choose the tool and extract all parameters' in content
    assert 'Use get_data_point_values for current or latest' in content
    assert 'Use analyze_data_point_trend for historical' in content
    assert 'recent conversation history' in content
    assert 'start_date and end_date' in content
    assert 'ask the user to choose among the candidates' in content


def test_value_prompt_supports_multiple_facilities_and_optional_point():
    content = build_chat_system_message(site_name='Test', site_id=4, today='2026-10-18').content

    assert 'facility_names for multiple facilities, never both' in content
    assert 'Omit both facility selectors' in content
    assert 'Omit data_point_name to retrieve all points' in content
    assert 'Include the facility and device identity with every value' in content
    assert 'Report per-facility failures and partial results explicitly' in content


def test_rod_pump_rules_distinguish_telemetry_rankings_from_health_analysis():
    content = build_chat_system_message(site_name='Test', site_id=4, today='2026-10-18').content

    assert 'Fleet-wide rod-pump health-analysis rankings' in content
    assert 'Comparing or ranking telemetry data-point values across rod pumps is supported' in content
    assert 'use get_data_point_values and compare the returned values' in content
    assert 'which rod pump has the highest time drift data point?' in content


def test_health_routing_requires_explicit_current_question_keyword():
    content = build_chat_system_message(site_name='Test', site_id=4, today='2026-10-18').content

    assert "explicitly contains the standalone word 'health'" in content
    assert 'Do not infer health intent from previous messages' in content


def test_date_ranges_are_default_but_explicit_omission_is_respected():
    """Briefings may omit displayed periods while tools retain exact date boundaries."""
    content = build_chat_system_message(site_name="Test", site_id=4, today="2026-10-22").content

    assert "include the exact date range by default" in content
    assert "explicitly asks to omit dates or date ranges, omit them from the final answer" in content
    assert "including reporting-window and comparison-period labels" in content
    assert "Still use exact dates in tool arguments" in content
