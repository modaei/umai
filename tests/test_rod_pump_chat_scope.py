from omai.services.chat_service import is_rod_pump_fleet_question


def test_blocks_expensive_rod_pump_fleet_questions():
    assert is_rod_pump_fleet_question("Rank all rod-pump wells by current health")
    assert is_rod_pump_fleet_question("Which wells have the highest health score?")
    assert is_rod_pump_fleet_question("Which pump wells need urgent health intervention?")


def test_allows_one_exact_well_question():
    assert not is_rod_pump_fleet_question("Analyze rod-pump health for well 5823")


def test_allows_rod_pump_telemetry_rankings():
    for question in (
        'which rod pump has the highest time drift data point?',
        'Which rod-pump well has the lowest time drift datapoint?',
        'Rank all rod pumps by their data_point stroke min values',
        'Show data-point pressure for all rod pumps',
        'Which rod pump has the highest stroke min?',
    ):
        assert not is_rod_pump_fleet_question(question), question


def test_still_blocks_explicit_fleet_health_analysis():
    for question in (
        'Rank all rod pumps by health',
        'Which wells have the highest health score?',
    ):
        assert is_rod_pump_fleet_question(question), question


def test_health_word_is_required_for_fleet_block():
    for question in (
        'Which wells have the highest mechanical risk?',
        'Which pump wells need urgent intervention?',
        'Analyze dynographs for all rod pump wells',
        'Show paraffin risk for all rod pumps',
        'Diagnoses for all rod pump wells',
        'Rank all rod pumps by operating condition',
        'Show all unhealthy rod pumps',
        'Rank all rod pumps by healthcare costs',
    ):
        assert not is_rod_pump_fleet_question(question), question
    assert is_rod_pump_fleet_question('Rank all rod pumps by HEALTH')
