import pytest
from sqlalchemy import create_engine, text

from umai.clients.work_order_client import WorkOrderClient, WorkOrderClientError


def make_work_order_client() -> WorkOrderClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE work_orders (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    time TEXT NOT NULL,
                    subject TEXT,
                    status TEXT,
                    priority INTEGER,
                    vendor TEXT,
                    comments TEXT,
                    final_cost REAL,
                    cost_estimate REAL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE work_order_notes (
                    id INTEGER PRIMARY KEY,
                    work_order_id INTEGER NOT NULL,
                    notes TEXT,
                    time TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO work_orders
                    (site_id, time, subject, status, vendor, comments, final_cost, cost_estimate)
                VALUES
                    (4, '2026-05-01 08:00:00', 'Repair pump', 'complete',
                     'Vendor A', 'replace part', 100.25, 125),
                    (4, '2026-05-02 09:00:00', 'Repair motor', 'complete',
                     'Vendor B', 'replace motor', 200, 250),
                    (4, '2026-06-01 10:00:00', 'June work', 'open',
                     'Vendor A', 'later work', 300, 350),
                    (4, '2026-05-03 10:00:00', 'Estimate only repair', 'open',
                     'Vendor C', 'planned work', NULL, 75),
                    (5, '2026-05-01 08:00:00', 'Other site repair', 'complete',
                     'Vendor A', 'other site', 400, 450)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO work_order_notes
                    (id, work_order_id, notes, time, created_at, updated_at)
                VALUES
                    (100, 1, 'Pump was inspected before repair.', '2026-05-01 09:00:00',
                     '2026-05-01 09:00:00', '2026-05-01 09:00:00'),
                    (101, 1, 'Replacement completed.', '2026-05-01 12:00:00',
                     '2026-05-01 12:00:00', '2026-05-01 12:00:00'),
                    (102, 2, 'Motor repair note.', '2026-05-02 10:00:00',
                     '2026-05-02 10:00:00', '2026-05-02 10:00:00')
                """
            )
        )
    return WorkOrderClient(engine)


def test_summarize_work_order_costs_returns_exact_sql_totals():
    result = make_work_order_client().summarize_costs(
        site_id=4,
        start_date="2026-05-01",
        end_date="2026-05-31",
        metric="cost",
    )

    assert result["work_order_count"] == 3
    assert result["primary_total_field"] == "final_cost"
    assert result["primary_total"] == 300.25
    assert result["totals_by_field"] == {
        "final_cost": 300.25,
        "cost_estimate": 450,
    }
    assert result["value_counts_by_field"] == {
        "final_cost": 2,
        "cost_estimate": 3,
    }
    assert result["requested_total_fields"] == ("final_cost",)
    assert result["requested_totals_by_field"] == {"final_cost": 300.25}


def test_summarize_work_order_estimates_uses_estimate_columns():
    result = make_work_order_client().summarize_costs(
        site_id=4,
        start_date="2026-05-01",
        end_date="2026-05-31",
        metric="estimate",
    )

    assert result["primary_total_field"] == "cost_estimate"
    assert result["primary_total"] == 450
    assert result["totals_by_field"] == {
        "final_cost": 300.25,
        "cost_estimate": 450,
    }
    assert result["requested_total_fields"] == ("cost_estimate",)
    assert result["requested_totals_by_field"] == {"cost_estimate": 450}


def test_summarize_work_order_costs_can_filter_by_text():
    result = make_work_order_client().summarize_costs(
        site_id=4,
        start_date="2026-05-01",
        end_date="2026-05-31",
        metric="all",
        search_text="motor",
    )

    assert result["work_order_count"] == 1
    assert result["primary_total_field"] == "final_cost"
    assert result["primary_total"] == 200
    assert result["totals_by_field"] == {
        "final_cost": 200,
        "cost_estimate": 250,
    }


def test_summarize_work_order_costs_includes_estimate_when_final_costs_are_missing():
    result = make_work_order_client().summarize_costs(
        site_id=4,
        start_date="2026-05-03",
        end_date="2026-05-03",
        metric="cost",
    )

    assert result["work_order_count"] == 1
    assert result["primary_total_field"] == "final_cost"
    assert result["primary_total"] is None
    assert result["totals_by_field"] == {
        "final_cost": None,
        "cost_estimate": 75,
    }
    assert result["value_counts_by_field"] == {
        "final_cost": 0,
        "cost_estimate": 1,
    }
    assert result["requested_totals_by_field"] == {"final_cost": None}


def test_summarize_work_order_costs_validates_date_order():
    with pytest.raises(WorkOrderClientError, match="start_date cannot be after"):
        make_work_order_client().summarize_costs(
            site_id=4,
            start_date="2026-05-31",
            end_date="2026-05-01",
        )


def test_summarize_work_order_costs_requires_supported_amount_columns():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE work_orders (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    time TEXT NOT NULL,
                    subject TEXT
                )
                """
            )
        )

    with pytest.raises(WorkOrderClientError, match="final_cost"):
        WorkOrderClient(engine).summarize_costs(
            site_id=4,
            start_date="2026-05-01",
            end_date="2026-05-31",
            metric="cost",
        )
