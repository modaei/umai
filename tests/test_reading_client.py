import pytest
from sqlalchemy import create_engine, text

from umai.clients.reading_client import (
    ReadingClient,
    ReadingClientError,
    UnavailableReadingClient,
)


def make_sqlite_client() -> ReadingClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE lacts (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE lact_readings (
                    id INTEGER PRIMARY KEY,
                    lact_id INTEGER NOT NULL,
                    reading REAL,
                    temperature REAL,
                    bs_w REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flares (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flare_readings (
                    id INTEGER PRIMARY KEY,
                    flare_id INTEGER NOT NULL,
                    pressure REAL,
                    volume REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO lacts (id, site_id, name, disable_reading)
                VALUES
                    (1, 1, 'LACT A', 0),
                    (2, 1, 'LACT B', 0),
                    (3, 1, 'LACT Disabled', 1),
                    (4, 2, 'Other Site LACT', 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flares (id, site_id, name, disable_reading)
                VALUES
                    (1, 1, 'Flare A', 0),
                    (2, 1, 'Flare B', 1),
                    (3, 2, 'Other Site Flare', 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO lact_readings
                    (id, lact_id, reading, temperature, bs_w, comments, time)
                VALUES
                    (10, 1, 123.4, 88.0, 0.1, 'normal', '2026-06-10 08:00:00'),
                    (11, 1, 140.0, 90.0, 0.1, 'next day', '2026-06-11 08:00:00'),
                    (12, 4, 999.0, 90.0, 0.1, 'other site', '2026-06-10 08:00:00')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flare_readings
                    (id, flare_id, pressure, volume, comments, time)
                VALUES
                    (20, 3, 10.0, 100.0, 'other site', '2026-06-10 08:00:00')
                """
            )
        )
    return ReadingClient(engine)


def make_equipment_relation_client() -> ReadingClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE batteries (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT
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
                    name TEXT NOT NULL,
                    key TEXT,
                    battery_id INTEGER,
                    type TEXT,
                    contents TEXT,
                    unusable_height REAL NOT NULL DEFAULT 0,
                    bbl_foot REAL,
                    monitored INTEGER NOT NULL DEFAULT 0,
                    disable_reading INTEGER NOT NULL DEFAULT 0,
                    non_linear_volume_mapping_id INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flow_meters (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT,
                    battery_id INTEGER,
                    type TEXT,
                    measurement_method TEXT,
                    monitored INTEGER NOT NULL DEFAULT 0,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flow_meter_readings (
                    id INTEGER PRIMARY KEY,
                    flow_meter_id INTEGER NOT NULL,
                    total REAL,
                    flow REAL,
                    odometer REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE water_plants (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT,
                    battery_id INTEGER,
                    monitored INTEGER NOT NULL DEFAULT 0,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE pumps (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT,
                    water_plant_id INTEGER,
                    type TEXT,
                    monitored INTEGER NOT NULL DEFAULT 0,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE pump_readings (
                    id INTEGER PRIMARY KEY,
                    pump_id INTEGER NOT NULL,
                    suction_pressure REAL,
                    discharge_pressure REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO batteries (id, site_id, name, key)
                VALUES
                    (6, 1, 'Battery 6', 'battery_6'),
                    (16, 1, 'Battery 16', 'battery_16')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO tanks
                    (id, site_id, name, key, battery_id, type, bbl_foot, monitored, disable_reading, non_linear_volume_mapping_id)
                VALUES
                    (101, 1, 'Tank 6-1 Oil', 'tank_6_1', 6, 'mixed-water-oil', 100.0, 1, 0, NULL),
                    (102, 1, 'Tank 6-2 Water', 'tank_6_2', 6, 'linear-volume', 50.0, 1, 0, NULL),
                    (103, 1, 'Tank 16-1 Oil', 'tank_16_1', 16, 'mixed-water-oil', 100.0, 1, 0, NULL)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flow_meters
                    (id, site_id, name, key, battery_id, type, measurement_method, monitored, disable_reading)
                VALUES
                    (1, 1, 'FM Battery 6 Gas', 'fm_b6_gas', 6, 'gas', 'total', 1, 0),
                    (2, 1, 'FM Battery 6 Water', 'fm_b6_water', 6, 'water', 'total', 1, 0),
                    (3, 1, 'FM Battery 16 Gas', 'fm_b16_gas', 16, 'gas', 'total', 1, 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flow_meter_readings
                    (id, flow_meter_id, total, flow, odometer, comments, time)
                VALUES
                    (10, 1, 150.0, 12.0, NULL, 'gas high', '2026-06-10 08:00:00'),
                    (11, 2, 90.0, 6.0, NULL, 'water low', '2026-06-10 08:00:00'),
                    (12, 3, 175.0, 11.0, NULL, 'battery 16', '2026-06-10 08:00:00')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO water_plants
                    (id, site_id, name, key, battery_id, monitored, disable_reading)
                VALUES
                    (20, 1, 'Water Plant 6', 'wp6', 6, 1, 0),
                    (21, 1, 'Water Plant 16', 'wp16', 16, 1, 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pumps
                    (id, site_id, name, key, water_plant_id, type, monitored, disable_reading)
                VALUES
                    (30, 1, 'Water Pump 6', 'pump6', 20, 'water', 1, 0),
                    (31, 1, 'Chemical Pump 6', 'chem6', 20, 'chemical', 1, 0),
                    (32, 1, 'Water Pump 16', 'pump16', 21, 'water', 1, 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pump_readings
                    (id, pump_id, suction_pressure, discharge_pressure, comments, time)
                VALUES
                    (40, 30, 30.0, 120.0, 'water pump', '2026-06-10 08:00:00'),
                    (41, 31, 20.0, 80.0, 'chemical pump', '2026-06-10 08:00:00'),
                    (42, 32, 40.0, 150.0, 'battery 16 pump', '2026-06-10 08:00:00')
                """
            )
        )
    return ReadingClient(engine)


def make_entity_resolution_client() -> ReadingClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        for table in (
            "lacts",
            "flares",
            "water_plants",
            "flow_meters",
            "treaters",
            "knock_outs",
            "pumps",
            "wells",
        ):
            connection.execute(
                text(
                    f"""
                    CREATE TABLE {table} (
                        id INTEGER PRIMARY KEY,
                        site_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        key TEXT
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
                    name TEXT NOT NULL,
                    key TEXT,
                    type TEXT NOT NULL,
                    contents TEXT,
                    unusable_height REAL NOT NULL DEFAULT 0,
                    bbl_foot REAL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flow_meter_readings (
                    id INTEGER PRIMARY KEY,
                    flow_meter_id INTEGER NOT NULL,
                    total REAL,
                    flow REAL,
                    odometer REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flare_readings (
                    id INTEGER PRIMARY KEY,
                    flare_id INTEGER NOT NULL,
                    pressure REAL,
                    volume REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE linear_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    level REAL,
                    feet INTEGER,
                    inches REAL,
                    temperature REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE mixed_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    top_level_feet INTEGER,
                    top_level_inches REAL,
                    water_level_feet INTEGER,
                    water_level_inches REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE non_linear_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    initial_feet INTEGER,
                    initial_inches REAL,
                    final_feet INTEGER,
                    final_inches REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flow_meters (id, site_id, name, key)
                VALUES
                    (1, 1, 'Battery 2 Vent', 'Battery2Vent'),
                    (2, 1, 'TRACT 5 Vent', 'TRACT5Vent'),
                    (3, 2, 'Battery 2 Vent', 'OtherSiteBattery2Vent')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flares (id, site_id, name, key)
                VALUES (1, 1, 'Battery 2 Vent', 'Battery2VentFlare')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO tanks (id, site_id, name, key, type, bbl_foot)
                VALUES
                    (1, 1, '11-1-1 Oil', '111Oil', 'mixed-water-oil', 100.0),
                    (2, 1, '10-1 Oil', '101Oil', 'linear-volume', 50.0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flow_meter_readings
                    (id, flow_meter_id, total, flow, odometer, comments, time)
                VALUES
                    (10, 1, 0.0, 1.5, 100.0, 'normal', '2026-06-01 07:00:00'),
                    (11, 2, 5.0, 2.0, 200.0, 'normal', '2026-06-01 07:00:00'),
                    (12, 3, 999.0, 9.0, 999.0, 'other site', '2026-06-01 07:00:00')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO mixed_tank_readings
                    (id, tank_id, top_level_feet, top_level_inches, water_level_feet, water_level_inches, comments, time)
                VALUES (20, 1, 8, 0, 1, 0, 'normal', '2026-06-01 07:00:00')
                """
            )
        )
    return ReadingClient(engine)


def make_sqlite_client_with_missing_exclusions() -> ReadingClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE report_configurations (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    function_name TEXT NOT NULL,
                    related_entities TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE lacts (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE lact_readings (
                    id INTEGER PRIMARY KEY,
                    lact_id INTEGER NOT NULL,
                    reading REAL,
                    temperature REAL,
                    bs_w REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flares (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flare_readings (
                    id INTEGER PRIMARY KEY,
                    flare_id INTEGER NOT NULL,
                    pressure REAL,
                    volume REAL,
                    comments TEXT,
                    time TEXT NOT NULL
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
                    name TEXT NOT NULL,
                    key TEXT,
                    type TEXT NOT NULL,
                    contents TEXT,
                    unusable_height REAL NOT NULL DEFAULT 0,
                    bbl_foot REAL,
                    monitored INTEGER NOT NULL DEFAULT 0,
                    disable_reading INTEGER NOT NULL DEFAULT 0,
                    battery_id INTEGER,
                    non_linear_volume_mapping_id INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE batteries (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE linear_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    level REAL,
                    feet INTEGER,
                    inches REAL,
                    percentage REAL,
                    pressure REAL,
                    temperature REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE mixed_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    top_level_feet INTEGER,
                    top_level_inches REAL,
                    water_level_feet INTEGER,
                    water_level_inches REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE non_linear_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    initial_feet INTEGER,
                    initial_inches REAL,
                    final_feet INTEGER,
                    final_inches REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE water_plants (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE water_plant_readings (
                    id INTEGER PRIMARY KEY,
                    water_plant_id INTEGER NOT NULL,
                    meter_reading REAL,
                    flow_rate REAL,
                    suction_pressure REAL,
                    discharge_pressure REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flow_meters (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE flow_meter_readings (
                    id INTEGER PRIMARY KEY,
                    flow_meter_id INTEGER NOT NULL,
                    total REAL,
                    flow REAL,
                    odometer REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE knock_outs (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE knock_out_readings (
                    id INTEGER PRIMARY KEY,
                    knock_out_id INTEGER NOT NULL,
                    inlet REAL,
                    oil_off REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE treaters (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE treater_readings (
                    id INTEGER PRIMARY KEY,
                    treater_id INTEGER NOT NULL,
                    oil_intake REAL,
                    pressure REAL,
                    temperature REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE pumps (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    disable_reading INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE pump_readings (
                    id INTEGER PRIMARY KEY,
                    pump_id INTEGER NOT NULL,
                    suction_pressure REAL,
                    discharge_pressure REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO report_configurations
                    (id, site_id, function_name, related_entities)
                VALUES
                    (
                        1,
                        1,
                        'hartzog_daily_missing',
                        '{"flare_ids":[1],"lact_ids":[1],"knock_out_ids":[1],"tank_ids":{"linear":[1],"mixed":[2],"nonLinear":[3]},"treater_ids":[1],"water_plant_ids":[1],"flow_meter_ids":{"water":[1],"gas":[2],"oil":[3]},"pump_ids":[1]}'
                    )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO lacts (id, site_id, name, disable_reading)
                VALUES (1, 1, 'LACT A', 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flares (id, site_id, name, disable_reading)
                VALUES (1, 1, 'Flare A', 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO tanks (id, site_id, name, type, bbl_foot, disable_reading)
                VALUES
                    (1, 1, 'Linear A', 'linear-volume', 50.0, 0),
                    (2, 1, 'Mixed A', 'mixed-water-oil', 100.0, 0),
                    (3, 1, 'Non Linear A', 'non-linear-volume', NULL, 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO water_plants (id, site_id, name, disable_reading)
                VALUES (1, 1, 'Water Plant A', 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO flow_meters (id, site_id, name, type, disable_reading)
                VALUES
                    (1, 1, 'Water Meter A', 'water', 0),
                    (2, 1, 'Gas Meter A', 'gas', 0),
                    (3, 1, 'Oil Meter A', 'oil', 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO knock_outs (id, site_id, name, disable_reading)
                VALUES (1, 1, 'Knock Out A', 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO treaters (id, site_id, name, disable_reading)
                VALUES (1, 1, 'Treater A', 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pumps (id, site_id, name, disable_reading)
                VALUES (1, 1, 'Pump A', 0)
                """
            )
        )
    return ReadingClient(engine)


def make_mixed_tank_client() -> ReadingClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE tanks (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT,
                    type TEXT NOT NULL,
                    contents TEXT,
                    unusable_height REAL NOT NULL DEFAULT 0,
                    bbl_foot REAL,
                    monitored INTEGER NOT NULL DEFAULT 0,
                    disable_reading INTEGER NOT NULL DEFAULT 0,
                    battery_id INTEGER,
                    non_linear_volume_mapping_id INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE batteries (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE mixed_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    top_level_feet INTEGER,
                    top_level_inches REAL,
                    water_level_feet INTEGER,
                    water_level_inches REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO tanks (id, site_id, name, type, bbl_foot, disable_reading)
                VALUES (1, 1, '2-1 Float Over', 'mixed-water-oil', 100.0, 0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO mixed_tank_readings
                    (
                        id,
                        tank_id,
                        top_level_feet,
                        top_level_inches,
                        water_level_feet,
                        water_level_inches,
                        comments,
                        time
                    )
                VALUES
                    (10, 1, 8, 0, 1, 2, 'first', '2026-05-09 08:00:00'),
                    (11, 1, 4, 1, 1, 2, 'second', '2026-05-10 08:00:00')
                """
            )
        )
    return ReadingClient(engine)


def make_tank_volume_client() -> ReadingClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE tanks (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT,
                    type TEXT NOT NULL,
                    contents TEXT NOT NULL,
                    unusable_height REAL NOT NULL DEFAULT 0,
                    bbl_foot REAL,
                    monitored INTEGER NOT NULL DEFAULT 0,
                    disable_reading INTEGER NOT NULL DEFAULT 0,
                    battery_id INTEGER,
                    non_linear_volume_mapping_id INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE batteries (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE linear_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    level REAL,
                    feet INTEGER,
                    inches REAL,
                    percentage REAL,
                    pressure REAL,
                    temperature REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE mixed_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    top_level_feet INTEGER,
                    top_level_inches REAL,
                    water_level_feet INTEGER,
                    water_level_inches REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE non_linear_tank_readings (
                    id INTEGER PRIMARY KEY,
                    tank_id INTEGER NOT NULL,
                    initial_feet INTEGER,
                    initial_inches REAL,
                    final_feet INTEGER,
                    final_inches REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE non_linear_tank_volume_mapping_details (
                    id INTEGER PRIMARY KEY,
                    tank_volume_mapping_id INTEGER NOT NULL,
                    feet INTEGER NOT NULL,
                    inches REAL NOT NULL,
                    volume REAL NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO batteries (id, site_id, name, key)
                VALUES (6, 1, 'Battery 6', 'battery_6')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO tanks
                    (id, site_id, name, key, type, contents, unusable_height, bbl_foot, monitored, disable_reading, battery_id, non_linear_volume_mapping_id)
                VALUES
                    (1, 1, 'Linear A', 'linear_a', 'linear-volume', 'water', 1.0, 50.0, 1, 0, 6, NULL),
                    (2, 1, 'Linear B', 'linear_b', 'linear-volume', 'oil', 1.0, 40.0, 0, 0, 6, NULL),
                    (3, 1, 'Mixed A', 'mixed_a', 'mixed-water-oil', 'water-oil', 2.0, 100.0, 1, 0, 6, NULL),
                    (4, 1, 'Mixed Missing', 'mixed_missing', 'mixed-water-oil', 'water-oil', 0, NULL, 0, 0, 6, NULL),
                    (5, 1, 'Non Linear A', 'non_linear_a', 'non-linear-volume', 'water', 0, NULL, 0, 0, 6, 50)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO non_linear_tank_volume_mapping_details
                    (id, tank_volume_mapping_id, feet, inches, volume)
                VALUES
                    (1, 50, 4, 0, 400.0),
                    (2, 50, 5, 0, 500.0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO linear_tank_readings
                    (id, tank_id, level, feet, inches, temperature, comments, time)
                VALUES
                    (10, 1, 2.5, NULL, NULL, NULL, 'level based', '2026-06-10 08:00:00'),
                    (11, 2, NULL, 3, 6, NULL, 'feet inches based', '2026-06-10 08:00:00')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO mixed_tank_readings
                    (id, tank_id, top_level_feet, top_level_inches, water_level_feet, water_level_inches, comments, time)
                VALUES
                    (20, 3, 4, 6, 1, 6, 'mixed complete', '2026-06-10 08:00:00'),
                    (21, 4, 4, 6, 1, 6, 'missing bbl foot', '2026-06-10 08:00:00')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO non_linear_tank_readings
                    (id, tank_id, initial_feet, initial_inches, final_feet, final_inches, comments, time)
                VALUES
                    (30, 5, 5, 0, 4, 0, 'non-linear unchanged', '2026-06-10 08:00:00')
                """
            )
        )
    return ReadingClient(engine)


def make_well_test_client() -> ReadingClient:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE batteries (
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
                CREATE TABLE wells (
                    id INTEGER PRIMARY KEY,
                    site_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    battery_id INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE well_tests (
                    id INTEGER PRIMARY KEY,
                    well_id INTEGER NOT NULL,
                    oil REAL,
                    water REAL,
                    gas REAL,
                    pip REAL,
                    m_temp REAL,
                    amps REAL,
                    tbgp REAL,
                    csgp REAL,
                    runtime REAL,
                    comments TEXT,
                    time TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO batteries (id, site_id, name)
                VALUES
                    (10, 1, 'Battery 6'),
                    (20, 1, 'Battery 13'),
                    (30, 2, 'Other Site Battery')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO wells (id, site_id, name, battery_id)
                VALUES
                    (1, 1, 'HARTZOG DRAW UNIT 4048', 10),
                    (2, 1, 'HARTZOG DRAW UNIT 4050', 20),
                    (3, 2, 'OTHER SITE WELL', 30)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO well_tests
                    (
                        id,
                        well_id,
                        oil,
                        water,
                        gas,
                        pip,
                        m_temp,
                        amps,
                        tbgp,
                        csgp,
                        runtime,
                        comments,
                        time
                    )
                VALUES
                    (10, 1, 25.5, 12.0, 30.0, NULL, NULL, NULL, NULL, NULL, 24.0, 'good', '2026-05-01 08:00:00'),
                    (11, 2, 19.5, 9.0, 20.0, NULL, NULL, NULL, NULL, NULL, 24.0, 'low oil', '2026-05-02 08:00:00'),
                    (12, 1, 40.0, 15.0, 35.0, NULL, NULL, NULL, NULL, NULL, 20.0, 'strong', '2026-06-05 08:00:00'),
                    (13, 1, 50.0, 20.0, 45.0, NULL, NULL, NULL, NULL, NULL, 20.0, 'outside range', '2026-06-06 08:00:00'),
                    (14, 3, 90.0, 1.0, 1.0, NULL, NULL, NULL, NULL, NULL, 24.0, 'other site', '2026-05-01 08:00:00')
                """
            )
        )
    return ReadingClient(engine)


def test_get_readings_for_date_filters_by_site_and_day():
    result = make_sqlite_client().get_readings_for_date(1, "lact", "2026-06-10")

    assert result["count"] == 1
    assert result["readings"][0] == {
        "entity_display_name": "LACT - LACT A",
        "entity_type": "LACT",
        "reading": 123.4,
        "temperature": 88.0,
        "bs_w": 0.1,
        "comments": "normal",
    }
    assert "reading_id" not in result["readings"][0]
    assert "time" not in result["readings"][0]


def test_linear_tank_readings_include_calculated_volume():
    result = make_tank_volume_client().get_readings_for_date(
        1, "linear_tank", "2026-06-10"
    )

    assert result["readings"] == [
        {
            "entity_display_name": "Tank - Linear A",
            "entity_type": "Tank",
            "level": 2.5,
            "comments": "level based",
            "bbl_foot": 50.0,
            "volume": 125.0,
                "water_volume": 125.0,
                "recoverable_oil_volume": 0.0,
            "content_type": "water",
        },
        {
            "entity_display_name": "Tank - Linear B",
            "entity_type": "Tank",
            "feet": 3,
            "inches": 6.0,
            "comments": "feet inches based",
            "bbl_foot": 40.0,
            "volume": 140.0,
                    "oil_volume": 140.0,
                    "recoverable_oil_volume": 100.0,
                "content_type": "oil",
        },
    ]


def test_mixed_tank_readings_include_calculated_oil_water_volumes():
    result = make_tank_volume_client().get_readings_for_date(
        1, "mixed_tank", "2026-06-10"
    )

    assert result["readings"] == [
        {
            "entity_display_name": "Tank - Mixed A",
            "entity_type": "Tank",
            "top_level_feet": 4,
            "top_level_inches": 6.0,
            "water_level_feet": 1,
            "water_level_inches": 6.0,
            "comments": "mixed complete",
            "bbl_foot": 100.0,
            "oil_volume": 300.0,
            "water_volume": 150.0,
                "total_volume": 450.0,
                "recoverable_oil_volume": 250.0,
                "content_type": "water-oil",
        },
        {
            "entity_display_name": "Tank - Mixed Missing",
            "entity_type": "Tank",
            "top_level_feet": 4,
            "top_level_inches": 6.0,
            "water_level_feet": 1,
            "water_level_inches": 6.0,
            "comments": "missing bbl foot",
            "volume_status": "missing_bbl_foot",
            "content_type": "water-oil",
        },
    ]


def test_all_tank_readings_include_volume_enriched_linear_and_mixed_groups():
    result = make_tank_volume_client().get_readings_for_date(1, "tank", "2026-06-10")

    linear_group = result["groups"][0]
    mixed_group = result["groups"][1]
    non_linear_group = result["groups"][2]
    assert linear_group["reading_type"] == "linear_tank"
    assert linear_group["readings"][0]["volume"] == 125.0
    assert mixed_group["reading_type"] == "mixed_tank"
    assert mixed_group["readings"][0]["oil_volume"] == 300.0
    assert non_linear_group["reading_type"] == "non_linear_tank"
    assert non_linear_group["readings"][0]["content_type"] == "water"


def test_search_tank_readings_filters_by_battery_relation_and_contains_oil():
    result = make_tank_volume_client().search_tank_readings(
        1,
        start_date="2026-06-10",
        end_date="2026-06-10",
        battery_name="Battery 6",
        contains="oil",
    )

    assert [row["tank_name"] for row in result["readings"]] == ["Linear B", "Mixed A"]
    assert result["readings"][0]["battery_name"] == "Battery 6"
    assert result["readings"][0]["oil_volume"] == 140.0


def test_search_tank_readings_uses_persisted_tank_contents():
    result = make_tank_volume_client().search_tank_readings(
        1,
        start_date="2026-06-10",
        end_date="2026-06-10",
        battery_name="6",
        contains="water",
    )

    names = [row["tank_name"] for row in result["readings"]]
    assert "Linear A" in names
    assert "Linear B" not in names
    assert "Non Linear A" in names
    linear = next(row for row in result["readings"] if row["tank_name"] == "Linear A")
    non_linear = next(
        row for row in result["readings"] if row["tank_name"] == "Non Linear A"
    )
    assert linear["water_volume"] == 125.0
    assert linear["contents"] == "water"
    assert non_linear["water_volume"] == 400.0
    assert non_linear["contents"] == "water"


def test_search_tank_readings_supports_computed_volume_filters():
    result = make_tank_volume_client().search_tank_readings(
        1,
        start_date="2026-06-10",
        end_date="2026-06-10",
        battery_name="6",
        contains="water",
        filters=[{"field": "water_volume", "operator": ">", "value": 130}],
    )

    assert [row["tank_name"] for row in result["readings"]] == [
        "Mixed A",
        "Non Linear A",
    ]


def test_search_tank_readings_filters_recoverable_oil_volume():
    result = make_tank_volume_client().search_tank_readings(
        1,
        start_date="2026-06-10",
        end_date="2026-06-10",
        filters=[{"field": "recoverable_oil_volume", "operator": ">", "value": 200}],
    )

    assert [row["tank_name"] for row in result["readings"]] == ["Mixed A"]
    assert result["readings"][0]["recoverable_oil_volume"] == 250.0


def test_search_tank_readings_accepts_display_prefixed_tank_name():
    result = make_tank_volume_client().search_tank_readings(
        1,
        start_date="2026-06-10",
        end_date="2026-06-10",
        tank_name="Tank Linear B",
    )

    assert result["count"] == 1
    assert result["readings"][0]["tank_name"] == "Linear B"
    assert result["readings"][0]["oil_volume"] == 140.0


def test_search_tank_readings_accepts_dash_display_prefixed_tank_name():
    result = make_tank_volume_client().search_tank_readings(
        1,
        start_date="2026-06-10",
        end_date="2026-06-10",
        tank_name="Tank - Mixed A",
    )

    assert result["count"] == 1
    assert result["readings"][0]["tank_name"] == "Mixed A"
    assert result["readings"][0]["oil_volume"] == 300.0


def test_search_equipment_readings_accepts_display_prefixed_tank_name():
    result = make_tank_volume_client().search_equipment_readings(
        1,
        "tank",
        start_date="2026-06-10",
        end_date="2026-06-10",
        entity_name="Tank - Mixed A",
    )

    assert result["count"] == 1
    assert result["readings"][0]["entity_name"] == "Mixed A"
    assert result["readings"][0]["oil_volume"] == 300.0


def test_compare_readings_between_dates_returns_entity_summary():
    result = make_sqlite_client().compare_readings_between_dates(
        1, "lact", "2026-06-10", "2026-06-11"
    )

    assert result["first_count"] == 1
    assert result["second_count"] == 1
    assert "first_readings" not in result
    assert "second_readings" not in result
    assert result["entity_summary"] == [
        {
            "entity_display_name": "LACT - LACT A",
            "first_count": 1,
            "second_count": 1,
            "status": "present_on_both_dates",
        }
    ]
    assert result["comparison_rows"][0]["entity_display_name"] == "LACT - LACT A"


def test_compare_mixed_tank_readings_returns_compact_changes():
    result = make_mixed_tank_client().compare_readings_between_dates(
        1, "mixed_tank", "2026-05-09", "2026-05-10"
    )

    assert "first_readings" not in result
    assert "second_readings" not in result
    assert result["comparison_rows"] == [
        {
            "entity_display_name": "Tank - 2-1 Float Over",
            "status": "compared",
            "changes": [
                {
                    "field": "top_level",
                    "label": "Top level",
                    "first_value": "8'0\"",
                    "second_value": "4'1\"",
                    "direction": "decrease",
                    "delta_inches": -47.0,
                    "delta": "3'11\"",
                },
            ],
        }
    ]


def test_missing_readings_excludes_disabled_and_other_sites():
    result = make_sqlite_client().missing_readings_for_date(1, "lact", "2026-06-10")

    assert result["missing_count"] == 1
    assert result["missing_entities"] == [
        {
            "entity_name": "LACT B",
            "entity_type": "LACT",
            "entity_display_name": "LACT - LACT B",
        }
    ]


def test_all_missing_readings_checks_all_supported_types_with_tables():
    result = make_sqlite_client().all_missing_readings_for_date(1, "2026-06-10")

    assert result["missing_count"] == 2
    assert [group["reading_type"] for group in result["groups"]] == ["lact", "flare"]
    assert result["groups"][0]["missing_entities"] == [
        {
            "entity_name": "LACT B",
            "entity_type": "LACT",
            "entity_display_name": "LACT - LACT B",
        }
    ]
    assert result["groups"][1]["missing_entities"] == [
        {
            "entity_name": "Flare A",
            "entity_type": "Flare",
            "entity_display_name": "Flare - Flare A",
        }
    ]


def test_missing_readings_respect_report_config_exclusions():
    client = make_sqlite_client_with_missing_exclusions()

    assert (
        client.missing_readings_for_date(1, "lact", "2026-06-10")["missing_count"]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "flare", "2026-06-10")["missing_count"]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "knock_out", "2026-06-10")[
            "missing_count"
        ]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "linear_tank", "2026-06-10")[
            "missing_count"
        ]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "mixed_tank", "2026-06-10")[
            "missing_count"
        ]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "non_linear_tank", "2026-06-10")[
            "missing_count"
        ]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "treater", "2026-06-10")[
            "missing_count"
        ]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "water_plant", "2026-06-10")[
            "missing_count"
        ]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "flow_meter", "2026-06-10")[
            "missing_count"
        ]
        == 0
    )
    assert (
        client.missing_readings_for_date(1, "pump", "2026-06-10")["missing_count"]
        == 0
    )
    assert client.all_missing_readings_for_date(1, "2026-06-10")["missing_count"] == 0


def test_search_well_tests_filters_date_range_site_and_oil_threshold():
    result = make_well_test_client().search_well_tests(
        1,
        "2026-05-01",
        "2026-06-05",
        min_oil=20,
    )

    assert result["count"] == 2
    assert result["filters"] == {"min_oil": 20}
    assert result["well_tests"] == [
        {
            "date": "2026-05-01",
            "entity_display_name": "Well - HARTZOG DRAW UNIT 4048",
            "entity_type": "Well",
            "oil": 25.5,
            "water": 12.0,
            "gas": 30.0,
            "runtime": 24.0,
            "comments": "good",
        },
        {
            "date": "2026-06-05",
            "entity_display_name": "Well - HARTZOG DRAW UNIT 4048",
            "entity_type": "Well",
            "oil": 40.0,
            "water": 15.0,
            "gas": 35.0,
            "runtime": 20.0,
            "comments": "strong",
        },
    ]


def test_analyze_well_tests_compares_latest_and_previous_per_well():
    result = make_well_test_client().analyze_well_tests(
        site_id=1,
        analysis_mode="latest_previous",
        group_by="well",
    )

    assert result["well_count"] == 1
    assert result["groups"] == [
        {
            "group_name": "HARTZOG DRAW UNIT 4048",
            "well_name": "HARTZOG DRAW UNIT 4048",
            "battery_name": "Battery 6",
            "latest_date": "2026-06-06",
            "previous_date": "2026-06-05",
            "latest_oil": 50.0,
            "previous_oil": 40.0,
            "oil_change": 10.0,
            "latest_water": 20.0,
            "previous_water": 15.0,
            "water_change": 5.0,
            "latest_gas": 45.0,
            "previous_gas": 35.0,
            "gas_change": 10.0,
            "latest_runtime": 20.0,
            "previous_runtime": 20.0,
            "runtime_change": 0.0,
        }
    ]


def test_analyze_well_tests_rolls_comparisons_up_by_battery_and_site():
    client = make_well_test_client()

    battery = client.analyze_well_tests(1, "latest_previous", "battery")
    site = client.analyze_well_tests(1, "latest_previous", "site")

    assert battery["groups"][0]["group_name"] == "Battery 6"
    assert battery["groups"][0]["oil_sum_change"] == 10.0
    assert site["groups"][0]["group_name"] == "Selected site"
    assert site["groups"][0]["well_count"] == 1
    assert site["groups"][0]["latest_oil_sum"] == 50.0


def test_analyze_well_tests_summarizes_range_by_battery_and_scopes_site():
    result = make_well_test_client().analyze_well_tests(
        site_id=1,
        analysis_mode="range_summary",
        group_by="battery",
        start_date="2026-05-01",
        end_date="2026-06-05",
    )

    groups = {item["group_name"]: item for item in result["groups"]}
    assert groups["Battery 6"]["test_count"] == 2
    assert groups["Battery 6"]["oil_sum"] == 65.5
    assert groups["Battery 6"]["oil_average"] == 32.75
    assert groups["Battery 13"]["test_count"] == 1
    assert "Other Site Battery" not in groups


def test_analyze_well_tests_compares_latest_three_tests_per_well():
    result = make_well_test_client().analyze_well_tests(
        site_id=1,
        analysis_mode="recent_tests",
        group_by="well",
        battery_name="Battery 6",
        test_count=3,
    )

    assert result["requested_test_count"] == 3
    assert result["well_count"] == 1
    tests = result["groups"][0]["tests"]
    assert [item["date"] for item in tests] == [
        "2026-05-01",
        "2026-06-05",
        "2026-06-06",
    ]
    assert [item["oil_change"] for item in tests] == [None, 14.5, 10.0]


def test_range_sequence_uses_only_preceding_test_inside_selected_range():
    result = make_well_test_client().analyze_well_tests(
        site_id=1,
        analysis_mode="range_sequence",
        group_by="well",
        start_date="2026-06-01",
        end_date="2026-06-30",
        battery_name="Battery 6",
    )

    tests = result["groups"][0]["tests"]
    assert [item["date"] for item in tests] == ["2026-06-05", "2026-06-06"]
    assert tests[0]["oil_change"] is None
    assert tests[1]["oil_change"] == 10.0


def test_range_sequence_keeps_well_with_only_one_selected_test():
    result = make_well_test_client().analyze_well_tests(
        site_id=1,
        analysis_mode="range_sequence",
        group_by="well",
        start_date="2026-05-01",
        end_date="2026-05-31",
        battery_name="Battery 13",
    )

    assert result["groups"][0]["well_name"] == "HARTZOG DRAW UNIT 4050"
    assert result["groups"][0]["tests"][0]["oil_change"] is None


def test_search_readings_filters_lact_range_and_numeric_field():
    result = make_sqlite_client().search_readings(
        1,
        "lact",
        "2026-06-10",
        "2026-06-11",
        filters=[{"field": "reading", "operator": ">", "value": 130}],
    )

    assert result["count"] == 1
    assert result["filters"] == {
        "field_filters": [{"field": "reading", "operator": ">", "value": 130}]
    }
    assert result["readings"] == [
        {
            "date": "2026-06-11",
            "entity_display_name": "LACT - LACT A",
            "entity_type": "LACT",
            "reading": 140.0,
            "temperature": 90.0,
            "bs_w": 0.1,
            "comments": "next day",
        }
    ]


def test_search_equipment_readings_filters_direct_battery_relation_and_fields():
    result = make_equipment_relation_client().search_equipment_readings(
        1,
        "flow_meter",
        start_date="2026-06-10",
        end_date="2026-06-10",
        battery_name="6",
        equipment_filters=[{"field": "type", "operator": "=", "value": "gas"}],
        reading_filters=[{"field": "total", "operator": ">", "value": 100}],
    )

    assert result["count"] == 1
    row = result["readings"][0]
    assert row["entity_name"] == "FM Battery 6 Gas"
    assert row["battery_name"] == "Battery 6"
    assert row["type"] == "gas"
    assert row["total"] == 150.0


def test_search_equipment_readings_numeric_battery_does_not_match_battery_16():
    result = make_equipment_relation_client().search_equipment_readings(
        1,
        "flow_meter",
        start_date="2026-06-10",
        end_date="2026-06-10",
        battery_name="6",
        reading_filters=[{"field": "total", "operator": ">", "value": 80}],
    )

    assert result["count"] == 2
    assert {row["battery_name"] for row in result["readings"]} == {"Battery 6"}
    assert {row["entity_name"] for row in result["readings"]} == {
        "FM Battery 6 Gas",
        "FM Battery 6 Water",
    }


def test_search_equipment_readings_filters_pump_by_water_plant_battery_relation():
    result = make_equipment_relation_client().search_equipment_readings(
        1,
        "pump",
        start_date="2026-06-10",
        end_date="2026-06-10",
        battery_name="Battery 6",
        equipment_filters=[{"field": "type", "operator": "=", "value": "water"}],
        reading_filters=[
            {"field": "discharge_pressure", "operator": ">=", "value": 100}
        ],
    )

    assert result["count"] == 1
    row = result["readings"][0]
    assert row["entity_name"] == "Water Pump 6"
    assert row["battery_name"] == "Battery 6"
    assert row["type"] == "water"
    assert row["discharge_pressure"] == 120.0


def test_list_equipment_lists_tanks_by_battery_without_reading_date():
    result = make_equipment_relation_client().list_equipment(
        1,
        "tank",
        battery_name="6",
    )

    assert result["count"] == 2
    assert {row["tank_name"] for row in result["entities"]} == {
        "Tank 6-1 Oil",
        "Tank 6-2 Water",
    }
    assert {row["battery_name"] for row in result["entities"]} == {"Battery 6"}


def test_list_equipment_filters_pumps_by_indirect_battery_and_type():
    result = make_equipment_relation_client().list_equipment(
        1,
        "pump",
        battery_name="Battery 6",
        equipment_filters=[{"field": "type", "operator": "=", "value": "water"}],
    )

    assert result["count"] == 1
    assert result["entities"][0]["entity_name"] == "Water Pump 6"
    assert result["entities"][0]["battery_name"] == "Battery 6"


def test_resolve_reading_entity_finds_unique_partial_name():
    result = make_entity_resolution_client().resolve_reading_entity(
        1, "Tract 5 vent", "2026-06-01"
    )

    assert result["count"] == 1
    assert result["candidates"][0]["reading_type"] == "flow_meter"
    assert result["candidates"][0]["entity_display_name"] == "Flow Meter - TRACT 5 Vent"
    assert result["candidates"][0]["has_reading"] is True


def test_resolve_reading_entity_reports_ambiguous_object_name():
    result = make_entity_resolution_client().resolve_reading_entity(
        1, "Battery 2 Vent", "2026-06-01"
    )

    assert result["count"] == 2
    assert {
        candidate["entity_display_name"] for candidate in result["candidates"]
    } == {"Flow Meter - Battery 2 Vent", "Flare - Battery 2 Vent"}


def test_get_reading_for_entity_uses_type_constraint_for_ambiguous_name():
    result = make_entity_resolution_client().get_reading_for_entity(
        1, "Battery 2 Vent", "2026-06-01", "flow_meter"
    )

    assert result["reading_type"] == "flow_meter"
    assert result["count"] == 1
    assert result["readings"] == [
        {
            "entity_display_name": "Flow Meter - Battery 2 Vent",
            "entity_type": "Flow Meter",
            "total": 0.0,
            "flow": 1.5,
            "odometer": 100.0,
            "comments": "normal",
        }
    ]


def test_get_reading_for_entity_returns_clarification_candidates():
    result = make_entity_resolution_client().get_reading_for_entity(
        1, "Battery 2 Vent", "2026-06-01"
    )

    assert result["needs_clarification"] is True
    assert result["readings"] == []
    assert {
        candidate["entity_display_name"] for candidate in result["candidates"]
    } == {"Flow Meter - Battery 2 Vent", "Flare - Battery 2 Vent"}


def test_resolve_reading_entity_maps_tank_type_to_specific_reading_type():
    result = make_entity_resolution_client().resolve_reading_entity(
        1, "11-1-1", "2026-06-01", "tank"
    )

    assert result["count"] == 1
    assert result["candidates"][0]["reading_type"] == "mixed_tank"
    assert result["candidates"][0]["entity_display_name"] == "Tank - 11-1-1 Oil"
    assert result["candidates"][0]["has_reading"] is True


def test_resolve_reading_entity_strips_tank_display_prefix():
    result = make_entity_resolution_client().resolve_reading_entity(
        1, "Tank 10-1 Oil", "2026-06-01", "tank"
    )

    assert result["count"] == 1
    assert result["candidates"][0]["reading_type"] == "linear_tank"
    assert result["candidates"][0]["entity_display_name"] == "Tank - 10-1 Oil"


def test_resolve_reading_entity_strips_tank_dash_display_prefix():
    result = make_entity_resolution_client().resolve_reading_entity(
        1, "Tank - 10-1 Oil", "2026-06-01", "tank"
    )

    assert result["count"] == 1
    assert result["candidates"][0]["reading_type"] == "linear_tank"
    assert result["candidates"][0]["entity_display_name"] == "Tank - 10-1 Oil"


def test_get_reading_for_entity_accepts_display_prefixed_tank_name():
    result = make_entity_resolution_client().get_reading_for_entity(
        1, "Tank 11-1-1 Oil", "2026-06-01", "tank"
    )

    assert result["reading_type"] == "mixed_tank"
    assert result["count"] == 1
    assert result["readings"][0]["entity_display_name"] == "Tank - 11-1-1 Oil"
    assert result["readings"][0]["oil_volume"] == 700.0
    assert result["readings"][0]["water_volume"] == 100.0


def test_resolve_reading_entity_is_scoped_to_site():
    result = make_entity_resolution_client().get_reading_for_entity(
        1, "OtherSiteBattery2Vent", "2026-06-01", "flow_meter"
    )

    assert result["count"] == 0
    assert result["readings"] == []


def test_search_readings_rejects_unsupported_filter_field():
    with pytest.raises(ReadingClientError, match="unsupported field"):
        make_sqlite_client().search_readings(
            1,
            "lact",
            "2026-06-10",
            "2026-06-11",
            filters=[{"field": "comments", "operator": "=", "value": 1}],
        )


def test_search_well_tests_rejects_reversed_date_range():
    with pytest.raises(ReadingClientError, match="end_date must be"):
        make_well_test_client().search_well_tests(
            1,
            "2026-06-05",
            "2026-05-01",
        )


def test_unknown_reading_type_is_rejected():
    with pytest.raises(ReadingClientError, match="Unsupported reading type"):
        make_sqlite_client().get_readings_for_date(1, "unknown", "2026-06-10")


def test_unavailable_client_lists_types_but_rejects_queries():
    client = UnavailableReadingClient("DB settings missing")

    assert any(item["reading_type"] == "lact" for item in client.list_reading_types())
    with pytest.raises(ReadingClientError, match="DB settings missing"):
        client.get_readings_for_date(1, "lact", "2026-06-10")
