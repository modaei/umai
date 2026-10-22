# Ometrics Operational Database Schema

This document is the curated schema context for a future read-only SQL tool.
It is intentionally limited to operational tables that may be queried by UMAI.
It is not a full database dump, and it must not be treated as permission to
query tables outside the allowlist below.

## SQL Safety Rules

- Only read-only `SELECT` queries may use this schema.
- Every query must be scoped to the selected site.
- Use the selected site through `site_id` filters or joins to a site-scoped parent table.
- Prefer display names over database IDs in final answers.
- Use ISO `YYYY-MM-DD` dates inside SQL/tool arguments.
- Format dates as `MM/DD/YYYY` in final user-facing answers.
- Do not query authentication, user, session, token, password, API key, or billing tables.
- Do not expose raw internal IDs unless they are required to explain a join or debug a data issue.

## SQL Dialect

Ometrics uses MySQL/MariaDB syntax for operational SQL. Do not use PostgreSQL
syntax.

- Use `LOWER(column) LIKE '%text%'` for case-insensitive matching.
- Do not use `ILIKE`.
- Use `DATE_FORMAT(time_column, '%Y-%m')` for monthly grouping.
- Do not use `DATE_TRUNC`.
- Use normal string date literals such as `'2026-05-01'`.
- Do not use `DATE '2026-05-01'` or casts such as `::date`.
- Use `DATE(timestamp_column)` only when matching a datetime to a date.

## Allowed Tables

The future SQL tool may only use these tables:

- `alarm_events`
- `alarm_logs`
- `batteries`
- `chart_notes`
- `data_points`
- `data_point_data`
- `flares`
- `flare_readings`
- `flow_meters`
- `flow_meter_readings`
- `general_notes`
- `knock_outs`
- `knock_out_readings`
- `lacts`
- `lact_readings`
- `linear_tank_readings`
- `mixed_tank_readings`
- `non_linear_tank_readings`
- `non_linear_tank_volume_mappings`
- `non_linear_tank_volume_mapping_details`
- `onrr_codes`
- `pumps`
- `pump_readings`
- `rod_pump_cards`
- `rod_pump_monitoring_data`
- `rod_pump_monitoring_data_cards`
- `rod_pump_notes`
- `run_tickets`
- `silent_profiles`
- `tanks`
- `treaters`
- `treater_readings`
- `water_draws`
- `water_plants`
- `water_plant_readings`
- `wells`
- `well_documents`
- `well_fluids`
- `well_histories`
- `well_injections`
- `well_shutdowns`
- `well_tests`
- `work_orders`
- `work_order_notes`

## Site-Scoped Parent Tables

These tables normally carry `site_id` directly and can be filtered with:

```sql
WHERE <table>.site_id = :site_id
```

- `batteries`
- `chart_notes`
- `data_points`
- `flares`
- `flow_meters`
- `general_notes`
- `knock_outs`
- `lacts`
- `pumps`
- `silent_profiles`
- `tanks`
- `treaters`
- `water_plants`
- `wells`
- `work_orders`

## Child Tables And Site Joins

Most reading/event tables do not need to expose `site_id` directly if they join
through their parent equipment table. Use these joins to enforce site scope.

### Alarms And Data Points

- `alarm_events.data_point_id -> data_points.id`
- `alarm_logs.data_point_id -> data_points.id`
- `data_point_data.data_point_id -> data_points.id`

Site filter:

```sql
JOIN data_points ON <child>.data_point_id = data_points.id
WHERE data_points.site_id = :site_id
```

Useful display fields are typically on `data_points`, such as facility, device,
and data point names.

Common alarm columns:

- `alarm_logs.data_point_id`
- `alarm_logs.event`
- `alarm_logs.comments`
- `alarm_logs.unix_timestamp`
- `alarm_logs.created_at`
- `data_points.facility_name`
- `data_points.device_name`
- `data_points.data_point_name`
- `data_points.tag_name`

For alarm-log date filtering, prefer `alarm_logs.created_at` unless the user
specifically needs the event timestamp embedded in comments. For "which data
points generated the most alarm logs", group by `data_points.data_point_name`
or by a combined display label using `facility_name`, `device_name`, and
`data_point_name`. For "which batteries had the most alarm logs", group by
`data_points.facility_name` when it contains battery names.

### Flares

- `flare_readings.flare_id -> flares.id`

Site filter:

```sql
JOIN flares ON flare_readings.flare_id = flares.id
WHERE flares.site_id = :site_id
```

### Flow Meters

- `flow_meter_readings.flow_meter_id -> flow_meters.id`

Site filter:

```sql
JOIN flow_meters ON flow_meter_readings.flow_meter_id = flow_meters.id
WHERE flow_meters.site_id = :site_id
```

Common flow meter columns:

- `flow_meters.name`
- `flow_meters.type`
- `flow_meters.battery_id`
- `flow_meter_readings.time`
- `flow_meter_readings.total`
- `flow_meter_readings.flow`
- `flow_meter_readings.odometer`

Current flow meter types commonly include `gas` and `water`. Do not assume an
`oil` flow meter type exists. If the user asks for oil flow meters, query
`flow_meters.type` and `flow_meters.name` first or explain that no oil type is
configured when none match.

### Knock Outs

- `knock_out_readings.knock_out_id -> knock_outs.id`

Site filter:

```sql
JOIN knock_outs ON knock_out_readings.knock_out_id = knock_outs.id
WHERE knock_outs.site_id = :site_id
```

### LACTs

- `lact_readings.lact_id -> lacts.id`

Site filter:

```sql
JOIN lacts ON lact_readings.lact_id = lacts.id
WHERE lacts.site_id = :site_id
```

### Pumps And Rod Pump Data

- `pump_readings.pump_id -> pumps.id`
- `rod_pump_monitoring_data.well_id -> wells.id`
- `rod_pump_notes.well_id -> wells.id`
- `rod_pump_monitoring_data_cards.rod_pump_monitoring_data_id -> rod_pump_monitoring_data.id`
- `rod_pump_monitoring_data_cards.rod_pump_card_id -> rod_pump_cards.id`

Site filter for ordinary pump records:

```sql
JOIN pumps ON <child>.pump_id = pumps.id
WHERE pumps.site_id = :site_id
```

Site filter for rod-pump monitoring data and notes:

```sql
JOIN wells ON <child>.well_id = wells.id
WHERE wells.site_id = :site_id
```

Site filter for monitoring-data cards:

```sql
JOIN rod_pump_monitoring_data
  ON rod_pump_monitoring_data_cards.rod_pump_monitoring_data_id = rod_pump_monitoring_data.id
JOIN wells ON rod_pump_monitoring_data.well_id = wells.id
WHERE wells.site_id = :site_id
```

`rod_pump_cards` has no direct well key. Scope it through
`rod_pump_monitoring_data_cards` and `rod_pump_monitoring_data`.

### Tanks

`tanks.contents` is the authoritative fluid classification. Its values are `oil`,
`water`, and `water-oil`. Mixed Water/Oil tank types always use `water-oil`;
linear and non-linear tank types use either `oil` or `water`.
`tanks.unusable_height` stores the unusable bottom height in decimal feet.
Recoverable oil excludes oil below this height, while water below it has no effect.

- `linear_tank_readings.tank_id -> tanks.id`
- `mixed_tank_readings.tank_id -> tanks.id`
- `non_linear_tank_readings.tank_id -> tanks.id`
- `non_linear_tank_volume_mappings.tank_id -> tanks.id`
- `run_tickets.tank_id -> tanks.id`
- `water_draws.tank_id -> tanks.id`

Site filter:

```sql
JOIN tanks ON <child>.tank_id = tanks.id
WHERE tanks.site_id = :site_id
```

Non-linear tank volume mapping details are scoped through their mapping:

```sql
JOIN non_linear_tank_volume_mappings
  ON non_linear_tank_volume_mapping_details.non_linear_tank_volume_mapping_id =
     non_linear_tank_volume_mappings.id
JOIN tanks ON non_linear_tank_volume_mappings.tank_id = tanks.id
WHERE tanks.site_id = :site_id
```

### Treaters

- `treater_readings.treater_id -> treaters.id`

Site filter:

```sql
JOIN treaters ON treater_readings.treater_id = treaters.id
WHERE treaters.site_id = :site_id
```

### Water Plants

- `water_plant_readings.water_plant_id -> water_plants.id`

Site filter:

```sql
JOIN water_plants ON water_plant_readings.water_plant_id = water_plants.id
WHERE water_plants.site_id = :site_id
```

### Wells

- `well_documents.well_id -> wells.id`
- `well_fluids.well_id -> wells.id`
- `well_histories.well_id -> wells.id`
- `well_injections.well_id -> wells.id`
- `well_shutdowns.well_id -> wells.id`
- `well_tests.well_id -> wells.id`

Site filter:

```sql
JOIN wells ON <child>.well_id = wells.id
WHERE wells.site_id = :site_id
```

Use `wells.name` as the display name. For Hartzog numeric references such as
`4048`, match by suffix or by `HARTZOG DRAW UNIT 4048` where appropriate.

ONRR code status:

- `wells.onrr_code_id -> onrr_codes.id`
- Active wells require `onrr_codes.active_well = 1`.
- Producing wells require `onrr_codes.active_well = 1` and `onrr_codes.injection_well = 0`.
- Injection wells require `onrr_codes.injection_well = 1`.
- Include `onrr_codes.description` when explaining why a well is active,
  producing, injection, inactive, or non-producing.
- If the question asks for historical status, use `well_histories` records where
  `property = 'onrr_code_id'` to determine the effective ONRR code at that time.

Common shutdown columns:

- `well_shutdowns.start`
- `well_shutdowns.end`
- `well_shutdowns.downtime_code`
- `well_shutdowns.comments`

For "total shutdown hours", compute the overlap between
`well_shutdowns.start`/`well_shutdowns.end` and the requested
date range, then group by `wells.name` or `well_shutdowns.downtime_code`.

For shutdown reason aggregations, group by `well_shutdowns.downtime_code`.
There is no `shutdown_codes` table in the curated SQL allowlist.

Example:

```sql
SELECT ws.downtime_code, SUM(ws.hours) AS total_hours
FROM well_shutdowns ws
JOIN wells w ON w.id = ws.well_id
WHERE w.site_id = :site_id
  AND ws.date >= '2026-05-01'
  AND ws.date < '2026-06-01'
GROUP BY ws.downtime_code
ORDER BY total_hours DESC
LIMIT 10
```

### Well Tests

Use `well_tests.time` for date filtering and `well_tests.oil`, `water`, `gas`,
and `runtime` for numeric aggregates. Use `wells.name` as the well display
name. Join through `wells` for site scoping, then optionally left join
`batteries` through `wells.battery_id`.

Only use well-test aggregates when the user explicitly asks about well tests.
For specific well or well-group production/injection contribution, use the
allocation report tools instead. Well tests are samples; allocation reports
attribute measured production and injection back to wells.

For well-attribute grouping, use case-insensitive comparisons. Rod wells mean
`LOWER(wells.pump_type) = LOWER('ROD')`; ESP/JET/flowing wells use
`wells.pump_type`; TA/POW/WIW/WIWSI wells use `onrr_codes.name` through
`wells.onrr_code_id`. Allocation questions should use the allocation tool when
available instead of SQL over `well_tests`.

Average oil from well tests by battery:

```sql
SELECT COALESCE(b.name, 'No Battery') AS battery,
       AVG(wt.oil) AS avg_oil,
       COUNT(*) AS test_count
FROM well_tests wt
JOIN wells w ON w.id = wt.well_id
LEFT JOIN batteries b ON b.id = w.battery_id
WHERE w.site_id = :site_id
  AND wt.time >= '2026-01-01'
  AND wt.time < '2027-01-01'
GROUP BY COALESCE(b.name, 'No Battery')
ORDER BY avg_oil DESC
LIMIT 100
```

Same-day well tests and shutdowns:

```sql
SELECT DISTINCT w.name AS well_name, ws.date AS event_date
FROM wells w
JOIN well_shutdowns ws ON ws.well_id = w.id
JOIN well_tests wt ON wt.well_id = w.id AND DATE(wt.time) = ws.date
WHERE w.site_id = :site_id
  AND ws.date >= '2026-06-01'
  AND ws.date < '2026-07-01'
ORDER BY w.name, ws.date
LIMIT 100
```

### Tank Volume Aggregates

For mixed tank oil volume, calculate oil height as top level minus water level
and multiply by `tanks.bbl_foot`. Use `mixed_tank_readings.time` for dates.
For recoverable mixed-tank oil, calculate top level minus the greater of water
level and `tanks.unusable_height`, clamp the height at zero, and multiply by
`tanks.bbl_foot`.

Computed fields returned by UMAI reading tools are not database columns. Do not
write SQL against `oil_volume`, `water_volume`, `total_volume`, or
`recoverable_oil_volume`, and do not
use a unified `tank_readings` table. For SQL, use the specific reading tables
such as `mixed_tank_readings`, `linear_tank_readings`, and
`non_linear_tank_readings`.

Monthly average mixed-tank oil volume:

```sql
SELECT DATE_FORMAT(m.time, '%Y-%m') AS month,
       AVG(
         (
           (COALESCE(m.top_level_feet, 0) + COALESCE(m.top_level_inches, 0) / 12.0)
           - (COALESCE(m.water_level_feet, 0) + COALESCE(m.water_level_inches, 0) / 12.0)
         ) * t.bbl_foot
       ) AS avg_oil_volume
FROM mixed_tank_readings m
JOIN tanks t ON t.id = m.tank_id
WHERE t.site_id = :site_id
  AND m.time >= '2026-01-01'
  AND m.time < '2027-01-01'
GROUP BY DATE_FORMAT(m.time, '%Y-%m')
ORDER BY month
LIMIT 12
```

Average daily mixed-tank oil volume for a date range:

```sql
SELECT ROUND(AVG(d.daily_oil_bbl), 2) AS average_daily_oil_bbl
FROM (
  SELECT DATE(m.time) AS reading_date,
         SUM(
           (
             (COALESCE(m.top_level_feet, 0) + COALESCE(m.top_level_inches, 0) / 12.0)
             - (COALESCE(m.water_level_feet, 0) + COALESCE(m.water_level_inches, 0) / 12.0)
           ) * t.bbl_foot
         ) AS daily_oil_bbl
  FROM mixed_tank_readings m
  JOIN tanks t ON t.id = m.tank_id
  WHERE t.site_id = :site_id
    AND m.time >= '2026-06-01'
    AND m.time < '2026-06-21'
    AND t.type = 'mixed-water-oil'
  GROUP BY DATE(m.time)
) d
LIMIT 100
```

### Work Orders

- `work_order_notes.work_order_id -> work_orders.id`

Site filter:

```sql
JOIN work_orders ON work_order_notes.work_order_id = work_orders.id
WHERE work_orders.site_id = :site_id
```

## Common Time Columns

The exact available columns should still be validated by the SQL tool before
execution, but the current operational conventions are:

- Daily notes often use `date`.
- Shutdowns use `start` and `end`.
- Reading and work-order records often use `time`.
- Alarm events often use `happened_on`.
- Alarm logs commonly use `created_at`; they do not normally have
  `alarm_date`, `alarm_time`, or `happened_on` columns.
- Current data-point values are stored in `data_point_data.data` as JSON, with
  `data_point_data.last_update` as an epoch timestamp.
- Well history uses `changed_at`.
- Standard Laravel tables may have `created_at` and `updated_at`.

## Query Preference

Use existing structured UMAI tools before SQL for standard report, reading,
shutdown, timeline, capability, and RAG questions. The future SQL tool should be
reserved for ad hoc analysis, cross-table checks, unusual aggregations, and
debugging questions that cannot be answered well by existing tools.
