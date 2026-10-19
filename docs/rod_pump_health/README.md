# Rod pump health analyzer

Parameter-trend and conclusion reviews are temporarily disabled by
`EXTENDED_HEALTH_ANALYSIS_ENABLED = False` in `src/omai/api/app.py`. The health endpoint runs only
the dynograph review and returns `status: disabled` for both optional stages, without invoking
their analyzers or LLMs. UMetriqs keeps existing optional results in storage and hides their dialog
sections via `extendedHealthAnalysisEnabled` in the health page. To restore the feature, re-enable
both flags and restart OMAI/rebuild the frontend. The implementation below is retained for restoration.

`POST /rod-pump-health-analysis` accepts a well database ID and an optional recent-card reference:

```json
{"well_id": 23}
```

UMetriqs validates the current-site scope, calls this endpoint synchronously, and saves the returned
analysis in `rod_pump_health_analyses`. OMAI resolves the well's site from the operational database
and checks the `rod_pump_monitoring` configuration before reading its archived cards.
The endpoint runs dynograph analysis first, followed by monthly parameter analysis and a final
conclusion for the same well.
These use separate LLM calls and prompts; parameter analysis receives its own time-series graphs,
with no dynographs or templates.
There is no separate parameter-trend endpoint.

## Data preparation and analysis

OMAI reads retained current cards marked `daily_baseline` from the archive database. It uses actual
capture timestamps, deduplicates repeated captures, and selects the lower median card for each day.
The window covers fourteen calendar days ending on the latest available daily card. Event-only and
pending cards are excluded. These daily representatives are not known healthy cards.

OMAI submits the selected surface cards to Omreports' existing `rod_pump_downhole_cards` report API
in one batch. Archive IDs are carried through the existing `card_id` field to match results to their
source cards. Omreports requires no changes. Settings snapshots are not captured; pump curves use
current controller settings. Surface analysis can continue if pump calculation fails.

OMAI computes physical features and Theil–Sen trends, renders field cards, pairwise normalized
field/template comparisons, and historical overlays, then sends images and measured JSON to its
configured vision-capable model. The prompt asks it to interpret recent cards against supplied
templates and explain historical trends using card data and rod-pump knowledge. It returns up to two
concise diagnoses with one reasoning paragraph each. OMAI preserves returned findings without
applying evidence-reference or match-strength filters. A successful empty review is healthy;
provider or review failures remain unavailable.
The analyzer does not write database records or restore the legacy score/episode pipeline.

OMAI computes the input signature and reads the latest saved analysis from the operational database.
An unchanged successful analysis returns `unchanged: true` without another LLM request. Downhole
calculation still runs before this comparison. UMetriqs verifies that its saved signature matches
before clearing the refresh indicator. New results include prepared `input_cards`, measurements,
findings, window timestamps, and the signature for UMetriqs to persist.

Monthly parameter analysis still runs when the dynograph review is unchanged. It retrieves the last
30 days of configured controller telemetry through `MONITORING_DATA_API_URL`, up to the latest
completed hour. The six parameters are Pump Fillage, Max Load LS, Min Load LS, Yesterday Strokes
/Min, Yesterday Percent, and Yesterday Cycles. Each available parameter is rendered as a time-series
graph using up to 720 monitoring-service samples, retaining null gaps. Numeric sample arrays and
daily median/min/max summaries are not sent to the LLM. Configured points are included even if their
alarm evaluation is disabled. Missing history is reported as insufficient data rather than healthy.

The response includes a `parameter_trends` object naming parameters with unusual overall trends,
each with a short trend title (such as "Sustained decline"), followed by one combined SAM1 oilfield
interpretation. Titles and interpretation omit dates, measured values and detailed episodes.
Its input signature, token usage and cache are independent of
the dynograph review. UMetriqs stores it under `findings.parameter_trends` without a migration and
shows it in the dialog's Trends section. If either review fails, the other can still be saved;
previous successful results for the failed review are retained.

The final conclusion uses a third, text-only LLM call. It receives the retained historical surface
and downhole diagnoses (excluding low-confidence findings and latest-card-only diagnoses), plus
the parameter-trend titles and interpretation. Cached card reviews are read from the saved analysis
so an unchanged dynograph response still supplies the historical diagnoses. The conclusion has
its own input signature and token usage, and returns one title and one paragraph. Incomplete source
reviews produce a qualified partial conclusion; unavailable evidence never implies healthy operation.
UMetriqs stores this under `findings.conclusion` and displays a collapsed Conclusion content box only
in the Historical analysis dialog, preserving the previous conclusion if synthesis fails.

Versions remain part of the API and cache signature, not separate database columns. Update the
corresponding constants in both services when analyzer or template behavior changes.
`reference-guide.pdf` is the supplied qualitative reference, not ground truth or instructions.

## Configuration

OMAI requires SELECT access to `wells`, `sites`, `data_points`, `report_configurations`, and
`rod_pump_health_analyses` in
the operational database, and to `rod_pump_card_archive` in the archive database. Its database
settings must identify the same operational and archive databases used by UMetriqs.

Operational connection settings remain `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME`.
Configure the archive connection with `DB_ARCHIVE_HOST`, `DB_ARCHIVE_PORT`, `DB_ARCHIVE_USERNAME`,
`DB_ARCHIVE_PASSWORD`, and `DB_ARCHIVE_DATABASE`. Unset archive options fall back to the operational
connection. `DB_ARCHIVE_USER` is also accepted when `DB_ARCHIVE_USERNAME` is not set. Set `TIMEZONE`
to UMetriqs' application timezone so both services use the same daily boundaries.

`OMREPORTS_API_URL` identifies the existing report endpoint. OMAI allows up to 180 seconds for its
calculation request. UMetriqs allows 720 seconds for the combined OMAI request; browser/controller
refresh limits are 780 seconds. Reverse-proxy and PHP-FPM timeouts must accommodate those limits.
No queue or worker is involved.

Install the OMAI requirements, including Pillow. Live interpretation requires configured LLM
credentials and a model supporting image input and structured output. Provider failures return a
review failure; UMetriqs retains the previous successful analysis.

## Validation

```sh
pytest tests/test_rod_pump_health_analysis.py tests/test_rod_pump_health_reader.py
pytest tests/test_rod_pump_parameter_trends.py
pytest tests/test_rod_pump_health_conclusion.py
```

Tests cover well/site scope, daily selection, unit conversion, existing report responses, missing
pump curves, changed-source invalidation, and unchanged-result caching. The request rejects uploaded
card arrays; OMAI always fetches source data itself. Trend estimates require at least three archived
cards with measured values; missing days are never interpolated.

## Failure diagnostics

Failed reviews now return `review_error` and log a warning with the well ID in OMAI's server output.
Diagnostics distinguish LLM/provider errors from unusable-card coverage. Provider messages are
bounded and redact authentication tokens and image blobs. OMAI preserves the model's returned
findings instead of dropping them for evidence-reference or match-strength checks.

UMetriqs logs the reason and includes it in the health command's per-well error and saved refresh error.
Previous successful results remain intact. Deploy both services and restart the OMAI process before
rerunning `php artisan rod-pump-health:analyze --site-id=...` to see these expanded diagnostics.

## Review guidance

The LLM reviews Lufkin SAM1 controller data from a unit without a VFD. It compares recent cards with
same-side templates and reviews archived cards for patterns and trends. Its prompt requests a
successful empty review when no problem is identified and asks it to explain the oil-field meaning
of a historical trend rather than only repeating measurements.

## Review behavior

The LLM returns up to two concise diagnoses and one reasoning paragraph for each. UMetriqs shows the
returned findings without evidence-based filtering. SAM1 pumping-state selection remains ten
minutes before and five minutes after the selected current card. A new analyzer version invalidates
cached reviews so older findings are replaced on the next analysis.

UMetriqs no longer rejects responses for an analyzer/template version mismatch. OMAI still includes
its versions in input signatures so the revised prompt invalidates cached earlier interpretations.
