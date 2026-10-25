from datetime import date, timedelta


def build_dashboard_briefing_prompt(current_date: date) -> str:
    """Build the shared briefing instructions for the seven complete days before today."""
    start = (current_date - timedelta(days=7)).isoformat()
    end = (current_date - timedelta(days=1)).isoformat()
    return (
        f"Use tools for a factual current-site briefing for {start} through {end} (inclusive). "
        'Compare oil production and sales with the preceding 7 days. Start: '
        '"Oil production up/down X.XX% and sales up/down Y.YY%." '
        'Omit comparison-period wording in the answer. Use actual changes. Include ONLY shutdowns whose '
        'start OR end falls inside the given window; '
        'exclude older ongoing shutdowns and overlap/open counts. Add significant operational notes. '
        'Use short well identifiers; avoid ambiguity. Omit site name, all dates/date ranges, totals, '
        'baseline values and headings. Include recorded reasons only; omit missing-cause commentary. '
        'Do not invent causes. Plain text, at most 65 words.'
    )
