"""Site timezones for moments stored by umetriqs.

umetriqs stores DATETIME moments (shutdowns, run tickets) in its storage timezone, which
UMAI reads as the TIMEZONE setting. Each site has its own IANA timezone in sites.timezone,
and its production days run midnight to midnight on that clock. Clients do their day
arithmetic on the site's clock and convert only at the database edge.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Timezone of sites without one, matching umetriqs' Site::DEFAULT_TIMEZONE.
DEFAULT_SITE_TIMEZONE = "America/Denver"


def lookup_site_zone(engine: Engine, site_id: int, storage_zone: ZoneInfo) -> ZoneInfo:
    """Return the site's timezone, or the default site timezone when it has none.

    Databases without a sites.timezone column (older schemas, test fixtures) keep the
    storage timezone, so their times are used unchanged.
    """
    try:
        with engine.connect() as connection:
            name = connection.execute(
                text("SELECT timezone FROM sites WHERE id = :site_id"), {"site_id": site_id}
            ).scalar()
        return ZoneInfo(name or DEFAULT_SITE_TIMEZONE)
    except SQLAlchemyError:
        return storage_zone


def site_to_storage(site_time: datetime, site_zone: ZoneInfo, storage_zone: ZoneInfo) -> datetime:
    """Convert a naive site wall-clock time to a naive storage-timezone time."""
    return site_time.replace(tzinfo=site_zone).astimezone(storage_zone).replace(tzinfo=None)


def storage_to_site(stored, site_zone: ZoneInfo, storage_zone: ZoneInfo) -> datetime | None:
    """Convert a stored naive moment (datetime or ISO string) to a naive site time."""
    if stored is None or stored == "":
        return None
    if not isinstance(stored, datetime):
        stored = datetime.fromisoformat(str(stored))
    return stored.replace(tzinfo=storage_zone).astimezone(site_zone).replace(tzinfo=None)
