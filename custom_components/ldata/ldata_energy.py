"""Hourly energy ledger for LDATA (ResidentialBreakerPanel) breakers.

An LDATA hub does not report a lifetime energy counter in normal operation.
The breaker ``energyConsumption``/``energyConsumption2`` fields, both in the
WebSocket burst the cloud sends every hour at HH:00:29 and in the REST panel
listing, carry the kWh each leg used in the hour that just ended. Now and then
(most likely after the hub reconnects to the cloud) the same fields carry the
breaker's lifetime counter instead, with 0.1 kWh resolution.

This ledger turns those values into a lifetime total and daily totals per
breaker and leg:

- A value above what the breaker can physically use in one hour is a lifetime
  counter. A WebSocket value is an hourly delta only inside the burst window
  around HH:00:29; outside it, a multiple of 0.1 kWh is a lifetime counter
  and anything else (a stale delta) is ignored. A listing value is an hourly
  delta, except a multiple of 0.1 kWh above 0.1 kWh, which may be a small
  lifetime counter and is ignored.
- Each delta is applied once per (breaker, leg, hour): the hour it covers is
  recorded, and a value for that hour or an earlier one is a duplicate. Only
  a non-zero burst value replaces a zero booked for the same hour. A
  lifetime counter not newer than the last one is ignored.
- The listing keeps its last value while the hub is offline, so a listing
  value within rounding of the last booked delta (also a zero after a
  booked zero) is not booked: it may be that hour's value again. The hour
  is left to the next lifetime counter.
- Each leg tracks ``complete_until``, the last hour up to which the ledger
  has all of the leg's energy. It advances over hours booked one after
  another, and when a lifetime counter matches the ledger or becomes the
  reference. A missed hour stops it, also when later hours are booked.
- A lifetime counter never lowers the ledger. The first one only records the
  offset between the cloud counter and the ledger. A later one that is more
  than 0.1 kWh above the expected counter adds the difference (energy from
  missed hours, e.g. while Home Assistant was down), but only when
  ``complete_until`` is before the hour before the current one, or when the
  difference is more than the breaker can have used in the current hour so
  far; otherwise the difference is the current hour's energy so far, which
  that hour's burst will bring. The added difference also covers the current hour up to
  now, so that hour is claimed (its burst is not booked) while
  ``complete_until`` stays at the hour before it: the next counter adds the
  rest of the claimed hour.
- Daily totals are bucketed by the local date of the hour the energy was used
  in, so the 00:00:29 burst counts for the previous day.

The ledger is plain data (see ``as_dict``/``load``) so the coordinator can
persist it across restarts. Once the coordinator persists it (``mark_saved``),
``lifetime`` and ``daily`` report the totals as of the last save, so a sensor
never counts energy that a crash before the next save would take out of the
ledger again.
"""

from __future__ import annotations

import datetime
import logging
import math
import threading
from typing import Any

from homeassistant.util import dt as dt_util

from .const import LOGGER_NAME

_LOGGER = logging.getLogger(LOGGER_NAME)

HOUR_SECS = 3600

# Nominal leg-to-neutral voltage. LDATA panels are North American split-phase
# (120/240 V); the per-leg energy fields are each leg's share.
NOMINAL_LEG_VOLTAGE = 120.0
# Headroom over rating x voltage x 1 h: covers high line voltage and a breaker
# carrying more than its rating for a while before it trips. The bound only has
# to separate an hourly delta (at most a few kWh) from a lifetime counter
# (typically hundreds or thousands of kWh).
HOURLY_BOUND_MARGIN = 2.0
# Used when a breaker has no usable currentRating. Above the rating of typical
# residential branch breakers, so a real hourly delta is never mistaken for a
# lifetime counter.
FALLBACK_RATING_AMPS = 60.0

# Lifetime counters have 0.1 kWh resolution, and the offset recorded from the
# first one carries up to 0.05 kWh of rounding, so smaller differences are
# rounding, not missed energy.
REANCHOR_MIN_KWH = 0.1

# The REST listing keeps showing the previous hour's delta until the HH:00:29
# burst has been processed by the cloud, so listing values are only booked
# from this point in the hour on.
REST_SETTLE_SECS = 300

# The hourly burst arrives over the WebSocket at HH:00:29 (every hour of the
# capture, all breakers in the same second); lifetime counters and stale
# values (e.g. a cached delta re-sent after a reconnect) arrive at other
# times. Only a WebSocket value received in [WS_DELTA_WINDOW_START_SECS,
# WS_DELTA_WINDOW_END_SECS) into the hour is an hourly delta (if within the
# physical bound). The window starts 9 s before the burst, for clock skew
# between Home Assistant and the cloud, so a stale value from the first 20 s
# of the hour cannot claim the hour before the burst. It ends 61 s after the
# burst, for delivery lag; the window is 70 s, short enough that a small
# lifetime counter (a new or rarely used breaker) seldom lands in it, and
# well before REST_SETTLE_SECS, so a listing value is never booked for an
# hour whose burst can still arrive. A burst outside the window is not
# booked; the listing recovery books that hour instead.
WS_DELTA_WINDOW_START_SECS = 20
WS_DELTA_WINDOW_END_SECS = 90

# Lifetime counters are multiples of 0.1 kWh; bursts are unrounded floats.
# The listing rounds its deltas to 0.01 kWh, so a listing value that is a
# multiple of 0.1 kWh may be a delta (one in ten rounded deltas is) or a
# small lifetime counter. Above this value such a listing value is ignored:
# a skipped delta leaves a gap that the next lifetime counter fills, while a
# counter booked as a delta adds energy nothing ever takes back. At or below
# it the delta is booked: 0.1 kWh is a common hourly delta, and a counter
# that small can add at most 0.1 kWh.
REST_AMBIGUOUS_MIN_KWH = 0.1

# The listing rounds deltas to 0.01 kWh: a listing value within this distance
# of the last booked delta may be that delta shown again (see
# LDATAEnergyLedger._may_be_stale_listing).
STALE_LISTING_TOLERANCE_KWH = 0.005 + 1e-9

# Daily totals kept per breaker (today, yesterday and one spare).
DAILY_KEEP_DAYS = 3

LEGS = ("1", "2")


def hourly_bound_kwh(rating: Any) -> float:
    """Return the most energy one leg of a breaker can use in one hour (kWh)."""
    try:
        amps = float(rating)
    except (TypeError, ValueError):
        amps = 0.0
    if not math.isfinite(amps) or amps <= 0:
        amps = FALLBACK_RATING_AMPS
    return amps * NOMINAL_LEG_VOLTAGE * HOURLY_BOUND_MARGIN / 1000.0


def hour_start(ts: float) -> int:
    """Return the epoch second of the start of the hour containing ``ts``."""
    return int(ts // HOUR_SECS) * HOUR_SECS


def delta_hour(ts: float) -> int:
    """Return the start of the hour an hourly delta received at ``ts`` covers."""
    return hour_start(ts) - HOUR_SECS


def local_date(ts: float) -> datetime.date:
    """Return the local date of epoch second ``ts``."""
    return dt_util.as_local(dt_util.utc_from_timestamp(ts)).date()


def current_daily_date(now: float) -> datetime.date:
    """Return the date whose daily total is still being filled at ``now``.

    The last hour of a day arrives at 00:00:29 the next day, so the daily
    total switches to the new date at 01:00.
    """
    return local_date(delta_hour(now))


def is_counter_resolution(value: float) -> bool:
    """Return True if ``value`` is a multiple of 0.1 kWh (lifetime counter resolution)."""
    return abs(value * 10 - round(value * 10)) < 1e-6


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _hour(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _migrated_complete_until(state: dict) -> int | None:
    """Return ``complete_until`` for a leg stored before the field existed.

    That version took the leg as complete up to its last booked hour, so
    this keeps its behaviour for the stored state: no energy is added on the
    assumption of a gap it did not track. An hour claimed by a lifetime
    counter (no booked value) is not complete, so the next counter adds the
    rest of it. A leg with only a reference (no hour) is complete up to the
    hour before the reference.
    """
    if state["last_hour"] is not None:
        if state["last_value"] is None:
            return state["last_hour"] - HOUR_SECS
        return state["last_hour"]
    if state["offset"] is not None and state["anchor_time"] is not None:
        return delta_hour(state["anchor_time"])
    return None


class LDATAEnergyLedger:
    """Lifetime and daily energy per LDATA breaker, built from hourly deltas."""

    def __init__(self) -> None:
        self._breakers: dict[str, dict[str, Any]] = {}
        # Written from the WebSocket task (event loop) and the REST polls
        # (executor threads).
        self._lock = threading.Lock()
        self.dirty = False
        # Totals as of the last save (see mark_saved); None while the ledger
        # is not persisted.
        self._saved: dict[str, dict[str, Any]] | None = None

    # ── Persistence ────────────────────────────────────────────────────

    def load(self, data: dict | None) -> None:
        """Load a dict produced by ``as_dict``. Invalid entries are dropped."""
        breakers: dict[str, dict[str, Any]] = {}
        stored = data.get("breakers") if isinstance(data, dict) else None
        for key, record in (stored if isinstance(stored, dict) else {}).items():
            if not isinstance(record, dict):
                continue
            legs = {}
            stored_legs = record.get("legs") if isinstance(record.get("legs"), dict) else {}
            for leg in LEGS:
                raw = stored_legs.get(leg) if isinstance(stored_legs.get(leg), dict) else {}
                legs[leg] = {
                    "lifetime": _number(raw.get("lifetime")) or 0.0,
                    "last_hour": _hour(raw.get("last_hour")),
                    "complete_until": _hour(raw.get("complete_until")),
                    "offset": raw.get("offset") if isinstance(raw.get("offset"), (int, float)) else None,
                    "anchor_time": _number(raw.get("anchor_time")),
                    "last_value": _number(raw.get("last_value")),
                    "last_source": raw.get("last_source") if isinstance(raw.get("last_source"), str) else None,
                    "last_counter": _number(raw.get("last_counter")),
                }
                if "complete_until" not in raw:
                    legs[leg]["complete_until"] = _migrated_complete_until(legs[leg])
            stored_daily = record.get("daily") if isinstance(record.get("daily"), dict) else {}
            daily = {
                day: _number(value)
                for day, value in stored_daily.items()
                if isinstance(day, str) and _number(value) is not None
            }
            breakers[str(key)] = {"legs": legs, "daily": daily}
        with self._lock:
            self._breakers = breakers
            self.dirty = False

    def as_dict(self) -> dict:
        """Return a JSON-serialisable copy of the ledger."""
        with self._lock:
            return self._as_dict_locked()

    def take_unsaved(self) -> dict | None:
        """Return a copy of the ledger and clear ``dirty``, or None if it is not dirty."""
        with self._lock:
            if not self.dirty:
                return None
            self.dirty = False
            return self._as_dict_locked()

    def mark_saved(self, data: dict) -> None:
        """Report ``data`` (an ``as_dict`` copy) as written to storage.

        From the first call on, ``lifetime`` and ``daily`` report the totals
        in the last saved copy instead of the live ledger.
        """
        saved = {}
        for key, record in data["breakers"].items():
            saved[key] = {
                "lifetime": sum(state["lifetime"] for state in record["legs"].values()),
                "daily": dict(record["daily"]),
            }
        with self._lock:
            self._saved = saved

    def _as_dict_locked(self) -> dict:
        return {
            "breakers": {
                key: {
                    "legs": {leg: dict(state) for leg, state in record["legs"].items()},
                    "daily": dict(record["daily"]),
                }
                for key, record in self._breakers.items()
            }
        }

    # ── Queries ────────────────────────────────────────────────────────

    def lifetime(self, key: str) -> float | None:
        """Return the breaker's lifetime kWh (both legs), or None if never seen.

        As of the last save once the ledger is persisted (see ``mark_saved``).
        """
        with self._lock:
            if self._saved is not None:
                saved = self._saved.get(key)
                return None if saved is None else saved["lifetime"]
            record = self._breakers.get(key)
            if record is None:
                return None
            return sum(state["lifetime"] for state in record["legs"].values())

    def daily(self, key: str, day: datetime.date) -> float | None:
        """Return the breaker's kWh for ``day`` (0.0 if none yet), or None if never seen.

        As of the last save once the ledger is persisted (see ``mark_saved``).
        """
        with self._lock:
            if self._saved is not None:
                record = self._saved.get(key)
            else:
                record = self._breakers.get(key)
            if record is None:
                return None
            return record["daily"].get(day.isoformat(), 0.0)

    def has_delta_for(self, key: str, hour: int) -> bool:
        """Return True if every leg that has ever had a delta has one for ``hour`` or later.

        A leg that never had a delta (energyConsumption2 null on a single-pole
        breaker) is ignored. A breaker without any delta yet returns False.
        """
        with self._lock:
            record = self._breakers.get(key)
            if record is None:
                return False
            hours = [
                state["last_hour"]
                for state in record["legs"].values()
                if state["last_hour"] is not None
            ]
            return bool(hours) and all(last >= hour for last in hours)

    # ── Updates ────────────────────────────────────────────────────────

    def apply(
        self,
        key: str,
        rating: Any,
        values: dict[str, Any],
        received: float,
        source: str,
        name: str | None = None,
    ) -> bool:
        """Apply energy values for one breaker.

        Args:
            key: Stable breaker id.
            rating: Breaker currentRating (A), used for the delta/lifetime bound.
            values: {"1": energyConsumption, "2": energyConsumption2}; a leg
                    that is absent, None or negative is skipped.
            received: Epoch seconds the value was received.
            source: "WS", "WS-bulk" or "REST".
            name: Breaker name, for logging only.

        Returns True if the ledger changed.
        """
        bound = hourly_bound_kwh(rating)
        hour = delta_hour(received)
        into_hour = received - hour_start(received)
        from_websocket = source.startswith("WS")
        in_burst_window = (
            from_websocket
            and WS_DELTA_WINDOW_START_SECS <= into_hour < WS_DELTA_WINDOW_END_SECS
        )
        label = name or key
        changed = False

        with self._lock:
            record = self._breakers.setdefault(
                key,
                {
                    "legs": {
                        leg: {
                            "lifetime": 0.0,
                            "last_hour": None,
                            "complete_until": None,
                            "offset": None,
                            "anchor_time": None,
                            "last_value": None,
                            "last_source": None,
                            "last_counter": None,
                        }
                        for leg in LEGS
                    },
                    "daily": {},
                },
            )
            for leg, raw in values.items():
                if leg not in record["legs"]:
                    continue
                value = _number(raw)
                if value is None:
                    continue
                state = record["legs"][leg]

                if value > bound:
                    changed |= self._apply_lifetime(record, state, value, bound, received, label, leg)
                    continue
                if from_websocket and not in_burst_window:
                    # No burst due: a lifetime counter if it has counter
                    # resolution, otherwise a stale delta.
                    if is_counter_resolution(value):
                        changed |= self._apply_lifetime(record, state, value, bound, received, label, leg)
                    continue
                if not from_websocket:
                    if into_hour < REST_SETTLE_SECS:
                        continue
                    if value > REST_AMBIGUOUS_MIN_KWH and is_counter_resolution(value):
                        continue  # A rounded delta or a small lifetime counter.

                if state["last_hour"] is not None and hour < state["last_hour"]:
                    continue
                if state["last_hour"] == hour:
                    changed |= self._correct_hour(record, state, hour, value, source, in_burst_window)
                    continue
                if not from_websocket and self._may_be_stale_listing(state, value):
                    _LOGGER.debug(
                        "Breaker %s leg %s: listing value %.2f kWh matches the last booked "
                        "delta and may be stale — leaving the hour to the next lifetime "
                        "counter", label, leg, value,
                    )
                    continue
                state["lifetime"] += value
                state["last_hour"] = hour
                if state["complete_until"] == hour - HOUR_SECS:
                    # Contiguous with the complete hours. complete_until is
                    # set first by the lifetime counter reference.
                    state["complete_until"] = hour
                state["last_value"] = value
                state["last_source"] = source
                self._add_daily(record, hour, value)
                changed = True

            if changed:
                self.dirty = True
        return changed

    @staticmethod
    def _may_be_stale_listing(state: dict, value: float) -> bool:
        """Return True if a listing value may be the last booked delta shown again.

        The listing keeps showing a breaker's last hourly delta until the
        next burst replaces it. When the hub is offline across HH:00:29 there
        is no new burst, and the listing shows the previous hour's delta
        again; neither ``requestVersionHour`` (0 in every capture) nor the
        breaker's ``lastUpdated`` (it moves with power changes, not with the
        energy fields) tells which hour the value belongs to. Booking it
        would count that energy twice, and no later counter takes it back.
        Skipping it leaves a gap that the next lifetime counter fills. This
        includes a zero after a booked zero: it adds nothing, but booking it
        would mark the hour complete, and the next lifetime counter would
        then take the energy of the missed hour as the current hour's (see
        ``_apply_lifetime``), which no burst books.
        """
        last = state["last_value"]
        return last is not None and abs(last - value) <= STALE_LISTING_TOLERANCE_KWH

    def _correct_hour(
        self,
        record: dict,
        state: dict,
        hour: int,
        value: float,
        source: str,
        in_burst_window: bool,
    ) -> bool:
        """Handle a second delta for an hour that is already booked.

        Normally a duplicate. Only a non-zero burst value replaces a booked
        zero, so a stray zero in the burst window does not claim the hour.
        A listing value never replaces anything: it is read after the burst
        window has closed and is at best a rounded copy of the burst.
        """
        booked = state["last_value"]
        if booked is None:
            return False  # Hour claimed by a lifetime counter.
        if not (in_burst_window and booked == 0 and value > 0):
            return False
        state["lifetime"] += value
        state["last_value"] = value
        state["last_source"] = source
        self._add_daily(record, hour, value)
        return True

    def _apply_lifetime(
        self,
        record: dict,
        state: dict,
        value: float,
        bound: float,
        received: float,
        label: str,
        leg: str,
    ) -> bool:
        """Handle a lifetime counter for one leg. Never lowers the lifetime."""
        if value <= 0:
            return False  # A zero outside the burst carries no counter information.
        last_counter = state["last_counter"]
        if last_counter is not None and value <= last_counter + 0.05:
            # The same (or an older) counter seen again, e.g. from the listing
            # after a burst was booked. Comparing it with the ledger now would
            # lower the offset by energy the counter does not contain yet.
            return False
        state["last_counter"] = value

        if state["offset"] is None:
            self._set_reference(state, value, received)
            _LOGGER.debug(
                "Breaker %s leg %s: lifetime counter %.1f kWh recorded as reference "
                "(ledger %.3f kWh)", label, leg, value, state["lifetime"],
            )
            return True

        expected = state["lifetime"] + state["offset"]
        missing = value - expected
        if missing <= REANCHOR_MIN_KWH:
            if missing < 0:
                # The ledger is ahead of the cloud counter. Lower the offset
                # so later comparisons are not inflated by rounding or by
                # partial-hour energy in the first reference.
                self._set_reference(state, value, received)
            else:
                # In step with the counter: the ledger is complete up to now.
                state["anchor_time"] = received
                state["complete_until"] = delta_hour(received)
            return True

        # The most the breaker can have used since the ledger was last known
        # to be complete: the end of the last complete hour, or the last
        # counter, whichever is later.
        known_until = state["anchor_time"] or 0.0
        if state["complete_until"] is not None:
            known_until = max(known_until, state["complete_until"] + HOUR_SECS)
        possible = bound * max(received - known_until, 0.0) / HOUR_SECS + bound
        if known_until <= 0 or missing > possible:
            # A counter jump the breaker cannot physically explain (a reset
            # or replaced hub, or a glitch). Take it as the new reference
            # without adding energy.
            self._set_reference(state, value, received)
            _LOGGER.warning(
                "Breaker %s leg %s: lifetime counter %.1f kWh is %.1f kWh above the "
                "ledger, more than the breaker can use in that time — taking it as "
                "the new reference without adding energy", label, leg, value, missing,
            )
            return True

        if (
            state["complete_until"] is not None
            and state["complete_until"] >= delta_hour(received)
            and missing
            <= bound * (received - hour_start(received)) / HOUR_SECS + REANCHOR_MIN_KWH
        ):
            # The ledger has every hour up to the current one, so the excess
            # is the current hour's energy so far; its burst books the whole
            # hour (or, if the hour is claimed by an earlier counter, the
            # next counter adds it). Keep the offset: raising it to this
            # counter would put that partial hour into the reference and
            # make the next real gap come up short by the same amount.
            # More than the breaker can have used since the start of the
            # hour means complete_until is wrong (a gap it did not track,
            # e.g. in a ledger stored before the field existed): the excess
            # is added below as missed energy.
            state["anchor_time"] = received
            _LOGGER.debug(
                "Breaker %s leg %s: lifetime counter %.1f kWh is %.2f kWh above the "
                "ledger, all of it in the current hour — left to the hourly burst",
                label, leg, value, missing,
            )
            return True

        # The cloud counted energy the ledger never saw (hours missed since
        # complete_until, also ones followed by booked hours). The counter
        # also covers the current hour up to now, so that hour is claimed:
        # its burst would otherwise count the same energy twice. The ledger
        # is complete up to the hour before; complete_until stays there, so
        # the next counter finds the claimed hour incomplete and adds the
        # rest of it.
        state["lifetime"] += missing
        state["anchor_time"] = received
        current_hour = hour_start(received)
        if state["last_hour"] is None or state["last_hour"] < current_hour:
            state["last_hour"] = current_hour
        state["complete_until"] = delta_hour(received)
        # The hour is claimed by the counter: no delta value to correct.
        state["last_value"] = None
        state["last_source"] = None
        self._add_daily(record, current_hour, missing)
        _LOGGER.info(
            "Breaker %s leg %s: lifetime counter %.1f kWh is %.2f kWh above the "
            "ledger — adding the missed energy", label, leg, value, missing,
        )
        return True

    @staticmethod
    def _set_reference(state: dict, value: float, received: float) -> None:
        # The reference takes in everything up to now, so the ledger is
        # complete up to the hour before the current one.
        state["offset"] = value - state["lifetime"]
        state["anchor_time"] = received
        state["complete_until"] = delta_hour(received)

    @staticmethod
    def _add_daily(record: dict, hour: int, value: float) -> None:
        daily = record["daily"]
        day = local_date(hour).isoformat()
        daily[day] = max(daily.get(day, 0.0) + value, 0.0)
        for old in sorted(daily)[:-DAILY_KEEP_DAYS]:
            del daily[old]
