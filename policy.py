"""Deterministic airline-policy checks for writes, and itinerary totals for flight searches.

tau2's airline tools do not enforce the policy (policy.md); the agent must. `policy_violation` re-derives the policy's
eligibility rules from the live reservation and user records (read-only tool calls) and returns the reason a write is
not allowed, or None. Replaying every task's gold actions (check_policy.py), it blocks none of the 31 gold writes it has rules for.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

NOW = datetime(2024, 5, 15, 15, 0, 0)     # tau2 airline's fixed current time (policy.md)

WRITES = {"book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages",
          "update_reservation_flights", "update_reservation_passengers"}


def policy_violation(name: str, args: dict, read: Callable, cancel_reason: Callable[[], str]) -> str | None:
    """Reason the write `name(args)` breaks the airline policy, or None if no rule is broken.

    read(tool, **args) runs a read-only tool and returns its result as plain JSON (None on error).
    cancel_reason() returns the customer's stated reason for cancelling (asked of Jev, only when it matters).
    Checks only what the records can prove; anything uncertain returns None and is left to the model.
    """
    try:
        if name == "cancel_reservation":
            res = read("get_reservation_details", reservation_id=args.get("reservation_id")) or {}
            if not res:
                return None
            statuses = [read("get_flight_status", flight_number=f["flight_number"], date=f["date"]) for f in res.get("flights", [])]
            if any(s in ("landed", "flying") for s in statuses):
                return "part of this reservation has already been flown, so it cannot be cancelled (transfer to a human agent instead)"
            created = datetime.fromisoformat(res.get("created_at"))
            if (NOW - created).total_seconds() <= 24 * 3600 or res.get("cabin") == "business" or "cancelled" in statuses:
                return None
            if res.get("insurance") == "yes" and cancel_reason() in ("health", "weather"):
                return None
            return ("cancellation is only allowed within 24 hours of booking, for business cabin, when the airline cancelled "
                    "a flight, or with travel insurance for a health or weather reason; none applies to this reservation")

        if name == "send_certificate":
            user = read("get_user_details", user_id=args.get("user_id")) or {}
            if not user:
                return None
            eligible = []
            for rid in user.get("reservations", []):
                res = read("get_reservation_details", reservation_id=rid) or {}
                statuses = [read("get_flight_status", flight_number=f["flight_number"], date=f["date"]) for f in res.get("flights", [])]
                if not (user.get("membership") in ("silver", "gold") or res.get("insurance") == "yes" or res.get("cabin") == "business"):
                    continue
                pax = len(res.get("passengers", []))
                if "cancelled" in statuses:
                    eligible.append(100 * pax)
                if "delayed" in statuses:
                    eligible.append(50 * pax)
            if float(args.get("amount", 0)) not in [float(x) for x in eligible]:
                return ("compensation is only for silver/gold members, insured or business reservations, as $100 x passengers "
                        "for a cancelled flight or $50 x passengers for a delayed one; this request does not match")
            return None

        if name == "update_reservation_flights":
            res = read("get_reservation_details", reservation_id=args.get("reservation_id")) or {}
            old = {(f["flight_number"], f["date"]) for f in res.get("flights", [])}
            new = {(f.get("flight_number"), f.get("date")) for f in args.get("flights", [])}
            if res.get("cabin") == "basic_economy" and old != new:
                return "basic economy flights cannot be modified"
    except Exception:
        return None
    return None


TOTALS_TAG = "[Per-passenger itinerary totals computed by the system, cheapest first]"


def itinerary_totals(direct: list, onestop: list) -> str:
    """Text appended to a flight-search result: the 3 cheapest itineraries per cabin, one-stop legs summed.
    This is why the harness can drop tau2's `calculate` tool."""
    direct = [f for f in direct if isinstance(f, dict) and f.get("prices")]
    onestop = [p for p in onestop if isinstance(p, list) and len(p) == 2 and all(isinstance(f, dict) and f.get("prices") for f in p)]
    lines = []
    for cabin in ("basic_economy", "economy", "business"):
        opts = [(f["prices"][cabin], f["flight_number"]) for f in direct if f.get("available_seats", {}).get(cabin, 0) > 0]
        opts += [(a["prices"][cabin] + b["prices"][cabin], f"{a['flight_number']}+{b['flight_number']}") for a, b in onestop
                 if a.get("available_seats", {}).get(cabin, 0) > 0 and b.get("available_seats", {}).get(cabin, 0) > 0]
        lines.append(f"{cabin}: " + (", ".join(f"{fn} ${p}" for p, fn in sorted(opts)[:3]) or "none available"))
    return f"\n\n{TOTALS_TAG} " + " | ".join(lines)
