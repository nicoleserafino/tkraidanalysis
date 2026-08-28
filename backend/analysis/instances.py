"""Central registry of TBC raid instances, bosses, and encounter IDs.

Single source of truth for which raids/bosses the tool understands. Adding a new
tier (or fixing an encounter ID) happens here and flows through attendance,
progress, wipe, and instance-detection code that imports from this module.

Encounter IDs are the *real* WCL Classic values returned by the v2 API
``fights.encounterID`` (verified against live reports):

  * SSC / TK  -> 100623-100628, 100730-100733   (WCL zone 1056 "SSC / TK")
  * MH / BT   -> 50618-50622,   50601-50609      (WCL zone 1060 "BT / Hyjal")

Boss *names* are kept as a fallback for old reports whose ``encounterID`` may be
missing or normalised, matching how the rest of the codebase does substring
detection.
"""

from __future__ import annotations

from typing import Any


class Instance:
    """One raid instance (a WCL "zone" tier)."""

    def __init__(
        self,
        code: str,
        display: str,
        zone_id: int,
        bosses: list[tuple[int, str]],
        name_substrings: set[str],
    ) -> None:
        self.code = code
        self.display = display
        self.zone_id = zone_id
        # Ordered canonical boss list: [(encounterID, full name), ...]
        self.bosses = bosses
        self.encounter_ids = {eid for eid, _ in bosses}
        self.boss_names = [name for _, name in bosses]
        # Substrings used for name-based fallback detection.
        self.name_substrings = name_substrings


# ── Instance definitions (canonical pull order per raid) ────────────────────
SSC = Instance(
    code="SSC",
    display="Serpentshrine Cavern",
    zone_id=1056,
    bosses=[
        (100623, "Hydross the Unstable"),
        (100624, "The Lurker Below"),
        (100625, "Leotheras the Blind"),
        (100626, "Fathom-Lord Karathress"),
        (100627, "Morogrim Tidewalker"),
        (100628, "Lady Vashj"),
    ],
    name_substrings={"Hydross", "Lurker", "Leotheras", "Karathress", "Morogrim", "Vashj"},
)

TK = Instance(
    code="TK",
    display="Tempest Keep: The Eye",
    zone_id=1056,
    bosses=[
        (100730, "Al'ar"),
        (100731, "Void Reaver"),
        (100732, "High Astromancer Solarian"),
        (100733, "Kael'thas Sunstrider"),
    ],
    name_substrings={"Al'ar", "Void Reaver", "Solarian", "Kael'thas"},
)

MH = Instance(
    code="MH",
    display="Mount Hyjal (Battle for Mount Hyjal)",
    zone_id=1060,
    bosses=[
        (50618, "Rage Winterchill"),
        (50619, "Anetheron"),
        (50620, "Kaz'rogal"),
        (50621, "Azgalor"),
        (50622, "Archimonde"),
    ],
    name_substrings={"Winterchill", "Anetheron", "Kaz'rogal", "Azgalor", "Archimonde"},
)

BT = Instance(
    code="BT",
    display="Black Temple",
    zone_id=1060,
    bosses=[
        (50601, "High Warlord Naj'entus"),
        (50602, "Supremus"),
        (50603, "Shade of Akama"),
        (50604, "Teron Gorefiend"),
        (50605, "Gurtogg Bloodboil"),
        (50606, "Reliquary of Souls"),
        (50607, "Mother Shahraz"),
        (50608, "The Illidari Council"),
        (50609, "Illidan Stormrage"),
    ],
    name_substrings={
        "Naj'entus", "Supremus", "Shade of Akama", "Teron", "Gurtogg",
        "Reliquary", "Shahraz", "Illidari Council", "Illidan",
    },
)

# Registry keyed by short code. Order here is the canonical raid-tier progression.
INSTANCES: dict[str, Instance] = {inst.code: inst for inst in (SSC, TK, MH, BT)}

# Flat, ordered canonical boss list across every known instance.
ALL_BOSS_ORDER: list[str] = [name for inst in INSTANCES.values() for name in inst.boss_names]

# Fast lookups.
ENCOUNTER_TO_INSTANCE: dict[int, str] = {
    eid: inst.code for inst in INSTANCES.values() for eid in inst.encounter_ids
}
ALL_ENCOUNTER_IDS: set[int] = set(ENCOUNTER_TO_INSTANCE)


def instance_for_fight(fight: dict[str, Any]) -> str | None:
    """Return the instance code ("SSC"/"TK"/"MH"/"BT") a fight belongs to, or None.

    Matches on the real ``encounterID`` first, then falls back to a boss-name
    substring (for reports with missing/normalised encounter IDs).
    """
    eid = int(fight.get("encounterID", 0) or 0)
    if eid in ENCOUNTER_TO_INSTANCE:
        return ENCOUNTER_TO_INSTANCE[eid]
    name = fight.get("name", "") or ""
    for inst in INSTANCES.values():
        if any(sub in name for sub in inst.name_substrings):
            return inst.code
    return None


def classify_instances(fights: list[dict]) -> set[str]:
    """Determine which raid instances a report covers (by short code)."""
    found: set[str] = set()
    for f in fights:
        code = instance_for_fight(f)
        if code:
            found.add(code)
    return found


def is_boss_fight(fight: dict[str, Any]) -> bool:
    """True if the fight is a known raid boss encounter."""
    return instance_for_fight(fight) is not None or bool(fight.get("encounterID"))


def boss_order(instance_codes: set[str] | None = None) -> list[str]:
    """Canonical boss ordering, optionally limited to given instance codes."""
    if not instance_codes:
        return list(ALL_BOSS_ORDER)
    return [
        name
        for code, inst in INSTANCES.items()
        if code in instance_codes
        for name in inst.boss_names
    ]


def boss_sort_key(boss_name: str) -> int:
    """Sort key placing a boss in canonical order; unknown bosses sort last."""
    try:
        return ALL_BOSS_ORDER.index(boss_name)
    except ValueError:
        return len(ALL_BOSS_ORDER)
