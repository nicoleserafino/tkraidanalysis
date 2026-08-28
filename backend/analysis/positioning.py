"""Player positioning analysis for raid fights.

Fetches position data from WCL v2 API (via includeResources on events)
and identifies positioning issues around key mechanic events like
Conflagration spread, grouping failures, etc.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from backend.wcl.client import graphql_query


# Mechanics worth tracking positions for (spell IDs)
TRACKED_MECHANICS = {
    # Kael'thas - Conflagration from Capernian: an AoE that catches players who
    # stand too close to her. WCL does not reliably log Capernian's own position
    # (her cast events carry no coords; her damage events carry the victim's
    # position), so we cannot measure true distance-to-Capernian. Instead we flag
    # the players actually hit, and when a single cast catches 2+ of them we call
    # them out as clustered too close together in her danger zone.
    37018: {"name": "Conflagration", "boss": "Kael'thas Sunstrider", "spread_range": 700, "type": "proximity_aoe"},
    37019: {"name": "Conflagration", "boss": "Kael'thas Sunstrider", "spread_range": 700, "type": "proximity_aoe"},
    # Hydross - Water Tomb (targets random players)
    38235: {"name": "Water Tomb", "boss": "Hydross the Unstable", "type": "stack"},
    # Vashj - Static Charge (must spread)
    38280: {"name": "Static Charge", "boss": "Lady Vashj", "spread_range": 4000, "type": "spread"},
    # Morogrim - Watery Grave (players teleported)
    37850: {"name": "Watery Grave", "boss": "Morogrim Tidewalker", "type": "displacement"},
    # Solarian - Wrath of the Astromancer (debuff that explodes on nearby)
    42783: {"name": "Wrath of the Astromancer", "boss": "High Astromancer Solarian", "spread_range": 4000, "type": "spread"},
    # Al'ar - Flame Quills (players must be below platform)
    35383: {"name": "Flame Quills", "boss": "Al'ar", "type": "positioning"},
    # Al'ar - Dive Bomb / Meteor (spread to minimize splash)
    35181: {"name": "Dive Bomb", "boss": "Al'ar", "spread_range": 700, "type": "spread"},
    # Void Reaver - Arcane Orb (spread to avoid splash + silence)
    34172: {"name": "Arcane Orb", "boss": "Void Reaver", "spread_range": 500, "type": "spread"},

    # ── Mount Hyjal ────────────────────────────────────────────────────────
    # Kaz'rogal - Mark of Kaz'rogal: drains mana, then detonates for AoE damage
    # to everyone nearby when mana hits 0. Mana users must spread from the raid.
    # (31447 = the applied debuff aura; 31463 = the detonation damage.)
    # The application itself is unavoidable (it targets mana users), so proximity
    # at apply-time is NOT a failure — the failure is detonating near others.
    31447: {"name": "Mark of Kaz'rogal", "boss": "Kaz'rogal", "spread_range": 400, "type": "spread", "application_expected": True},
    # Archimonde - Air Burst: knocks the target into the air; the fall is what kills
    # (Tears of the Goddess negates it). We flag who got hit / who was clustered.
    32014: {"name": "Air Burst", "boss": "Archimonde", "spread_range": 500, "type": "spread"},

    # ── Black Temple ───────────────────────────────────────────────────────
    # Spell IDs verified against a real full-clear log (BHLKntbk6xMc839R) by
    # counting applydebuff events per boss.
    # Mother Shahraz - Fatal Attraction: teleports 3 players together; they take
    # escalating damage while close to each other and MUST spread apart fast.
    # (41001 = the applied debuff; 40870 is the parent spell, never applied.)
    # Targets are TELEPORTED together, so they are ~0yd at application by design —
    # apply-time proximity is expected; the real metric is how fast they separate.
    41001: {"name": "Fatal Attraction", "boss": "Mother Shahraz", "spread_range": 4000, "type": "spread", "application_expected": True},
    # Illidan Stormrage - Parasitic Shadowfiend: debuff that spawns adds; affected
    # players spread from the raid so the fiends don't chain-infect everyone.
    41917: {"name": "Parasitic Shadowfiend", "boss": "Illidan Stormrage", "spread_range": 800, "type": "spread"},
    # Illidan Stormrage - Agonizing Flames: a fire DoT that also burns nearby
    # allies; the target runs away from the raid to avoid spreading it.
    40932: {"name": "Agonizing Flames", "boss": "Illidan Stormrage", "spread_range": 700, "type": "spread"},
}

POSITION_QUERY = """
query ($code: String!, $fightIDs: [Int]!, $dataType: EventDataType!, $startTime: Float!, $endTime: Float!, $filterExpression: String) {
  reportData {
    report(code: $code) {
      events(
        fightIDs: $fightIDs
        dataType: $dataType
        startTime: $startTime
        endTime: $endTime
        includeResources: true
        filterExpression: $filterExpression
      ) {
        data
        nextPageTimestamp
      }
    }
  }
}
"""

DEBUFF_QUERY = """
query ($code: String!, $fightIDs: [Int]!, $startTime: Float!, $endTime: Float!, $filterExpression: String) {
  reportData {
    report(code: $code) {
      events(
        fightIDs: $fightIDs
        dataType: Debuffs
        startTime: $startTime
        endTime: $endTime
        filterExpression: $filterExpression
      ) {
        data
        nextPageTimestamp
      }
    }
  }
}
"""

META_QUERY = """
query ($code: String!) {
  reportData {
    report(code: $code) {
      fights(killType: Encounters) {
        id
        name
        encounterID
        startTime
        endTime
        kill
      }
      masterData {
        actors { id name type subType }
        abilities { gameID name }
      }
    }
  }
}
"""


def _distance(p1: dict, p2: dict) -> float:
    """Euclidean distance between two position points."""
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


# Two players cannot truly occupy the exact same integer coordinate; identical
# coords (or the (0,0) sentinel) almost always mean the position was unresolved
# in the sampling window, so a computed "0 yard" gap there is an artifact, not a
# real stack. We also reject pairs whose position samples are too far apart in
# time to be a meaningful simultaneous distance.
_MAX_PAIR_SKEW_MS = 1500


def _reliable_pair(p1: dict, p2: dict) -> bool:
    """True only when a distance between two sampled positions is trustworthy."""
    for p in (p1, p2):
        if p.get("x", 0) == 0 and p.get("y", 0) == 0:
            return False  # (0,0) sentinel / unresolved
    if p1["x"] == p2["x"] and p1["y"] == p2["y"]:
        return False  # identical coords = sampling artifact, not a real 0yd stack
    if abs(p1.get("ts", 0) - p2.get("ts", 0)) > _MAX_PAIR_SKEW_MS:
        return False  # samples too far apart in time to compare
    return True


# WCL coordinates are ~87.5 units per yard (derived from conflag 8yd = ~700 units)
UNITS_PER_YARD = 87.5


async def _fetch_positions_at_time(
    code: str, fight_id: int, timestamp: float, window_ms: int = 3000,
    players: dict[int, str] | None = None,
) -> dict[str, dict]:
    """Fetch player positions near a timestamp by combining multiple event types.
    
    Returns {player_name: {x, y, ts}} for the closest event to timestamp.
    """
    start = timestamp - window_ms
    end = timestamp + window_ms

    positions: dict[str, dict] = {}

    async def _fetch_one(dtype: str):
        variables = {
            "code": code,
            "fightIDs": [fight_id],
            "dataType": dtype,
            "startTime": start,
            "endTime": end,
        }
        try:
            data = await graphql_query(POSITION_QUERY, variables)
            return data["reportData"]["report"]["events"].get("data", [])
        except Exception:
            return []

    # Fetch all data types in parallel
    results = await asyncio.gather(
        _fetch_one("Casts"), _fetch_one("DamageDone"), _fetch_one("Healing")
    )

    for events in results:
        for e in events:
            if "x" not in e or "sourceID" not in e:
                continue
            sid = e["sourceID"]
            if players and sid not in players:
                continue
            name = players[sid] if players else str(sid)
            # Keep position closest to target timestamp
            if name not in positions or abs(e["timestamp"] - timestamp) < abs(positions[name]["ts"] - timestamp):
                positions[name] = {"x": e["x"], "y": e["y"], "ts": e["timestamp"]}

    return positions


async def fetch_positioning_data(report_code: str, fight_id: int) -> dict[str, Any]:
    """Fetch positioning snapshots for tracked mechanic events in a fight.
    
    Returns mechanic events with player positions at each event time,
    distance calculations, and spread analysis.
    """
    # Get metadata
    meta = await graphql_query(META_QUERY, {"code": report_code})
    report = meta["reportData"]["report"]

    fight = next((f for f in report["fights"] if f["id"] == fight_id), None)
    if not fight:
        return {"error": "Fight not found", "events": []}

    start = fight["startTime"]
    end = fight["endTime"]
    duration_s = (end - start) / 1000

    # Build actor maps
    actors = report["masterData"]["actors"]
    player_map = {a["id"]: a["name"] for a in actors if a["type"] == "Player"}
    player_classes = {a["name"]: a.get("subType", "Unknown") for a in actors if a["type"] == "Player"}
    abilities = {a["gameID"]: a["name"] for a in report["masterData"]["abilities"]}

    # Find which tracked mechanics are in this fight's ability list
    fight_mechanics = {}
    for ability in report["masterData"]["abilities"]:
        gid = ability["gameID"]
        if gid in TRACKED_MECHANICS:
            fight_mechanics[gid] = TRACKED_MECHANICS[gid]

    if not fight_mechanics:
        # No tracked mechanics — return all player positions at a few timestamps
        # to still show a general positioning view
        snapshots = []
        # Sample 5 evenly-spaced timestamps
        for i in range(5):
            ts = start + (end - start) * (i + 1) / 6
            positions = await _fetch_positions_at_time(
                report_code, fight_id, ts, window_ms=5000, players=player_map
            )
            snapshots.append({
                "time_s": round((ts - start) / 1000, 1),
                "label": f"Positions at {round((ts - start) / 1000)}s",
                "positions": {
                    name: {"x": p["x"], "y": p["y"], "class": player_classes.get(name, "Unknown")}
                    for name, p in positions.items()
                },
                "highlights": [],
            })
        return {
            "fight_name": fight["name"],
            "fight_id": fight_id,
            "kill": fight.get("kill", False),
            "duration_s": round(duration_s, 1),
            "has_mechanics": False,
            "snapshots": snapshots,
        }

    # Fetch debuff application events for tracked mechanics
    mechanic_ids = list(fight_mechanics.keys())
    filter_expr = " OR ".join(f"ability.id={mid}" for mid in mechanic_ids)

    variables = {
        "code": report_code,
        "fightIDs": [fight_id],
        "startTime": float(start),
        "endTime": float(end),
        "filterExpression": filter_expr,
    }
    debuff_data = await graphql_query(DEBUFF_QUERY, variables)
    debuff_events = debuff_data["reportData"]["report"]["events"].get("data", [])

    # Group apply events by timestamp (same-time applies = one mechanic cast)
    mechanic_instances: list[dict] = []
    current_group: dict | None = None
    fired_ids: list[int] = []

    for e in debuff_events:
        if e.get("type") != "applydebuff":
            continue
        ts = e["timestamp"]
        ability_id = e.get("abilityGameID", 0)
        target_name = player_map.get(e.get("targetID"), "Unknown")
        if ability_id not in fired_ids:
            fired_ids.append(ability_id)

        # Group events within 500ms of the SAME ability as one cast. Different
        # abilities firing close together must not be merged into one snapshot.
        if (current_group is None
                or ability_id != current_group["ability_id"]
                or abs(ts - current_group["timestamp"]) > 500):
            current_group = {
                "timestamp": ts,
                "ability_id": ability_id,
                "ability_name": abilities.get(ability_id, "Unknown"),
                "mechanic": fight_mechanics.get(ability_id, {}),
                "targets": [],
            }
            mechanic_instances.append(current_group)

        current_group["targets"].append(target_name)

    # Limit mechanic instances to avoid excessive API calls (3 calls per snapshot)
    MAX_SNAPSHOTS = 10
    if len(mechanic_instances) > MAX_SNAPSHOTS:
        mechanic_instances = mechanic_instances[:MAX_SNAPSHOTS]

    # Fetch all position snapshots in parallel
    position_tasks = [
        _fetch_positions_at_time(report_code, fight_id, float(inst["timestamp"]), window_ms=3000, players=player_map)
        for inst in mechanic_instances
    ]
    all_positions = await asyncio.gather(*position_tasks)

    # Build snapshots with distance calculations
    snapshots = []
    for instance, positions in zip(mechanic_instances, all_positions):

        targets = instance["targets"]
        mechanic_info = instance["mechanic"]
        spread_range = mechanic_info.get("spread_range", 3000)

        # Calculate proximity analysis: find clusters of hit players
        proximity_issues = []
        if mechanic_info.get("type") == "proximity_aoe" and len(targets) >= 1:
            # Conflagration-style AoE: everyone hit was too close to the caster.
            # We can't measure distance to Capernian (not logged), so the callout
            # is simply "who got hit", and when a single cast catches 2+ players
            # we flag them as clustered too close together in the danger zone.
            hit_with_pos = [t for t in targets if t in positions]
            if len(targets) >= 2:
                seen_pairs = set()
                for i, t1 in enumerate(hit_with_pos):
                    for t2 in hit_with_pos[i + 1:]:
                        key = tuple(sorted((t1, t2)))
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        if not _reliable_pair(positions[t1], positions[t2]):
                            continue
                        dist = _distance(positions[t1], positions[t2])
                        proximity_issues.append({
                            "player": t2,
                            "near": t1,
                            "distance": round(dist / UNITS_PER_YARD, 1),
                            "type": "caught_together",
                        })
                proximity_issues.sort(key=lambda x: x["distance"])
        elif mechanic_info.get("type") == "spread" and len(targets) > 1:
            # Find pairs of hit players that were close to each other
            # This indicates stacking/clustering which causes multi-target hits
            for i, t1 in enumerate(targets):
                if t1 not in positions:
                    continue
                for t2 in targets[i + 1:]:
                    if t2 not in positions:
                        continue
                    if not _reliable_pair(positions[t1], positions[t2]):
                        continue
                    dist = _distance(positions[t1], positions[t2])
                    if dist < spread_range:
                        proximity_issues.append({
                            "player": t2,
                            "near": t1,
                            "distance": round(dist / UNITS_PER_YARD, 1),
                            "type": "clustered_hit",
                        })
            # Find non-hit players who were dangerously close to any hit player
            for t in targets:
                if t not in positions:
                    continue
                for name, pos in positions.items():
                    if name in targets:
                        continue
                    if not _reliable_pair(positions[t], pos):
                        continue
                    dist = _distance(positions[t], pos)
                    if dist < spread_range:
                        proximity_issues.append({
                            "player": name,
                            "near": t,
                            "distance": round(dist / UNITS_PER_YARD, 1),
                            "type": "close_call",
                        })
            proximity_issues.sort(key=lambda x: x["distance"])

        # Format positions for response
        pos_data = {}
        for name, p in positions.items():
            pos_data[name] = {
                "x": p["x"],
                "y": p["y"],
                "class": player_classes.get(name, "Unknown"),
                "hit": name in targets,
            }

        snapshots.append({
            "time_s": round((instance["timestamp"] - start) / 1000, 1),
            "label": f"{instance['ability_name']} at {round((instance['timestamp'] - start) / 1000, 1)}s",
            "ability": instance["ability_name"],
            "ability_id": instance["ability_id"],
            "targets": targets,
            "target_count": len(targets),
            "positions": pos_data,
            "highlights": targets,
            "proximity_issues": proximity_issues,
            "spread_range": spread_range,
            "mechanic_type": mechanic_info.get("type", "unknown"),
            "initial_expected": bool(mechanic_info.get("application_expected", False)),
        })

    return {
        "fight_name": fight["name"],
        "fight_id": fight_id,
        "kill": fight.get("kill", False),
        "duration_s": round(duration_s, 1),
        "has_mechanics": bool(fired_ids),
        "tracked_abilities": [
            {"id": mid, "name": fight_mechanics[mid]["name"],
             "type": fight_mechanics[mid].get("type", "unknown")}
            for mid in fired_ids if mid in fight_mechanics
        ],
        "snapshots": snapshots,
    }


def summarize_player_positioning(
    positioning: dict[str, Any], player_name: str
) -> list[str]:
    """Distil a positioning result into terse per-player findings for AI coaching.

    Returns human-readable lines describing where THIS player was hit by, or
    clustered too close to others during, a tracked spread/proximity mechanic.
    Distances are only asserted when the underlying position samples are
    reliable (see ``_reliable_pair``); for teleport / unavoidable-application
    mechanics we describe the exposure rather than claiming an apply-time gap.
    Empty list when the player had no positioning issues (or no mechanics fired).
    """
    findings: list[str] = []
    if not positioning or not positioning.get("has_mechanics"):
        return findings

    for snap in positioning.get("snapshots", []):
        ability = snap.get("ability", "a mechanic")
        t = snap.get("time_s", 0)
        targets = snap.get("targets", []) or []
        issues = snap.get("proximity_issues", []) or []
        initial_expected = snap.get("initial_expected", False)

        was_hit = player_name in targets

        # Closest clustering issue that involves this player (already reliability-
        # filtered upstream, so any distance here is trustworthy).
        involved = [
            iss for iss in issues
            if iss.get("player") == player_name or iss.get("near") == player_name
        ]
        involved.sort(key=lambda x: x.get("distance", 9999))

        if initial_expected:
            # Targets are teleported/marked together by design — apply-time
            # proximity is expected, so we report exposure, not a "spread" miss.
            if was_hit:
                others = [n for n in targets if n != player_name]
                with_who = f" with {', '.join(others)}" if others else ""
                findings.append(
                    f"{ability} at {t}s: was a target{with_who} — react immediately "
                    f"(separate/manage), this is not an apply-time spread error."
                )
            continue

        if was_hit and involved:
            near = involved[0]
            other = near["near"] if near.get("player") == player_name else near["player"]
            findings.append(
                f"{ability} at {t}s: HIT while only {near.get('distance')}yd from "
                f"{other} — needed to spread further."
            )
        elif was_hit:
            findings.append(f"{ability} at {t}s: was a target (hit).")
        elif involved:
            near = involved[0]
            other = near["near"] if near.get("player") == player_name else near["player"]
            itype = near.get("type", "")
            if itype == "close_call":
                findings.append(
                    f"{ability} at {t}s: was {near.get('distance')}yd from {other} "
                    f"(hit) — close call, tighten spread."
                )
            else:
                findings.append(
                    f"{ability} at {t}s: clustered {near.get('distance')}yd from {other}."
                )
    return findings