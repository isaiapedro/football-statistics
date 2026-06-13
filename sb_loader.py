"""
StatsBomb free data loader.

Covers (free open-data):
  International:  WC 2018, WC 2022, Euro 2020, Copa América 2021, AFCON 2021/2023
  Club (partial): La Liga 2004-2021 (no top leagues 2021-2025 → use soccerdata instead)

Install: pip install statsbombpy
Cache:   parquet files written to ./data/sb_cache/ to avoid re-fetching.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

try:
    from statsbombpy import sb
    SB_AVAILABLE = True
except ImportError:
    SB_AVAILABLE = False
    print("statsbombpy not installed. Run: pip install statsbombpy")

CACHE = Path(__file__).parent / "data" / "sb_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Free StatsBomb international competitions
# competition_id → {season_id: label}
FREE_INTL = {
    43: {3: "WC 2018",   106: "WC 2022"},
    55: {43: "Euro 2020"},
    16: {44: "Copa América 2021"},
    6:  {44: "AFCON 2021", 235: "AFCON 2023"},
}


def _cache_path(name: str) -> Path:
    return CACHE / f"{name}.parquet"


def _load_or_fetch(name: str, fetch_fn):
    p = _cache_path(name)
    if p.exists():
        return pd.read_parquet(p)
    df = fetch_fn()
    df.to_parquet(p, index=False)
    return df


# ── Core loaders ──────────────────────────────────────────────────────────────

def get_free_competitions() -> pd.DataFrame:
    """All available StatsBomb free competitions."""
    if not SB_AVAILABLE:
        return pd.DataFrame()
    all_comps = sb.competitions()
    ids = list(FREE_INTL.keys())
    return all_comps[all_comps["competition_id"].isin(ids)].reset_index(drop=True)


def _fetch_matches_for_comp(competition_id: int, season_id: int) -> pd.DataFrame:
    try:
        df = sb.matches(competition_id=competition_id, season_id=season_id)
        df["competition_id"] = competition_id
        df["season_id"]      = season_id
        label = FREE_INTL.get(competition_id, {}).get(season_id, f"{competition_id}/{season_id}")
        df["competition_label"] = label
        return df
    except Exception as e:
        print(f"  WARN: could not fetch {competition_id}/{season_id}: {e}")
        return pd.DataFrame()


def load_all_intl_matches() -> pd.DataFrame:
    """All match metadata across every free international competition."""
    def fetch():
        frames = []
        for cid, seasons in FREE_INTL.items():
            for sid in seasons:
                frames.append(_fetch_matches_for_comp(cid, sid))
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return _load_or_fetch("all_intl_matches", fetch)


def _fetch_events_for_match(match_id: int) -> pd.DataFrame:
    try:
        return sb.events(match_id=match_id)
    except Exception:
        return pd.DataFrame()


def _fetch_lineups_for_match(match_id: int) -> pd.DataFrame:
    """Returns flat DataFrame: match_id, team, player_id, player_name, jersey_number."""
    try:
        raw = sb.lineups(match_id=match_id)
        rows = []
        for team_name, players in raw.items():
            for p in players:
                rows.append({
                    "match_id":    match_id,
                    "team":        team_name,
                    "player_id":   p["player_id"],
                    "player_name": p["player_name"],
                    "jersey_number": p.get("jersey_number"),
                    "country":     p.get("country", {}).get("name") if isinstance(p.get("country"), dict) else None,
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ── Player stats aggregation ──────────────────────────────────────────────────

def _summarise_player_events(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """
    From a match events DataFrame, extract per-player summary:
    goals, shots, assists (key passes → shot), passes_completed, carries.
    """
    if events.empty:
        return pd.DataFrame()

    rows = {}

    def get(pid, pname, team):
        if pid not in rows:
            rows[pid] = {
                "player_id": pid, "player_name": pname, "team": team,
                "match_id": match_id,
                "goals": 0, "shots": 0, "shots_on_target": 0,
                "assists": 0, "passes": 0, "key_passes": 0,
                "dribbles_completed": 0, "minutes_played": 0,
            }
        return rows[pid]

    for _, ev in events.iterrows():
        pid   = ev.get("player_id")
        pname = ev.get("player")
        team  = ev.get("team")
        etype = ev.get("type")
        if not pid or pd.isna(pid):
            continue

        r = get(pid, pname, team)
        if etype == "Shot":
            r["shots"] += 1
            outcome = ev.get("shot_outcome")
            if isinstance(outcome, dict) and outcome.get("name") == "Goal":
                r["goals"] += 1
            if isinstance(outcome, dict) and outcome.get("name") in ("Goal", "Saved"):
                r["shots_on_target"] += 1
        elif etype == "Pass":
            pass_outcome = ev.get("pass_outcome")
            if pd.isna(pass_outcome) or (isinstance(pass_outcome, dict) and pass_outcome.get("name") == "Complete"):
                r["passes"] += 1
            if ev.get("pass_goal_assist") == True:
                r["assists"] += 1
            if ev.get("pass_shot_assist") == True or ev.get("pass_key_pass") == True:
                r["key_passes"] += 1
        elif etype == "Dribble":
            outcome = ev.get("dribble_outcome")
            if isinstance(outcome, dict) and outcome.get("name") == "Complete":
                r["dribbles_completed"] += 1

    return pd.DataFrame(list(rows.values()))


def load_player_intl_stats(force_refresh: bool = False) -> pd.DataFrame:
    """
    Aggregate player stats across all free StatsBomb international competitions.
    Slow on first run (fetches ~200+ matches). Cached to parquet after.

    Returns: one row per player per match.
    Aggregate yourself with .groupby(["player_id", "player_name"]).sum() for career totals.
    """
    cache = _cache_path("player_intl_stats")
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)

    if not SB_AVAILABLE:
        return pd.DataFrame()

    matches = load_all_intl_matches()
    print(f"Fetching events for {len(matches)} matches...")

    all_stats = []
    for i, row in matches.iterrows():
        mid = row["match_id"]
        label = row.get("competition_label", "")
        if i % 20 == 0:
            print(f"  {i}/{len(matches)} — {label}")
        events = _fetch_events_for_match(mid)
        stats  = _summarise_player_events(events, mid)
        if not stats.empty:
            stats["competition_label"] = label
            stats["competition_id"]    = row["competition_id"]
            stats["season_id"]         = row["season_id"]
            stats["match_date"]        = row.get("match_date")
            stats["home_team"]         = row.get("home_team")
            stats["away_team"]         = row.get("away_team")
            all_stats.append(stats)

    df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    df.to_parquet(cache, index=False)
    return df


def player_career_intl(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up per-match stats into career totals per player.
    Returns: player_id, player_name, team, matches, goals, shots, assists, etc.
    """
    agg = (
        stats_df
        .groupby(["player_id", "player_name", "team"])
        .agg(
            matches=("match_id", "nunique"),
            goals=("goals", "sum"),
            shots=("shots", "sum"),
            shots_on_target=("shots_on_target", "sum"),
            assists=("assists", "sum"),
            key_passes=("key_passes", "sum"),
            passes=("passes", "sum"),
            dribbles=("dribbles_completed", "sum"),
        )
        .reset_index()
    )
    agg["goals_per_match"]  = (agg["goals"] / agg["matches"]).round(3)
    agg["shot_conversion"]  = (agg["goals"] / agg["shots"].clip(lower=1)).round(3)
    return agg.sort_values("goals", ascending=False)


def player_stats_vs_team(stats_df: pd.DataFrame, player_name: str, opponent_team: str) -> dict:
    """
    Record of a specific player against a specific opponent team.
    Looks at all matches where player played and the opponent was the other team.
    """
    player = stats_df[stats_df["player_name"].str.lower() == player_name.lower()]
    if player.empty:
        return {"narrative": f"No StatsBomb data found for {player_name}"}

    player_team = player["team"].iloc[0]

    # Matches where player appeared and opponent_team was the other side
    def is_against(row):
        home, away = str(row.get("home_team", "")), str(row.get("away_team", ""))
        return opponent_team.lower() in (home.lower(), away.lower()) and \
               player_team.lower() not in (home.lower(), away.lower()) or \
               (player_team.lower() in (home.lower(), away.lower()) and
                opponent_team.lower() in (home.lower(), away.lower()))

    vs = player[player.apply(is_against, axis=1)]
    if vs.empty:
        return {
            "player": player_name, "opponent": opponent_team,
            "matches": 0, "goals": 0, "assists": 0,
            "narrative": f"{player_name} has never faced {opponent_team} in StatsBomb-covered competitions",
        }

    g = int(vs["goals"].sum())
    a = int(vs["assists"].sum())
    m = int(vs["match_id"].nunique())
    narrative = (
        f"{player_name} vs {opponent_team}: {g} goals, {a} assists in {m} matches"
        if g + a > 0 else
        f"{player_name} vs {opponent_team}: {m} matches, no goals or assists recorded"
    )
    return {"player": player_name, "opponent": opponent_team,
            "matches": m, "goals": g, "assists": a, "narrative": narrative}


# ── Lineups for cohesion ──────────────────────────────────────────────────────

# ── soccerdata FBref — international career stats ─────────────────────────────

INTL_LEAGUES_FBREF = [
    "FIFA World Cup",
    "Copa América",
    "Africa Cup of Nations",
    "UEFA Euro",
    "CONCACAF Gold Cup",
    "AFC Asian Cup",
]


def load_player_intl_career_stats(
    leagues=None,
    seasons=None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Player career stats across international competitions via soccerdata (FBref).
    Covers 2000–present for most competitions. First run is slow (scraping).
    Cached to parquet.

    Returns: player, team, competition, season, goals, assists, apps, minutes, ...
    """
    cache = _cache_path("player_intl_career_fbref")
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)

    try:
        import soccerdata as sd
    except ImportError:
        print("soccerdata not installed. Run: pip install soccerdata")
        return pd.DataFrame()

    if leagues is None:
        leagues = INTL_LEAGUES_FBREF
    if seasons is None:
        seasons = list(range(2000, 2026))

    frames = []
    for lg in leagues:
        try:
            reader = sd.FBref(leagues=lg, seasons=seasons)
            stats  = reader.read_player_season_stats(stat_type="summary")
            stats  = stats.reset_index()
            stats["competition"] = lg
            frames.append(stats)
            print(f"  OK: {lg} — {len(stats)} rows")
        except Exception as e:
            print(f"  WARN {lg}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df


def player_career_summary(career_df: pd.DataFrame, player_name: str) -> dict:
    """
    Aggregate career totals for a player from FBref international stats.
    player_name: matches FBref display name (e.g. "Neymar", "Cristiano Ronaldo").
    """
    p = career_df[career_df["player"].str.lower() == player_name.lower()]
    if p.empty:
        # Try partial match
        p = career_df[career_df["player"].str.lower().str.contains(player_name.lower())]

    if p.empty:
        return {"player": player_name, "narrative": f"No FBref career data found for {player_name}"}

    goals    = int(p["goals"].sum()) if "goals" in p else 0
    assists  = int(p["assists"].sum()) if "assists" in p else 0
    apps     = int(p["apps"].sum()) if "apps" in p else 0
    by_comp  = (p.groupby("competition")["goals"].sum()
                  .sort_values(ascending=False).to_dict()) if "goals" in p else {}
    team     = p["team"].mode()[0] if "team" in p else "unknown"

    return {
        "player": player_name, "team": team,
        "goals": goals, "assists": assists, "apps": apps,
        "by_competition": by_comp,
        "narrative": (
            f"{player_name}: {goals} intl goals, {assists} assists in {apps} appearances. "
            f"Top comp: {next(iter(by_comp), 'N/A')} ({next(iter(by_comp.values()), 0)}g)"
        ),
    }


# ── Club form — last 4 years ──────────────────────────────────────────────────

CLUB_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "EUR-Champions League",
    "EUR-Europa League",
]
CLUB_SEASONS_4Y = ["2021-22", "2022-23", "2023-24", "2024-25"]


def load_player_club_form(
    leagues=None,
    seasons=None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Per-player per-season club stats via soccerdata FBref (last 4 years).
    Covers top 5 leagues + UCL/UEL.

    Cols (FBref summary): player, squad, nation, season, competition,
                          goals, assists, apps, minutes, npxG, xAG, ...

    pip install soccerdata
    """
    cache = _cache_path("player_club_form")
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)

    try:
        import soccerdata as sd
    except ImportError:
        print("soccerdata not installed. Run: pip install soccerdata")
        return pd.DataFrame()

    if leagues is None:
        leagues = CLUB_LEAGUES
    if seasons is None:
        seasons = CLUB_SEASONS_4Y

    frames = []
    for lg in leagues:
        try:
            reader = sd.FBref(leagues=lg, seasons=seasons)
            stats  = reader.read_player_season_stats(stat_type="summary")
            stats  = stats.reset_index()

            # FBref returns MultiIndex columns — flatten to lowercase strings
            if isinstance(stats.columns, pd.MultiIndex):
                stats.columns = [
                    "_".join(str(c).strip() for c in col if str(c) not in ("", "nan")).lower()
                    for col in stats.columns
                ]
            else:
                stats.columns = [str(c).lower().strip() for c in stats.columns]

            # Normalise the player column name (FBref uses "player" at various levels)
            for candidate in ["player", "player_player", "unnamed: 1_level_0_player"]:
                if candidate in stats.columns:
                    stats = stats.rename(columns={candidate: "player"})
                    break

            stats["competition"] = lg
            frames.append(stats)
            print(f"  OK club: {lg} — {len(stats)} rows, cols: {list(stats.columns[:6])}")
        except Exception as e:
            print(f"  WARN club {lg}: {e}")

    if not frames:
        return pd.DataFrame()

    # Align columns across leagues before concat
    all_cols = sorted(set().union(*[set(f.columns) for f in frames]))
    frames = [f.reindex(columns=all_cols) for f in frames]

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df


def player_club_form_summary(club_df: pd.DataFrame, player_name: str,
                              seasons_back: int = 4) -> dict:
    """
    Career club stats summary for a player over the last N seasons.
    Aggregates goals, assists, apps, minutes, xG if available.
    """
    p = club_df[club_df["player"].str.lower() == player_name.lower()]
    if p.empty:
        p = club_df[club_df["player"].str.lower().str.contains(player_name.lower())]
    if p.empty:
        return {"player": player_name, "narrative": f"No FBref club data for {player_name}"}

    # Take last N seasons
    if "season" in p.columns:
        seasons = sorted(p["season"].unique())[-seasons_back:]
        p = p[p["season"].isin(seasons)]

    g  = int(p["goals"].sum())   if "goals"   in p.columns else 0
    a  = int(p["assists"].sum()) if "assists" in p.columns else 0
    m  = int(p["apps"].sum())    if "apps"    in p.columns else 0
    by_season = (
        p.groupby("season")
        .agg(goals=("goals","sum"), assists=("assists","sum"), apps=("apps","sum"))
        .to_dict("index") if "season" in p.columns else {}
    )
    club = p["squad"].mode()[0] if "squad" in p.columns else "unknown"

    return {
        "player": player_name, "current_club": club,
        "goals": g, "assists": a, "apps": m,
        "by_season": by_season,
        "narrative": (
            f"{player_name} ({club}) last {seasons_back} club seasons: "
            f"{g}G {a}A in {m} apps"
        ),
    }


def club_coappearances(club_df: pd.DataFrame, player_a: str, player_b: str) -> dict:
    """
    How many seasons have player_a and player_b shared a club?
    Uses FBref squad (club) data — same club + same season = co-appearance proxy.
    Returns seasons list, shared club, and a cohesion label.
    """
    def find(name):
        r = club_df[club_df["player"].str.lower() == name.lower()]
        if r.empty:
            r = club_df[club_df["player"].str.lower().str.contains(name.lower())]
        return r

    a_df = find(player_a)
    b_df = find(player_b)

    if a_df.empty or b_df.empty:
        return {"player_a": player_a, "player_b": player_b,
                "seasons_together": 0, "narrative": "One or both players not found in club data"}

    # Index by (season, squad) and find intersection
    a_keys = set(zip(a_df["season"], a_df["squad"]))
    b_keys = set(zip(b_df["season"], b_df["squad"]))
    shared = a_keys & b_keys

    if not shared:
        return {
            "player_a": player_a, "player_b": player_b,
            "seasons_together": 0, "shared_clubs": [],
            "narrative": f"{player_a} and {player_b} have never shared a club (last 4 seasons)",
        }

    clubs = list({club for _, club in shared})
    seasons = sorted({s for s, _ in shared})
    ucl = any("Champions" in str(c) for _, c in shared)
    n = len(seasons)

    label = ("veterans" if n >= 3 else "established" if n >= 2 else "recent")
    return {
        "player_a": player_a, "player_b": player_b,
        "seasons_together": n, "shared_clubs": clubs,
        "ucl_together": ucl, "cohesion_label": label,
        "seasons": seasons,
        "narrative": (
            f"{player_a} + {player_b}: {n} season{'s' if n > 1 else ''} together "
            f"at {clubs[0]} — {label} pair"
            + (" (UCL experience)" if ucl else "")
        ),
    }


def load_all_intl_lineups(force_refresh: bool = False) -> pd.DataFrame:
    """
    All player lineups across free international competitions.
    Used as input to cohesion.py for national team pair co-appearance counts.

    Returns: match_id, team, player_id, player_name, competition_label, match_date
    """
    cache = _cache_path("all_intl_lineups")
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)

    if not SB_AVAILABLE:
        return pd.DataFrame()

    matches = load_all_intl_matches()
    print(f"Fetching lineups for {len(matches)} matches...")

    all_lineups = []
    for i, row in matches.iterrows():
        mid = row["match_id"]
        lu  = _fetch_lineups_for_match(mid)
        if not lu.empty:
            lu["competition_label"] = row.get("competition_label", "")
            lu["match_date"]        = row.get("match_date")
            all_lineups.append(lu)

    df = pd.concat(all_lineups, ignore_index=True) if all_lineups else pd.DataFrame()
    df.to_parquet(cache, index=False)
    return df
