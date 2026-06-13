"""
Squad cohesion calculator — two axes, five tiers each.

NATIONAL TIER   (how much this pair has played together for their country)
  T0  < 5 appearances together   → strangers
  T1  5–14                       → familiar
  T2  15–29                      → established
  T3  30–49                      → core pair
  T4  50+                        → veteran partnership

CLUB TIER       (how long this pair has trained/played together at club level)
  T0  different clubs             → no shared club chemistry
  T1  same club < 1 season        → new teammates
  T2  same club 1–2 seasons       → building chemistry
  T3  same club 3+ seasons        → deep chemistry
  T4  same club 3+ seasons + UCL  → elite chemistry (high-pressure games together)

COMPOSITE SCORE = national_tier + club_tier  →  0–8 per pair
Team cohesion   = mean composite score across all pairs in squad

Data sources used:
  - jfjelstul  player_appearances.csv  → historical WC co-appearances (national)
  - sb_loader  load_all_intl_lineups() → recent intl competitions (national)
  - Kaggle     FIFA/Sofifa dataset      → current club affiliation
  - soccerdata FBref                   → club tenure (years together, UCL flag)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict

DATA = Path(__file__).parent / "data"

# Top UEFA coefficient leagues / competitions where pressure = UCL-equivalent
UCL_LEVEL_COMPS = {
    "UEFA Champions League",
    "UEFA Europa League",
}


# ── Tier definitions ──────────────────────────────────────────────────────────

def national_tier(co_appearances: int) -> int:
    if co_appearances >= 50: return 4
    if co_appearances >= 30: return 3
    if co_appearances >= 15: return 2
    if co_appearances >= 5:  return 1
    return 0


def club_tier(seasons_together: float, played_ucl_together: bool = False) -> int:
    if seasons_together <= 0:
        return 0
    if seasons_together < 1:
        t = 1
    elif seasons_together < 3:
        t = 2
    else:
        t = 3
    if t >= 3 and played_ucl_together:
        t = 4
    return t


def composite_score(nat_tier: int, cl_tier: int) -> int:
    return nat_tier + cl_tier


# ── National cohesion from jfjelstul (WC history) ────────────────────────────

def national_pairs_from_jfjelstul(player_appearances_path=DATA / "player_appearances.csv") -> pd.DataFrame:
    """
    Count how many WC matches each pair of players appeared in together.
    Uses jfjelstul player_appearances.csv (covers WC 1930–2018).

    Returns: team, player_a, player_b, co_appearances, national_tier
    """
    df = pd.read_csv(player_appearances_path)
    # Expected cols: player_name, team_name, match_id (or similar)
    # Normalize column names defensively
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    player_col = next((c for c in df.columns if "player" in c and "name" in c), None)
    team_col   = next((c for c in df.columns if "team"   in c and "name" in c), None)
    match_col  = next((c for c in df.columns if "match"  in c and "id"   in c), None)

    if not all([player_col, team_col, match_col]):
        print(f"WARN: unexpected columns in player_appearances.csv: {list(df.columns)}")
        return pd.DataFrame()

    df = df[[player_col, team_col, match_col]].dropna()
    df.columns = ["player", "team", "match_id"]

    rows = []
    for (team, match_id), group in df.groupby(["team", "match_id"]):
        players = group["player"].tolist()
        for a, b in combinations(sorted(players), 2):
            rows.append({"team": team, "player_a": a, "player_b": b})

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pd.DataFrame()

    result = (
        pairs.groupby(["team", "player_a", "player_b"])
        .size()
        .reset_index(name="co_appearances")
    )
    result["national_tier"] = result["co_appearances"].apply(national_tier)
    return result


def national_pairs_from_sb_lineups(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count co-appearances from StatsBomb lineups (recent: WC 2018/2022, Euro, Copa Am, AFCON).

    lineups_df = sb_loader.load_all_intl_lineups()
    Returns: team, player_a, player_b, co_appearances, national_tier
    """
    if lineups_df.empty:
        return pd.DataFrame()

    rows = []
    for (team, match_id), group in lineups_df.groupby(["team", "match_id"]):
        players = group["player_name"].tolist()
        for a, b in combinations(sorted(players), 2):
            rows.append({"team": team, "player_a": a, "player_b": b})

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pd.DataFrame()

    result = (
        pairs.groupby(["team", "player_a", "player_b"])
        .size()
        .reset_index(name="co_appearances")
    )
    result["national_tier"] = result["co_appearances"].apply(national_tier)
    return result


def merge_national_pairs(jf_pairs: pd.DataFrame, sb_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge jfjelstul (WC history) + StatsBomb (recent) national pair counts.
    Sums co_appearances, re-derives national_tier.
    """
    frames = [f for f in [jf_pairs, sb_pairs] if not f.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    result = (
        combined.groupby(["team", "player_a", "player_b"])["co_appearances"]
        .sum()
        .reset_index()
    )
    result["national_tier"] = result["co_appearances"].apply(national_tier)
    return result.sort_values("co_appearances", ascending=False)


# ── Club cohesion from FIFA/soccerdata ───────────────────────────────────────

def load_club_affiliations_kaggle(fifa_path: str) -> pd.DataFrame:
    """
    Load player → club mapping from Kaggle FIFA dataset (Sofifa).
    Expected: players.csv with short_name, nationality_name, club_name, overall, age.

    Returns: player_name, nationality, club, overall_rating
    """
    df = pd.read_csv(fifa_path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    name_col    = next((c for c in df.columns if c in ("short_name", "player_name", "name")), None)
    nat_col     = next((c for c in df.columns if "nation" in c), None)
    club_col    = next((c for c in df.columns if "club" in c and "name" in c), None)
    overall_col = next((c for c in df.columns if c == "overall"), None)

    cols = {k: v for k, v in
            {"player_name": name_col, "nationality": nat_col,
             "club": club_col, "overall": overall_col}.items() if v}
    df = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    return df.dropna(subset=["player_name", "club"])


def load_club_affiliations_soccerdata(leagues=None, seasons=None) -> pd.DataFrame:
    """
    Load player → club from FBref via soccerdata (last 4 years).

    pip install soccerdata

    Returns: player, nationality, squad (club), season, competition
    """
    try:
        import soccerdata as sd
    except ImportError:
        print("soccerdata not installed. Run: pip install soccerdata")
        return pd.DataFrame()

    if leagues is None:
        leagues = [
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1",
            "EUR-Champions League",
        ]
    if seasons is None:
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]

    frames = []
    for lg in leagues:
        try:
            reader = sd.FBref(leagues=lg, seasons=seasons)
            stats  = reader.read_player_season_stats(stat_type="standard")
            stats  = stats.reset_index()
            stats["competition"] = lg
            frames.append(stats[["player", "squad", "nation", "season", "competition"]])
        except Exception as e:
            print(f"  WARN soccerdata {lg}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.columns = ["player", "club", "nationality", "season", "competition"]
    return df


def club_pairs_for_squad(squad_players: list[str], club_df: pd.DataFrame) -> pd.DataFrame:
    """
    For a list of player names (WC squad), compute club tier for each pair.

    club_df: output of load_club_affiliations_* — needs cols: player, club, season
    Returns: player_a, player_b, shared_club, seasons_together, ucl_together, club_tier
    """
    # Filter to squad players (fuzzy match not applied — names must match)
    squad_df = club_df[club_df["player"].isin(squad_players)]

    # Count seasons each pair shared a club
    rows = []
    players_in_data = squad_df["player"].unique()
    for a, b in combinations(sorted(players_in_data), 2):
        a_clubs = squad_df[squad_df["player"] == a].set_index("season")["club"]
        b_clubs = squad_df[squad_df["player"] == b].set_index("season")["club"]
        shared_seasons = [
            s for s in a_clubs.index
            if s in b_clubs.index and a_clubs[s] == b_clubs[s]
        ]
        if not shared_seasons:
            rows.append({"player_a": a, "player_b": b, "shared_club": None,
                         "seasons_together": 0, "ucl_together": False, "club_tier": 0})
            continue

        shared_club = a_clubs[shared_seasons[0]]
        seasons_n   = len(shared_seasons)

        # UCL flag: did they share a club in a season where that club played UCL?
        ucl = False
        if "competition" in squad_df.columns:
            for s in shared_seasons:
                both_ucl = squad_df[
                    (squad_df["player"].isin([a, b])) &
                    (squad_df["season"] == s) &
                    (squad_df["competition"].str.contains("Champions", na=False))
                ]
                if len(both_ucl["player"].unique()) == 2:
                    ucl = True
                    break

        rows.append({
            "player_a": a, "player_b": b, "shared_club": shared_club,
            "seasons_together": seasons_n, "ucl_together": ucl,
            "club_tier": club_tier(seasons_n, ucl),
        })

    return pd.DataFrame(rows)


# ── Full squad cohesion score ─────────────────────────────────────────────────

def squad_cohesion_score(
    team: str,
    national_pairs_df: pd.DataFrame,
    club_pairs_df: pd.DataFrame,
) -> dict:
    """
    Compute composite cohesion score for a squad.

    national_pairs_df : output of merge_national_pairs() — filtered or full
    club_pairs_df     : output of club_pairs_for_squad()

    Returns dict with:
      team, mean_composite, tier_distribution, top_pairs, narrative
    """
    nat = national_pairs_df[national_pairs_df["team"] == team].copy() if "team" in national_pairs_df.columns else national_pairs_df.copy()
    club = club_pairs_df.copy()

    # Merge on player pair
    merged = pd.merge(
        nat[["player_a", "player_b", "co_appearances", "national_tier"]],
        club[["player_a", "player_b", "shared_club", "seasons_together", "club_tier"]],
        on=["player_a", "player_b"],
        how="outer",
    )
    merged["national_tier"] = merged["national_tier"].fillna(0).astype(int)
    merged["club_tier"]     = merged["club_tier"].fillna(0).astype(int)
    merged["composite"]     = merged["national_tier"] + merged["club_tier"]

    if merged.empty:
        return {"team": team, "mean_composite": 0, "narrative": "No cohesion data available"}

    mean_c = round(merged["composite"].mean(), 2)
    max_c  = int(merged["composite"].max())

    # Tier distribution
    dist = merged["composite"].value_counts().sort_index().to_dict()

    # Top 5 pairs
    top = merged.nlargest(5, "composite")[
        ["player_a", "player_b", "co_appearances", "seasons_together", "shared_club", "composite"]
    ].to_dict("records")

    # Club-heavy vs national-heavy
    mean_nat  = merged["national_tier"].mean()
    mean_club = merged["club_tier"].mean()
    basis = "club-chemistry-heavy" if mean_club > mean_nat else "international-experience-heavy"

    # Club clustering (biggest club groups)
    club_groups = (
        merged[merged["shared_club"].notna()]
        .groupby("shared_club").size()
        .sort_values(ascending=False)
        .head(3)
        .to_dict()
    )

    narrative = (
        f"{team} squad cohesion: {mean_c:.1f}/8 avg composite ({basis}). "
        f"Top club cluster: {list(club_groups.keys())[0] if club_groups else 'none'} "
        f"({list(club_groups.values())[0] if club_groups else 0} pairs). "
        f"Max pair score: {max_c}/8."
    )

    return {
        "team": team,
        "mean_composite": mean_c,
        "max_composite": max_c,
        "mean_national_tier": round(float(mean_nat), 2),
        "mean_club_tier": round(float(mean_club), 2),
        "basis": basis,
        "tier_distribution": dist,
        "top_pairs": top,
        "club_clusters": club_groups,
        "narrative": narrative,
    }


# ── Convenience: compare two squads ──────────────────────────────────────────

def compare_cohesion(
    team_a: str, team_b: str,
    national_pairs_df: pd.DataFrame,
    club_df: pd.DataFrame,
    squad_a: list[str],
    squad_b: list[str],
) -> dict:
    """
    Full cohesion comparison for a matchup.

    squad_a/b : list of player names (from jfjelstul squads.csv or manual WC 2026 squads)
    """
    club_a = club_pairs_for_squad(squad_a, club_df)
    club_b = club_pairs_for_squad(squad_b, club_df)

    score_a = squad_cohesion_score(team_a, national_pairs_df, club_a)
    score_b = squad_cohesion_score(team_b, national_pairs_df, club_b)

    more_cohesive = team_a if score_a["mean_composite"] >= score_b["mean_composite"] else team_b
    delta = abs(score_a["mean_composite"] - score_b["mean_composite"])

    return {
        team_a: score_a,
        team_b: score_b,
        "more_cohesive": more_cohesive,
        "delta": round(delta, 2),
        "narrative": (
            f"{more_cohesive} has stronger squad cohesion "
            f"({max(score_a['mean_composite'], score_b['mean_composite']):.1f} vs "
            f"{min(score_a['mean_composite'], score_b['mean_composite']):.1f} avg composite score)"
        ),
    }
