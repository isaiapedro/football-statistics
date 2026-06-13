"""
World Cup Shorts — Fact Generation Pipeline
Run: python pipeline.py --matchup "Brazil vs Argentina"
     python pipeline.py --team "Brazil"
     python pipeline.py --all-matchups
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import kagglehub
import os
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DATA = Path(__file__).parent / "data"

WC_2026_TEAMS = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", 
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia", 
    "Croatia", "Curaçao", "Czechia", "DR Congo", "Ecuador", 
    "Egypt", "England", "France", "Germany", "Ghana", 
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan", 
    "Jordan", "Mexico", "Morocco", "Netherlands", "New Zealand", 
    "Norway", "Panama", "Paraguay", "Portugal", "Qatar", 
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea", 
    "Spain", "Sweden", "Switzerland", "Tunisia", "Türkiye", 
    "United States", "Uruguay", "Uzbekistan"
]

# WC 2026 venue coordinates + altitude (lat, lon, altitude_m)
WC_2026_VENUE_COORDS = {
    "New York":      (40.8135, -74.0745,    10),
    "Los Angeles":   (33.9534, -118.3392,   71),
    "Dallas":        (32.7480, -97.0927,   185),
    "Kansas City":   (39.0490, -94.4839,   263),
    "Houston":       (29.6847, -95.4107,    15),
    "Miami":         (25.9580, -80.2389,     3),
    "San Francisco": (37.4033, -121.9694,   16),
    "Seattle":       (47.5952, -122.3316,   58),
    "Philadelphia":  (39.9008, -75.1675,    12),
    "Boston":        (42.0909, -71.2643,     9),
    "Vancouver":     (49.2767, -123.1117,   70),
    "Toronto":       (43.6334, -79.4179,    76),
    "Mexico City":   (19.3030, -99.1500,  2240),
    "Guadalajara":   (20.6892, -103.4668, 1566),
    "Monterrey":     (25.6693, -100.2466,  537),
}

# WC tournament start dates — used to snapshot ELO at correct moment
WC_START_DATES = {
    1930: "1930-07-13", 1934: "1934-05-27", 1938: "1938-06-04",
    1950: "1950-06-24", 1954: "1954-06-16", 1958: "1958-06-08",
    1962: "1962-05-30", 1966: "1966-07-11", 1970: "1970-05-31",
    1974: "1974-06-13", 1978: "1978-06-01", 1982: "1982-06-13",
    1986: "1986-05-31", 1990: "1990-06-08", 1994: "1994-06-17",
    1998: "1998-06-10", 2002: "2002-05-31", 2006: "2006-06-09",
    2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14",
    2022: "2022-11-20", 2026: "2026-06-11",
}

# Knockout stage score (user-defined scale)
# 0 = didn't reach knockout
# 1 = Round of 32 exit (2026+ format)
# 2 = Round of 16 exit
# 3 = Quarter-finals exit
# 4 = Semi-finals exit  (Third place match teams also score 4)
# 5 = Final, runner-up
# 6 = Champion
def _stage_ko_score(stage_name: str) -> int:
    """
    Map jfjelstul stage_name → knockout score (0–5).
    Substring matching — robust against name variations across editions.
    Score 6 (champion) applied separately in knockout_stage_score().
    """
    s = stage_name.lower()
    if "third" in s or "place" in s:   return 4  # before 'final' check
    if "semi" in s:                     return 4
    if "quarter" in s:                  return 3
    if "32" in s:                       return 1  # 2026+ round of 32
    if "16" in s or "sixteen" in s:     return 2
    if "second round" in s:             return 2  # pre-1998 naming
    if "final" in s:                    return 5  # plain "Final"
    return 0


def _parse_year(tournament_id: str) -> int:
    """Extract year from tournament_id (e.g. 'WC-1930' → 1930)."""
    import re
    m = re.search(r"(\d{4})", str(tournament_id))
    return int(m.group(1)) if m else 0


# Historical team name aliases → canonical modern name.
# East Germany intentionally excluded — distinct football history, never qualified 2026.
TEAM_ALIASES = {
    "West Germany": "Germany",
}

# WC editions with no traditional knockout Final — assign scores manually.
# 1950: final round-robin pool (no Final match). Pool standings: 1st Uruguay,
#       2nd Brazil, 3rd Sweden, 4th Spain.
SPECIAL_WC_RESULTS = {
    1950: {"Uruguay": 6, "Brazil": 5, "Sweden": 4, "Spain": 4},
}

def _normalize_teams(df: "pd.DataFrame") -> "pd.DataFrame":
    """Replace historical team names with canonical names in home/away columns."""
    for col in ["home_team_name", "away_team_name"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_ALIASES)
    return df


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_international():
    path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
    csv_path = os.path.join(path, "results.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df

def load_goalscorers():
    """
    martj42 companion file — every recorded international goal 1872–present.
    Cols: date, home_team, away_team, team, scorer, own_goal, penalty
    """
    path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
    csv_path = os.path.join(path, "goalscorers.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def load_recent_internationals(intl_df=None, years=4):
    """
    martj42 filtered to last N years. Same schema as load_international().
    Default: 4 years back from today (covers 2022–2026).
    """
    if intl_df is None:
        intl_df = load_international()
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    return intl_df[intl_df["date"] >= cutoff].copy().reset_index(drop=True)

# ── WC 2026 live / current data (ESPN public API) ─────────────────────────────
#
# ESPN scoreboard endpoint — no API key required.
# Returns scores, match stats (shots, SOT, possession, saves, corners), odds.
# Shot-level coordinates not available → use approximation or Poisson model.

_ESPN_WC_URL   = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
_ESPN_SUM_URL  = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"


def _american_to_prob(odds: int) -> float:
    """Convert American moneyline odds → implied probability (no vig removal)."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _remove_vig(p_home, p_draw, p_away):
    """Normalize implied probs to sum to 1.0 (remove bookmaker vig)."""
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def _poisson_match_probs(lam1: float, lam2: float, max_goals: int = 12):
    """Poisson win/draw/loss probs for given (lambda1, lambda2)."""
    from scipy.stats import poisson
    p_win = p_draw = p_loss = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, lam1) * poisson.pmf(j, lam2)
            if i > j:   p_win  += p
            elif i == j: p_draw += p
            else:         p_loss += p
    return p_win, p_draw, p_loss


def match_xg_from_odds(home_ml: int, draw_ml: int, away_ml: int):
    """
    Derive implied xG (expected goals) for each team from moneyline odds.
    Uses Poisson model: find (λ_home, λ_away) that match market win/draw/loss probs.

    home_ml, draw_ml, away_ml: American moneyline odds (e.g. -150, +275, +450)

    Returns dict: home_xg, away_xg, home_win_prob, draw_prob, away_win_prob

    Usage:
        # Brazil vs Morocco from ESPN odds
        xg = match_xg_from_odds(-150, 275, 450)
        # → {'home_xg': 1.62, 'away_xg': 0.81, ...}
    """
    from scipy.optimize import minimize

    p_h, p_d, p_a = _remove_vig(
        _american_to_prob(home_ml),
        _american_to_prob(draw_ml),
        _american_to_prob(away_ml),
    )

    def loss(params):
        l1, l2 = max(params[0], 0.01), max(params[1], 0.01)
        pw, pd, pl = _poisson_match_probs(l1, l2)
        return (pw - p_h)**2 + (pd - p_d)**2 + (pl - p_a)**2

    res = minimize(loss, x0=[1.5, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 5000})
    lam1, lam2 = max(res.x[0], 0.01), max(res.x[1], 0.01)

    return {
        "home_xg":        round(lam1, 3),
        "away_xg":        round(lam2, 3),
        "home_win_prob":  round(p_h, 3),
        "draw_prob":      round(p_d, 3),
        "away_win_prob":  round(p_a, 3),
        "method":         "poisson_odds",
    }


def match_xg_from_shots(
    home_shots: int, home_sot: int,
    away_shots: int, away_sot: int,
):
    """
    Approximate xG from shot counts using WC historical averages.

    Calibrated on StatsBomb WC 2018+2022 data:
      - avg xG per off-target shot: ~0.028
      - avg xG per on-target shot:  ~0.153  (includes goals in SOT)

    Returns dict: home_xg, away_xg, method

    Usage:
        xg = match_xg_from_shots(home_shots=9, home_sot=4, away_shots=2, away_sot=1)
    """
    # xG per shot type calibrated from StatsBomb WC 2018+2022
    XG_SOT   = 0.153
    XG_MISS  = 0.028

    def team_xg(shots, sot):
        off_target = max(shots - sot, 0)
        return round(sot * XG_SOT + off_target * XG_MISS, 3)

    return {
        "home_xg":  team_xg(home_shots, home_sot),
        "away_xg":  team_xg(away_shots, away_sot),
        "method":   "shot_approximation",
    }


def _parse_espn_event(event: dict) -> dict:
    """Extract structured match data from a single ESPN event dict."""
    comp       = event["competitions"][0]
    comps_list = comp.get("competitors", [])
    status     = event["status"]["type"]

    teams = {}
    for c in comps_list:
        side = "home" if c.get("homeAway") == "home" else "away"
        teams[side] = {
            "name":  c["team"]["displayName"],
            "score": int(c.get("score", 0) or 0),
        }

    odds_raw  = comp.get("odds", [{}])[0] if comp.get("odds") else {}
    moneyline = odds_raw.get("moneyline", {}) if isinstance(odds_raw, dict) else {}
    draw_odds = odds_raw.get("drawOdds", {}) if isinstance(odds_raw, dict) else {}

    def _ml_odds(side_dict):
        """current → close → open priority for moneyline odds string."""
        for key in ("current", "close", "open"):
            val = side_dict.get(key, {}).get("odds")
            if val is not None:
                return val
        return None

    return {
        "game_id":    event["id"],
        "name":       event["name"],
        "date":       event["date"],
        "state":      status.get("state"),        # pre / in / post
        "status":     status.get("description"),
        "home_team":  teams.get("home", {}).get("name"),
        "away_team":  teams.get("away", {}).get("name"),
        "home_score": teams.get("home", {}).get("score"),
        "away_score": teams.get("away", {}).get("score"),
        "ml_home":    _ml_odds(moneyline.get("home", {})),
        "ml_draw":    draw_odds.get("moneyLine"),
        "ml_away":    _ml_odds(moneyline.get("away", {})),
        "over_under": odds_raw.get("overUnder"),
    }


def fetch_espn_wc2026(date_str: str = None) -> pd.DataFrame:
    """
    Fetch WC 2026 scoreboard from ESPN.
    date_str: 'YYYYMMDD' e.g. '20260613'. None = today.
    Returns DataFrame with one row per match.

    Usage:
        today = fetch_espn_wc2026()
        june14 = fetch_espn_wc2026('20260614')
    """
    import requests
    params = {}
    if date_str:
        params["dates"] = date_str
    r = requests.get(_ESPN_WC_URL, params=params, timeout=15)
    r.raise_for_status()
    events = r.json().get("events", [])
    return pd.DataFrame([_parse_espn_event(e) for e in events])


def fetch_espn_match_stats(game_id: str) -> dict:
    """
    Fetch shot stats for a specific WC 2026 match from ESPN.
    Returns dict: home_team, away_team, and per-team stats (shots, sot, possession, saves, corners).

    Usage:
        stats = fetch_espn_match_stats("760420")
        xg = match_xg_from_shots(stats["home_shots"], stats["home_sot"],
                                   stats["away_shots"], stats["away_sot"])
    """
    import requests
    r = requests.get(_ESPN_SUM_URL, params={"event": game_id}, timeout=15)
    r.raise_for_status()
    data = r.json()

    box   = data.get("boxscore", {})
    teams = box.get("teams", [])

    result = {}
    for t in teams:
        name   = t.get("team", {}).get("displayName", "unknown")
        stats  = {s["label"].lower().replace(" ", "_"): s.get("displayValue")
                  for s in t.get("statistics", [])}
        side   = "home" if not result else "away"
        result[f"{side}_team"]    = name
        result[f"{side}_shots"]   = int(stats.get("shots", 0) or 0)
        result[f"{side}_sot"]     = int(stats.get("on_goal", 0) or 0)
        result[f"{side}_poss"]    = float(stats.get("possession", 0) or 0)
        result[f"{side}_saves"]   = int(stats.get("saves", 0) or 0)
        result[f"{side}_corners"] = int(stats.get("corner_kicks", 0) or 0)

    result["game_id"] = game_id
    return result


def wc2026_xg_table(date_str: str = None, method: str = "both") -> pd.DataFrame:
    """
    Full xG table for WC 2026 matches on a given date (or today).

    method: "shots" | "odds" | "both"
      - "shots": approximate xG from shots+SOT (requires match to have started)
      - "odds":  Poisson-implied xG from moneyline (pre-match or live)
      - "both":  compute both and show side by side

    Returns DataFrame with columns:
      home_team, away_team, score, state,
      home_xg_shots, away_xg_shots (if shots/both),
      home_xg_odds,  away_xg_odds  (if odds/both)

    Usage:
        df = wc2026_xg_table()                          # today
        df = wc2026_xg_table('20260614', method='odds') # tomorrow pre-match
    """
    matches = fetch_espn_wc2026(date_str)
    if matches.empty:
        print("No matches found.")
        return pd.DataFrame()

    rows = []
    for _, m in matches.iterrows():
        row = {
            "home_team":  m["home_team"],
            "away_team":  m["away_team"],
            "score":      f"{m['home_score']}–{m['away_score']}",
            "state":      m["state"],
            "status":     m["status"],
        }

        # Shot-based xG (only if match started)
        if method in ("shots", "both") and m["state"] in ("in", "post"):
            try:
                s = fetch_espn_match_stats(str(m["game_id"]))
                sg = match_xg_from_shots(s["home_shots"], s["home_sot"],
                                          s["away_shots"], s["away_sot"])
                row["home_xg_shots"] = sg["home_xg"]
                row["away_xg_shots"] = sg["away_xg"]
                row["home_shots"]    = s["home_shots"]
                row["away_shots"]    = s["away_shots"]
                row["home_sot"]      = s["home_sot"]
                row["away_sot"]      = s["away_sot"]
                row["home_poss"]     = s["home_poss"]
                row["away_poss"]     = s["away_poss"]
            except Exception as exc:
                row["home_xg_shots"] = row["away_xg_shots"] = None

        # Odds-based xG (Poisson — works pre-match and live)
        if method in ("odds", "both"):
            try:
                ml_h = m["ml_home"]
                ml_d = m["ml_draw"]
                ml_a = m["ml_away"]
                if all(v is not None for v in [ml_h, ml_d, ml_a]):
                    og = match_xg_from_odds(int(ml_h), int(ml_d), int(ml_a))
                    row["home_xg_odds"]      = og["home_xg"]
                    row["away_xg_odds"]      = og["away_xg"]
                    row["home_win_prob"]     = og["home_win_prob"]
                    row["draw_prob"]         = og["draw_prob"]
                    row["away_win_prob"]     = og["away_win_prob"]
            except Exception:
                row["home_xg_odds"] = row["away_xg_odds"] = None

        rows.append(row)

    return pd.DataFrame(rows)


# ── Club stats loaders ────────────────────────────────────────────────────────
#
# FBref via soccerdata: season-level stats, Top 5 European leagues.
# Understat: match-level and season-level xG, Top 5 + Russian league.
#
# xT at club level requires event coordinates (x,y per pass/carry).
# socceraction provides this framework but is broken on Python 3.13 / NumPy 2.0.
# Workaround documented below in load_club_xt_note().
#
# Supported FBref leagues (soccerdata): "Big 5 European Leagues Combined",
#   "ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"
# Understat leagues: "EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"

_FBREF_TOP5 = [
    "ENG-Premier League", "ESP-La Liga",
    "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A",
]

# Understat league name map (their naming convention)
_UNDERSTAT_LEAGUES = {
    "ENG-Premier League": "EPL",
    "ESP-La Liga":        "La_liga",
    "GER-Bundesliga":     "Bundesliga",
    "ITA-Serie A":        "Serie_A",
    "FRA-Ligue 1":        "Ligue_1",
}

# FBref season notation: year = start of season (2022 → 2022-23)
_CLUB_SEASONS = [2022, 2023, 2024, 2025]


def load_fbref_club_stats(
    leagues=None,
    seasons=None,
    stat_type="standard",
    cache_path=None,
):
    """
    Player season stats from FBref for club leagues via soccerdata.

    leagues: list of FBref league codes. Default: Top 5 combined.
      Use "Big 5 European Leagues Combined" for one-shot fetch of all 5.
    seasons: list of start-years. Default: [2022, 2023, 2024, 2025] (4 seasons).
    stat_type: "standard" | "shooting" | "passing" | "goal_shot_creation" | "defense" | "misc"
      - "standard"   → goals, assists, xg, xag, prgc (progressive carries), prgp, prgr
      - "shooting"   → shots, sot, xg, npxg, dist
      - "passing"    → cmp, att, cmp%, prgp, final_third, key_passes
      - "defense"    → tkl, int, blocks, clr, err
    cache_path: saves/loads result as CSV.

    Usage:
        # All Top 5 leagues, last 4 seasons — standard stats
        df = load_fbref_club_stats(cache_path="data/fbref_club_standard.csv")

        # Shooting stats only (xG + npxG)
        sh = load_fbref_club_stats(stat_type="shooting", cache_path="data/fbref_club_shooting.csv")

        # Find Mbappé's last 4 seasons
        df[df["player"].str.contains("Mbapp")]
    """
    import soccerdata as sd

    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    if leagues is None:
        leagues = ["Big 5 European Leagues Combined"]
    if seasons is None:
        seasons = _CLUB_SEASONS

    frames = []
    for season in seasons:
        try:
            fb = sd.FBref(leagues=leagues, seasons=[season])
            df = fb.read_player_season_stats(stat_type=stat_type)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(filter(None, c)).strip("_") for c in df.columns]
            df = df.reset_index()
            df["season_start"] = season
            frames.append(df)
        except Exception as exc:
            print(f"FBref club {season} failed: {exc}")
            continue

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result


def load_understat_player_xg(player_name, seasons=None):
    """
    Season-by-season xG from Understat for a single player.
    Covers Top 5 leagues + Russian Premier League.
    Async internally — wrapped synchronously here.

    seasons: list of year strings e.g. ["2022", "2023"]. Default: last 4.
    Returns DataFrame: season, goals, shots, xG, xA, key_passes, yellow, red, position.

    Requires network. Data is per-league-season for clubs only.

    Usage:
        df = load_understat_player_xg("Kylian Mbappé")
        df = load_understat_player_xg("Vinicius Junior", seasons=["2023", "2024"])
    """
    import asyncio
    from understat import Understat

    if seasons is None:
        seasons = ["2022", "2023", "2024", "2025"]

    async def _fetch():
        async with Understat() as u:
            # search player by name first
            results = await u.get_stats()  # global stats
            # Understat requires player_id — search via league players
            # Try each league until found
            for league, ustat_name in _UNDERSTAT_LEAGUES.items():
                for season in seasons:
                    try:
                        players = await u.get_league_players(ustat_name, int(season))
                        match = [p for p in players
                                 if player_name.lower() in p.get("player_name", "").lower()]
                        if match:
                            pid = match[0]["id"]
                            grouped = await u.get_player_grouped_stats(pid)
                            return pd.DataFrame(grouped.get("season", []))
                    except Exception:
                        continue
        return pd.DataFrame()

    try:
        return asyncio.run(_fetch())
    except RuntimeError:
        # Already inside event loop (Jupyter)
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_fetch())


def player_club_profile(player_name, fbref_df=None, cache_path=None):
    """
    Club stats for a player across last 4 seasons.
    Filters load_fbref_club_stats() by player name (partial match).

    fbref_df: pre-loaded club stats DataFrame. If None, loads from cache_path or fetches.

    Returns dict with:
      - club_seasons: DataFrame (season, team, goals, assists, xg, ...)
      - understat_xg: season xG from Understat (if reachable)
      - summary: aggregated totals

    Usage:
        profile = player_club_profile("Kylian Mbappé")
        profile = player_club_profile("Vinicius", fbref_df=pre_loaded_df)
    """
    if fbref_df is None:
        fbref_df = load_fbref_club_stats(cache_path=cache_path or str(DATA / "fbref_club_standard.csv"))

    if fbref_df.empty:
        return {"player": player_name, "club_seasons": pd.DataFrame(), "error": "FBref data unavailable"}

    # Find player column — FBref multi-index flattening may rename it
    player_col = next((c for c in fbref_df.columns if "player" in c.lower()), None)
    if player_col is None:
        return {"player": player_name, "club_seasons": pd.DataFrame(), "error": "player column not found"}

    mask = fbref_df[player_col].fillna("").str.lower().str.contains(player_name.lower())
    club_df = fbref_df[mask].copy()

    if club_df.empty:
        return {"player": player_name, "club_seasons": pd.DataFrame(),
                "error": f"'{player_name}' not found in FBref club data"}

    # Build summary over numeric cols
    num_cols = club_df.select_dtypes(include=[np.number]).columns.tolist()
    summary = club_df[num_cols].sum().round(3).to_dict() if num_cols else {}

    return {
        "player": player_name,
        "club_seasons": club_df,
        "summary": summary,
    }


def load_club_xt_note():
    """
    Returns explanation of the xT gap at club level.
    xT from club matches requires event coordinates — not available free for all leagues.

    Options ranked by effort:
      1. StatsBomb open club data (limited competitions, mainly women's/lower tier)
      2. soccerdata WhoScored scraper → kloppy → socceraction xT
         — WhoScored requires session cookie + socceraction broken on Python 3.13 / NumPy 2.0
      3. Opta / TRACAB / StatsBomb paid API
      4. Use FBref progressive carries (prgc) as xT proxy — free, available now

    Progressive carries (prgc) from FBref: carries that move the ball ≥10 yards forward
    and toward the goal. Highly correlated with xT accumulation (~r=0.78 per research).
    Available via load_fbref_club_stats(stat_type="standard") → "prgc" column.
    """
    msg = (
        "Club-level xT gap:\n"
        "  - Event coordinates not freely available for all club matches.\n"
        "  - socceraction broken on Python 3.13 / NumPy 2.0 (np.string_ removed).\n"
        "  - Best free proxy: FBref progressive carries (prgc) ≈ xT accumulation.\n"
        "  - Full fix: pin Python 3.11 + NumPy 1.26, then socceraction + soccerdata WhoScored."
    )
    print(msg)
    return msg


def load_matches(path=DATA / "matches.csv"):
    """jfjelstul worldcup matches (1930–2018). Men's only. Normalizes team aliases."""
    df = pd.read_csv(path)
    df = df[df["tournament_name"].str.contains("FIFA Men's World Cup", case=False)].copy()
    df["year"] = df["tournament_id"].apply(_parse_year)
    df = _normalize_teams(df)
    return df

def load_geolocation():
    """
    Country centroids. Filters out USA state rows, deduplicates on country name.
    Cols: country, country_code, latitude, longitude
    """
    path = kagglehub.dataset_download(
        "paultimothymooney/latitude-and-longitude-for-every-country-and-state"
    )
    csv_path = os.path.join(
        path, "world_country_and_usa_states_latitude_and_longitude_values.csv"
    )
    df = pd.read_csv(csv_path)
    # Drop USA state rows (have a non-null usa_state_code)
    df = df[df["usa_state_code"].isna()].copy()
    df = df[["country", "country_code", "latitude", "longitude"]].dropna(subset=["country"])
    df = df.drop_duplicates(subset=["country"])
    return df

def load_group_standings(path=DATA / "group_standings.csv"):
    df = pd.read_csv(path)
    return df

def load_manager_appearances(path=DATA / "manager_appearances.csv"):
    df = pd.read_csv(path)
    return df

def load_elo():
    """
    Loads historical ELO ratings directly from Kaggle.
    """
    path = kagglehub.dataset_download("afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings")
    
    # 1. Target the correct file in the downloaded Kaggle folder
    csv_path = os.path.join(path, "elo_ratings_wc2026.csv")

    # 2. Map the actual Kaggle columns to the pipeline's expected columns
    COL_TEAM = "country"
    COL_DATE = "snapshot_date"
    COL_ELO  = "rating"

    # 3. Read and standardize the DataFrame
    df = pd.read_csv(csv_path, parse_dates=[COL_DATE])
    df = df.rename(columns={COL_TEAM: "team", COL_DATE: "date", COL_ELO: "elo"})
    return df


# ── Derived datasets ──────────────────────────────────────────────────────────

def group_stage_points(matches_df):
    """
    Derives group stage points per team per year directly from matches_df.
    Uses jfjelstul's confirmed columns: group_stage (0/1 flag), home_team_win,
    away_team_win, draw, home_team_name, away_team_name, year.
    Handles pre-1994 (W=2) vs 1994+ (W=3) rule change.
    """
    grp = matches_df[matches_df["group_stage"] == 1].copy()

    rows = []
    for _, row in grp.iterrows():
        year = int(row["year"])
        w = 3 if year >= 1994 else 2
        home, away = row["home_team_name"], row["away_team_name"]

        if row["home_team_win"]:
            rows += [{"year": year, "team": home, "pts": w},
                     {"year": year, "team": away, "pts": 0}]
        elif row["away_team_win"]:
            rows += [{"year": year, "team": home, "pts": 0},
                     {"year": year, "team": away, "pts": w}]
        else:  # draw
            rows += [{"year": year, "team": home, "pts": 1},
                     {"year": year, "team": away, "pts": 1}]

    df = pd.DataFrame(rows)
    return (
        df.groupby(["year", "team"])["pts"]
        .sum()
        .reset_index()
        .rename(columns={"pts": "group_points"})
    )


def knockout_stage_score(matches_df):
    """
    Knockout stage score per team per year (0–6 scale).
    Uses jfjelstul confirmed columns: knockout_stage (0/1), stage_name,
    home_team_win, away_team_win, home_team_name, away_team_name, year.
    """
    ko = matches_df[matches_df["knockout_stage"] == 1].copy()

    # Champions: home_team_win / away_team_win flags (cleaner than score comparison)
    is_final = ko["stage_name"].str.lower().str.contains("final") & \
               ~ko["stage_name"].str.lower().str.contains("semi|quarter|third|place")
    finals = ko[is_final]
    champions = set()
    for _, row in finals.iterrows():
        year = int(row["year"])
        if row["home_team_win"]:
            champions.add((year, row["home_team_name"]))
        elif row["away_team_win"]:
            champions.add((year, row["away_team_name"]))

    rows = []
    for _, row in ko.iterrows():
        year  = int(row["year"])
        score = _stage_ko_score(row["stage_name"])
        for team in [row["home_team_name"], row["away_team_name"]]:
            if not team:
                continue
            s = 6 if (score == 5 and (year, team) in champions) else score
            rows.append({"year": year, "team": team, "ko_raw": s})

    df = pd.DataFrame(rows)
    result = (
        df.groupby(["year", "team"])["ko_raw"]
        .max()
        .reset_index()
        .rename(columns={"ko_raw": "ko_score"})
    )

    # Patch WC editions that had no traditional Final (e.g. 1950 round-robin pool)
    for year, team_scores in SPECIAL_WC_RESULTS.items():
        for team, score in team_scores.items():
            mask = (result["year"] == year) & (result["team"] == team)
            if mask.any():
                result.loc[mask, "ko_score"] = score
            else:
                result = pd.concat(
                    [result, pd.DataFrame([{"year": year, "team": team, "ko_score": score}])],
                    ignore_index=True,
                )

    return result

def elo_at_wc_start(elo_df):
    """Snapshot ELO for each team at each WC start date."""
    rows = []
    for year, date_str in WC_START_DATES.items():
        cutoff = pd.Timestamp(date_str)
        snapshot = elo_df[elo_df["date"] <= cutoff]
        latest = snapshot.groupby("team")["elo"].last().reset_index()
        latest["year"] = year
        rows.append(latest)
    return pd.concat(rows, ignore_index=True)

def coach_wc_experience(manager_appearances_df):
    """Number of prior WC tournaments each manager had managed."""
    # VERIFY column names after downloading
    # Expected: manager_id, manager_name, team_name, year (or tournament_id)
    df = manager_appearances_df.copy()
    df = df.sort_values("year")
    df["prior_wc_count"] = df.groupby("manager_id").cumcount()
    return df[["manager_name", "team_name", "year", "prior_wc_count"]]


# ── Fact generators ───────────────────────────────────────────────────────────

def drought_since_last_title(matches_df, teams=WC_2026_TEAMS):
    """Years since each team last won the WC (or 'never')."""
    is_final = (
        matches_df["stage_name"].str.lower().str.contains("final") &
        ~matches_df["stage_name"].str.lower().str.contains("semi|quarter|third|place")
    )
    finals = matches_df[is_final].copy()
    facts = []
    for team in teams:
        team_finals = finals[
            (finals["home_team_name"] == team) | (finals["away_team_name"] == team)
        ]
        won_years = []
        for _, row in team_finals.iterrows():
            if row["home_team_name"] == team and row["home_team_win"]:
                won_years.append(int(row["year"]))
            elif row["away_team_name"] == team and row["away_team_win"]:
                won_years.append(int(row["year"]))

        # Patch special WC editions with no Final match
        for yr, team_scores in SPECIAL_WC_RESULTS.items():
            if team in team_scores and team_scores[team] == 6:
                if yr not in won_years:
                    won_years.append(yr)

        if won_years:
            last = max(won_years)
            drought = 2026 - last
            strength = min(drought / 24, 1.0)  # normalized — 24y is ~1 full generation
            facts.append({
                "type": "drought",
                "team": team,
                "value": drought,
                "strength": strength,
                "narrative": f"{team} last won in {last} — {drought}-year drought",
                "titles": won_years,
            })
        else:
            facts.append({
                "type": "no_title",
                "team": team,
                "value": None,
                "strength": 0.6,
                "narrative": f"{team} has never won the World Cup ({len(team_finals)} finals appearances)",
                "titles": [],
            })
    return pd.DataFrame(facts)

def head_to_head_wc(intl_df, team_a, team_b):
    """WC-only head-to-head record between two teams."""
    mask = (
        intl_df["tournament"].str.contains("FIFA World Cup", na=False) &
        (
            ((intl_df["home_team"] == team_a) & (intl_df["away_team"] == team_b)) |
            ((intl_df["home_team"] == team_b) & (intl_df["away_team"] == team_a))
        )
    )
    matches = intl_df[mask].copy()
    if matches.empty:
        return None

    def result_for(row, team):
        if row["home_team"] == team:
            if row["home_score"] > row["away_score"]: return "win"
            if row["home_score"] < row["away_score"]: return "loss"
        else:
            if row["away_score"] > row["home_score"]: return "win"
            if row["away_score"] < row["home_score"]: return "loss"
        return "draw"

    matches["result_a"] = matches.apply(lambda r: result_for(r, team_a), axis=1)
    counts = matches["result_a"].value_counts().to_dict()
    wins, draws, losses = counts.get("win", 0), counts.get("draw", 0), counts.get("loss", 0)
    total = len(matches)
    dominant = team_a if wins > losses else (team_b if losses > wins else None)
    lead = abs(wins - losses)
    strength = lead / total if total > 0 else 0

    narrative = (
        f"{dominant} leads WC H2H {max(wins,losses)}–{min(wins,losses)} ({draws} draws)"
        if dominant else f"Dead even in WC H2H — {wins}W {draws}D {losses}L"
    )
    return {
        "type": "head_to_head",
        "teams": (team_a, team_b),
        "wins": wins, "draws": draws, "losses": losses,
        "total": total,
        "dominant": dominant,
        "strength": strength,
        "narrative": narrative,
        "last_3": matches.sort_values("date").tail(3)[["date", "result_a"]].to_dict("records"),
    }

def knockout_stage_win_rate(ko_df, team):
    """
    Advance rate per knockout stage for a team across all WC editions.
    ko_df = knockout_stage_score() output.
    """
    team_df = ko_df[ko_df["team"] == team]
    stage_labels = {
        1: "Round of 32",
        2: "Round of 16",
        3: "Quarter-finals",
        4: "Semi-finals",
        5: "Final",
    }
    results = {}
    for threshold, label in stage_labels.items():
        appeared = team_df[team_df["ko_score"] >= threshold]
        advanced = team_df[team_df["ko_score"] > threshold]
        if len(appeared) > 0:
            results[label] = {
                "appearances": len(appeared),
                "advanced": len(advanced),
                "advance_rate": round(len(advanced) / len(appeared), 2),
            }
    return results

def best_wc_finish(ko_df, team):
    """Best ko_score a team ever achieved, with the year(s)."""
    t = ko_df[ko_df["team"] == team]
    if t.empty:
        return None
    best = t["ko_score"].max()
    best_years = sorted(t[t["ko_score"] == best]["year"].tolist())
    labels = {0: "group stage", 1: "Round of 32", 2: "Round of 16",
              3: "Quarter-finals", 4: "Semi-finals", 5: "Final (runner-up)", 6: "Champion"}
    label = labels.get(best, f"score {best}")
    years_str = ", ".join(str(y) for y in best_years)
    strength = best / 6
    return {
        "type": "best_finish",
        "team": team,
        "value": best,
        "strength": strength,
        "narrative": f"{team}'s best WC finish: {label} ({years_str})",
        "best_years": best_years,
    }


def wc_appearances(matches_df, team):
    """Total WC tournaments a team appeared in."""
    mask = (matches_df["home_team_name"] == team) | (matches_df["away_team_name"] == team)
    years = matches_df[mask]["year"].unique()
    n = len(years)
    strength = min(n / 20, 1.0)
    return {
        "type": "appearances",
        "team": team,
        "value": n,
        "strength": strength,
        "narrative": f"{team} has appeared in {n} World Cups ({min(years)}–{max(years)})" if n > 0 else f"{team} has no WC appearances on record",
        "years": sorted(years.tolist()),
    }


# ── Competition / year query functions ───────────────────────────────────────

def team_competition_record(intl_df, team, competition_pattern, year=None):
    """
    Team record in a specific competition (and optionally a specific year/edition).

    competition_pattern: partial string matched case-insensitively.
      e.g. "African Cup", "Copa America", "World Cup", "Friendly", "UEFA Euro"
    year: int — filter to matches in that calendar year.

    Example: team_competition_record(intl, "Morocco", "African Cup", 1998)

    Returns dict with W/D/L, goals, narrative.
    """
    mask = (
        ((intl_df["home_team"] == team) | (intl_df["away_team"] == team)) &
        intl_df["tournament"].str.contains(competition_pattern, case=False, na=False)
    )
    if year is not None:
        mask &= intl_df["date"].dt.year == year

    df = intl_df[mask].copy()
    if df.empty:
        return {
            "team": team, "competition": competition_pattern, "year": year,
            "matches": 0,
            "narrative": f"No matches found for {team} in '{competition_pattern}'"
                         + (f" {year}" if year else ""),
        }

    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

    wins   = int((home["home_score"] > home["away_score"]).sum() +
                 (away["away_score"] > away["home_score"]).sum())
    draws  = int((home["home_score"] == home["away_score"]).sum() +
                 (away["away_score"] == away["home_score"]).sum())
    losses = len(df) - wins - draws
    gf     = int(home["home_score"].sum() + away["away_score"].sum())
    ga     = int(home["away_score"].sum() + away["home_score"].sum())

    label = f"{competition_pattern}" + (f" {year}" if year else "")
    return {
        "type": "competition_record",
        "team": team, "competition": competition_pattern, "year": year,
        "matches": len(df), "wins": wins, "draws": draws, "losses": losses,
        "goals_for": gf, "goals_against": ga,
        "win_rate": round(wins / len(df), 3),
        "strength": round(wins / len(df) * min(len(df) / 5, 1.0), 3),
        "narrative": f"{team} in {label}: {wins}W {draws}D {losses}L, {gf}–{ga} goals",
        "matches_df": df,
    }


def player_competition_stats(goals_df, intl_df, player_name,
                              competition_pattern, year=None):
    """
    Player goals in a specific competition (and optionally a specific year).

    Example: player_competition_stats(goals, intl, "Ronaldo", "World Cup", 2002)
    """
    enriched = _enrich_goalscorers(goals_df, intl_df)
    mask = (
        (enriched["scorer"].str.lower() == player_name.lower()) &
        (~enriched["own_goal"].fillna(False)) &
        enriched["tournament"].str.contains(competition_pattern, case=False, na=False)
    )
    if year is not None:
        mask &= enriched["date"].dt.year == year

    df = enriched[mask]
    label = competition_pattern + (f" {year}" if year else "")
    if df.empty:
        return {
            "player": player_name, "competition": label, "goals": 0,
            "narrative": f"{player_name}: 0 goals in {label}",
        }

    goals     = len(df)
    penalties = int(df["penalty"].fillna(False).sum())
    by_opp    = (
        df.assign(opponent=np.where(df["home_team"] == df["team"],
                                    df["away_team"], df["home_team"]))
        .groupby("opponent").size()
        .sort_values(ascending=False)
        .to_dict()
    )
    return {
        "type": "player_competition_stats",
        "player": player_name, "competition": label,
        "goals": goals, "penalties": penalties,
        "by_opponent": by_opp,
        "narrative": f"{player_name} in {label}: {goals} goals ({penalties} pens)",
    }


# ── Player career facts ───────────────────────────────────────────────────────

def _enrich_goalscorers(goals_df, intl_df):
    """Join goalscorers with match metadata to get tournament and opponent context."""
    meta = intl_df[["date", "home_team", "away_team", "tournament"]].copy()
    return goals_df.merge(meta, on=["date", "home_team", "away_team"], how="left")


def player_career_goals(goals_df, intl_df, player_name):
    """
    Whole-career international goal record for a player.
    Returns totals + breakdown by tournament type + top opponent victims.
    """
    enriched = _enrich_goalscorers(goals_df, intl_df)
    mask = (
        enriched["scorer"].str.lower() == player_name.lower()
    ) & (~enriched["own_goal"].fillna(False))
    df = enriched[mask].copy()

    if df.empty:
        return {"player": player_name, "goals": 0,
                "narrative": f"No goals found for {player_name} in martj42 dataset"}

    total     = len(df)
    penalties = int(df["penalty"].fillna(False).sum())

    # Opponent = whichever team is NOT the player's team
    df["opponent"] = np.where(df["home_team"] == df["team"], df["away_team"], df["home_team"])
    top_victims = (
        df.groupby("opponent").size()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    by_tournament = (
        df.groupby("tournament").size()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    # WC-only
    wc_goals = int(df["tournament"].str.contains("FIFA World Cup", na=False).sum())

    narrative = (
        f"{player_name}: {total} intl goals ({wc_goals} at WC, {penalties} pens). "
        f"Top victim: {next(iter(top_victims))} ({next(iter(top_victims.values()))} goals)"
        if top_victims else f"{player_name}: {total} intl goals ({wc_goals} at WC)"
    )
    return {
        "type": "player_career_goals", "player": player_name,
        "goals": total, "wc_goals": wc_goals, "penalties": penalties,
        "top_victims": top_victims, "by_tournament": by_tournament,
        "strength": min(total / 50, 1.0),
        "narrative": narrative,
    }


def player_vs_team(goals_df, intl_df, player_name, opponent_team):
    """
    How many goals has player_name scored against opponent_team, in which competitions.
    Uses martj42 goalscorers.csv — whole career, all competitions.
    """
    enriched = _enrich_goalscorers(goals_df, intl_df)
    mask = (
        (enriched["scorer"].str.lower() == player_name.lower()) &
        (~enriched["own_goal"].fillna(False)) &
        (
            (enriched["home_team"].str.lower() == opponent_team.lower()) |
            (enriched["away_team"].str.lower() == opponent_team.lower())
        )
    )
    df = enriched[mask]

    if df.empty:
        return {
            "type": "player_vs_team", "player": player_name, "opponent": opponent_team,
            "goals": 0, "strength": 0.4,
            "narrative": f"{player_name} has never scored against {opponent_team}",
        }

    goals   = len(df)
    by_comp = df.groupby("tournament").size().sort_values(ascending=False).to_dict()
    dates   = df["date"].sort_values().dt.strftime("%Y-%m-%d").tolist()
    strength = min(goals / 5, 1.0)
    narrative = (
        f"{player_name} scored {goals} goal{'s' if goals > 1 else ''} vs {opponent_team} "
        f"({', '.join(f'{k}: {v}' for k, v in list(by_comp.items())[:3])})"
    )
    return {
        "type": "player_vs_team", "player": player_name, "opponent": opponent_team,
        "goals": goals, "by_competition": by_comp, "dates": dates,
        "strength": round(strength, 3), "narrative": narrative,
    }


def top_scorers_in_matchup(goals_df, intl_df, team_a, team_b, top_n=3):
    """
    Top scorers from each team in all-time H2H meetings.
    Returns ranked list: player, team, goals_in_h2h.
    """
    enriched = _enrich_goalscorers(goals_df, intl_df)
    mask = (
        (~enriched["own_goal"].fillna(False)) &
        (enriched["team"].isin([team_a, team_b])) &
        (
            ((enriched["home_team"] == team_a) & (enriched["away_team"] == team_b)) |
            ((enriched["home_team"] == team_b) & (enriched["away_team"] == team_a))
        )
    )
    df = enriched[mask]
    if df.empty:
        return []

    result = (
        df.groupby(["scorer", "team"])
        .size()
        .reset_index(name="goals")
        .sort_values("goals", ascending=False)
        .head(top_n * 2)
    )
    return result.to_dict("records")


# ── Geo / venue facts ─────────────────────────────────────────────────────────

# WC 2026 host city altitudes (metres). Mexico City is the narrative anchor.
WC_2026_VENUES = {
    "Mexico City":   2240, "Guadalajara": 1566, "Monterrey":    537,
    "Dallas":         185, "Kansas City":  263, "Los Angeles":   71,
    "San Francisco":   16, "Seattle":       58, "Houston":       15,
    "Miami":            3, "Philadelphia":  12, "New York":      10,
    "Boston":           9, "Vancouver":     70, "Toronto":       76,
}

# Rough continent mapping for WC host countries
_CONTINENT = {
    "Brazil":"South America","Argentina":"South America","Uruguay":"South America",
    "Colombia":"South America","Ecuador":"South America","Chile":"South America",
    "Peru":"South America","Venezuela":"South America","Paraguay":"South America",
    "France":"Europe","Germany":"Europe","Spain":"Europe","Italy":"Europe",
    "England":"Europe","Portugal":"Europe","Netherlands":"Europe","Belgium":"Europe",
    "Croatia":"Europe","Serbia":"Europe","Switzerland":"Europe","Denmark":"Europe",
    "Sweden":"Europe","Norway":"Europe","Poland":"Europe","Austria":"Europe",
    "Czech Republic":"Europe","Hungary":"Europe","Romania":"Europe","Slovakia":"Europe",
    "Morocco":"Africa","Senegal":"Africa","Nigeria":"Africa","Ghana":"Africa",
    "Cameroon":"Africa","Ivory Coast":"Africa","Egypt":"Africa","Tunisia":"Africa",
    "South Africa":"Africa","Algeria":"Africa","Mali":"Africa","Burkina Faso":"Africa",
    "United States":"North America","Mexico":"North America","Canada":"North America",
    "Costa Rica":"North America","Honduras":"North America","Jamaica":"North America",
    "Japan":"Asia","South Korea":"Asia","Iran":"Asia","Saudi Arabia":"Asia",
    "Australia":"Asia","China":"Asia","Qatar":"Asia","South Korea":"Asia",
    "New Zealand":"Oceania",
}


def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def home_distance_advantage(geo_df, team_a, team_b,
                             host_country="United States", city=None):
    """
    How far each team travels to a specific WC 2026 venue (or host country centroid).
    city: one of WC_2026_VENUE_COORDS keys (e.g. "Mexico City", "Dallas").
    """
    if city and city in WC_2026_VENUE_COORDS:
        host_lat, host_lon, _ = WC_2026_VENUE_COORDS[city]
    else:
        geo = geo_df.set_index("country")[["latitude", "longitude"]]
        aliases = {"United States": "United States of America"}
        name = aliases.get(host_country, host_country)
        if name not in geo.index:
            return None
        host_lat = float(geo.loc[name, "latitude"])
        host_lon = float(geo.loc[name, "longitude"])

    geo = geo_df.set_index("country")[["latitude", "longitude"]]

    # Aliases: football team name → geo dataset country name
    _GEO_ALIASES = {
        "United States":  "United States of America",
        "South Korea":    "Korea, South",
        "North Korea":    "Korea, North",
        "Ivory Coast":    "Cote d'Ivoire",
        "DR Congo":       "Congo, Democratic Republic of the",
        "England":        "United Kingdom",
        "Scotland":       "United Kingdom",
        "Wales":          "United Kingdom",
    }

    def coords(team):
        for name in [team, _GEO_ALIASES.get(team, team)]:
            if name in geo.index:
                row = geo.loc[name]
                # Handle duplicate index (returns Series, not scalar)
                lat = float(row["latitude"].iloc[0] if hasattr(row["latitude"], "iloc") else row["latitude"])
                lon = float(row["longitude"].iloc[0] if hasattr(row["longitude"], "iloc") else row["longitude"])
                return lat, lon
        return None, None

    results = {}
    for team in [team_a, team_b]:
        lat, lon = coords(team)
        if lat is None:
            results[team] = None
            continue
        dist = _haversine(float(lat), float(lon), host_lat, host_lon)
        continent = _CONTINENT.get(team, "Unknown")
        results[team] = {"distance_km": round(dist), "continent": continent}

    if any(v is None for v in results.values()):
        return None

    closer = min(results, key=lambda t: results[t]["distance_km"])
    farther = max(results, key=lambda t: results[t]["distance_km"])
    gap = results[farther]["distance_km"] - results[closer]["distance_km"]
    venue_label = city if city else host_country
    strength = min(gap / 15000, 1.0)

    return {
        "type": "home_distance",
        "teams": (team_a, team_b),
        "venue": venue_label,
        "distances": results,
        "closer_team": closer,
        "gap_km": gap,
        "strength": round(strength, 3),
        "narrative": (
            f"{closer} travels {results[closer]['distance_km']:,}km to {venue_label} "
            f"vs {results[farther]['distance_km']:,}km for {farther} — "
            f"{gap:,}km gap"
        ),
    }


def altitude_context(team_a, team_b, city=None):
    """
    Altitude fact for a specific city or all WC 2026 high-altitude venues.
    city: one of WC_2026_VENUE_COORDS keys. If None, summarises all high-alt venues.
    """
    high_alt_nations = {"Mexico", "Colombia", "Ecuador", "Peru", "Bolivia",
                        "Chile", "Argentina", "Ethiopia", "Kenya", "Morocco"}
    a_adapted = team_a in high_alt_nations
    b_adapted = team_b in high_alt_nations

    if city and city in WC_2026_VENUE_COORDS:
        _, _, alt = WC_2026_VENUE_COORDS[city]
        high_venue = alt > 500
        if a_adapted and not b_adapted and high_venue:
            note = f"{team_a} altitude-adapted; {team_b} not — {city} ({alt}m) favours {team_a}"
        elif b_adapted and not a_adapted and high_venue:
            note = f"{team_b} altitude-adapted; {team_a} not — {city} ({alt}m) favours {team_b}"
        elif high_venue:
            note = f"{city} at {alt}m — neither team has clear altitude edge"
        else:
            note = f"{city} at {alt}m (sea level) — altitude irrelevant"
        strength = 0.7 if (a_adapted != b_adapted and high_venue) else 0.2
        return {
            "type": "altitude_context", "city": city, "altitude_m": alt,
            "teams": (team_a, team_b),
            "team_a_adapted": a_adapted, "team_b_adapted": b_adapted,
            "strength": strength, "narrative": note,
        }

    # No city — summarise all high-alt venues
    high = {c: v[2] for c, v in WC_2026_VENUE_COORDS.items() if v[2] > 500}
    if a_adapted and not b_adapted:
        note = f"{team_a} altitude-adapted; {team_b} not"
    elif b_adapted and not a_adapted:
        note = f"{team_b} altitude-adapted; {team_a} not"
    else:
        note = "Neither team has clear altitude edge"
    return {
        "type": "altitude_context", "city": None,
        "high_altitude_venues": high,
        "teams": (team_a, team_b),
        "team_a_adapted": a_adapted, "team_b_adapted": b_adapted,
        "strength": 0.55 if (a_adapted != b_adapted) else 0.25,
        "narrative": f"WC 2026: {len(high)} high-alt venues (up to {max(high.values())}m). {note}.",
    }


# ── Depth facts ───────────────────────────────────────────────────────────────

CONTINENTAL_KEYWORDS = {
    "Copa America":   ["copa america", "copa américa"],
    "AFCON":          ["africa cup of nations", "african cup of nations"],
    "Euro":           ["european championship"],
    "Gold Cup":       ["concacaf gold cup", "gold cup"],
    "Asian Cup":      ["afc asian cup", "asian cup"],
    "Nations League": ["nations league"],
    "Confed Cup":     ["confederations cup"],
}


def _team_results(intl_df, team):
    """
    Vectorized: one row per match played by team.
    Cols: date, opponent, result (W/D/L), goals_for, goals_against, tournament, venue.
    """
    neutral_col = "neutral" if "neutral" in intl_df.columns else None

    home = intl_df[intl_df["home_team"] == team].copy()
    home["opponent"]      = home["away_team"]
    home["goals_for"]     = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["result"] = np.where(home["home_score"] > home["away_score"], "W",
                    np.where(home["home_score"] < home["away_score"], "L", "D"))
    home["venue"] = (np.where(home[neutral_col], "neutral", "home")
                     if neutral_col else "home")

    away = intl_df[intl_df["away_team"] == team].copy()
    away["opponent"]      = away["home_team"]
    away["goals_for"]     = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["result"] = np.where(away["away_score"] > away["home_score"], "W",
                    np.where(away["away_score"] < away["home_score"], "L", "D"))
    away["venue"] = (np.where(away[neutral_col], "neutral", "away")
                     if neutral_col else "away")

    cols = ["date", "opponent", "result", "goals_for", "goals_against", "tournament", "venue"]
    return pd.concat([home[cols], away[cols]], ignore_index=True).sort_values("date")


def all_time_h2h(intl_df, team_a, team_b):
    """H2H across all competitions (friendlies, qualifiers, continentals, WC)."""
    mask = (
        ((intl_df["home_team"] == team_a) & (intl_df["away_team"] == team_b)) |
        ((intl_df["home_team"] == team_b) & (intl_df["away_team"] == team_a))
    )
    meetings = intl_df[mask].copy()
    if meetings.empty:
        return {
            "type": "all_time_h2h", "teams": (team_a, team_b), "total": 0,
            "strength": 0.65,
            "narrative": f"{team_a} and {team_b} have NEVER played each other in recorded history",
        }

    a_home = meetings[meetings["home_team"] == team_a]
    a_away = meetings[meetings["away_team"] == team_a]
    wins   = int((a_home["home_score"] > a_home["away_score"]).sum() +
                 (a_away["away_score"] > a_away["home_score"]).sum())
    draws  = int((a_home["home_score"] == a_home["away_score"]).sum() +
                 (a_away["away_score"] == a_away["home_score"]).sum())
    losses = len(meetings) - wins - draws
    total  = len(meetings)

    # Break down by competition bucket
    def bucket(t):
        tl = str(t).lower()
        if "world cup" in tl:       return "World Cup"
        if "friendly" in tl:        return "Friendly"
        if "qualif" in tl:          return "Qualifier"
        return "Continental/Other"
    by_type = meetings["tournament"].apply(bucket).value_counts().to_dict()

    dominant = team_a if wins > losses else (team_b if losses > wins else None)
    lead = abs(wins - losses)
    strength = max(lead / total, 0.35) if total > 0 else 0.35
    narrative = (
        f"{dominant} leads all-time H2H {max(wins,losses)}–{min(wins,losses)} ({draws}D) in {total} games"
        if dominant else
        f"All-time H2H dead even — {wins}W {draws}D {losses}L in {total} games"
    )
    last_3 = (meetings.sort_values("date").tail(3)
              [["date","home_team","home_score","away_score","away_team","tournament"]]
              .to_dict("records"))
    return {
        "type": "all_time_h2h", "teams": (team_a, team_b),
        "wins": wins, "draws": draws, "losses": losses, "total": total,
        "dominant": dominant, "strength": round(strength, 3),
        "narrative": narrative, "by_type": by_type, "last_3": last_3,
    }


def continental_record(intl_df, team):
    """Win/draw/loss record per continental competition."""
    results = _team_results(intl_df, team)
    facts = []
    for cup_name, keywords in CONTINENTAL_KEYWORDS.items():
        pattern = "|".join(keywords)
        mask = results["tournament"].str.lower().str.contains(pattern, na=False)
        sub = results[mask]
        if len(sub) < 3:
            continue
        w     = int((sub["result"] == "W").sum())
        d     = int((sub["result"] == "D").sum())
        l     = int((sub["result"] == "L").sum())
        total = len(sub)
        wr    = w / total
        editions = sub["tournament"].nunique()
        strength = round(wr * min(total / 15, 1.0), 3)
        facts.append({
            "type": "continental_record", "team": team, "cup": cup_name,
            "wins": w, "draws": d, "losses": l, "total": total,
            "win_rate": round(wr, 3), "editions": editions,
            "strength": max(strength, 0.3),
            "narrative": f"{team} in {cup_name}: {w}W {d}D {l}L across {editions} editions ({wr:.0%} win rate)",
        })
    return facts


def form_trajectory(intl_df, team, n=15):
    """Rising or falling? Compare last N games vs previous N games."""
    results = _team_results(intl_df, team).sort_values("date")
    if len(results) < n * 2:
        return None
    recent = results.tail(n)
    older  = results.iloc[-(n * 2):-n]
    recent_wr = (recent["result"] == "W").sum() / n
    older_wr  = (older["result"]  == "W").sum() / n
    delta     = recent_wr - older_wr
    trend     = "rising" if delta > 0.08 else ("falling" if delta < -0.08 else "stable")
    arrow     = "↑" if trend == "rising" else ("↓" if trend == "falling" else "→")
    strength  = max(min(abs(delta) * 2.5, 1.0), 0.3)
    return {
        "type": "form_trajectory", "team": team,
        "recent_win_rate": round(recent_wr, 3), "older_win_rate": round(older_wr, 3),
        "delta": round(delta, 3), "trend": trend,
        "strength": round(strength, 3),
        "narrative": f"{team} form {arrow} — win rate {older_wr:.0%} → {recent_wr:.0%} over last {n} games",
    }


def performance_by_location(intl_df, team):
    """Win rate home / away / neutral — matters at WC (always neutral)."""
    results = _team_results(intl_df, team)
    breakdown = {}
    for venue in ["home", "neutral", "away"]:
        sub = results[results["venue"] == venue]
        if len(sub) < 5:
            continue
        wr = (sub["result"] == "W").sum() / len(sub)
        breakdown[venue] = {"win_rate": round(float(wr), 3), "games": len(sub)}

    if len(breakdown) < 2:
        return None
    rates  = [v["win_rate"] for v in breakdown.values()]
    parts  = [f"{v} {breakdown[v]['win_rate']:.0%}" for v in breakdown]
    gap    = max(rates) - min(rates)
    neutral_wr = breakdown.get("neutral", {}).get("win_rate")
    note   = f" (neutral: {neutral_wr:.0%})" if neutral_wr is not None else ""
    return {
        "type": "location_performance", "team": team, "breakdown": breakdown,
        "strength": round(max(gap * 1.2, 0.3), 3),
        "narrative": f"{team} win rates — {', '.join(parts)}{note}",
    }


def performance_vs_similar_strength(intl_df, team_a, team_b, window_years=8):
    """
    How does team_a perform against opponents at team_b's strength level?
    Strength proxy = opponent win rate over last window_years.
    """
    cutoff = pd.Timestamp("today") - pd.DateOffset(years=window_years)

    b_all    = _team_results(intl_df, team_b)
    b_recent = b_all[b_all["date"] >= cutoff]
    sample   = b_recent if len(b_recent) >= 10 else b_all.tail(30)
    if sample.empty:
        return None
    b_wr = (sample["result"] == "W").sum() / len(sample)

    a_all    = _team_results(intl_df, team_a)
    a_recent = a_all[a_all["date"] >= cutoff]

    similar_rows = []
    for opp in a_recent["opponent"].unique():
        if opp == team_b:
            continue
        opp_res = _team_results(intl_df, opp)
        opp_rec = opp_res[opp_res["date"] >= cutoff]
        if len(opp_rec) < 5:
            continue
        opp_wr = (opp_rec["result"] == "W").sum() / len(opp_rec)
        if abs(opp_wr - b_wr) <= 0.10:
            similar_rows.append(a_recent[a_recent["opponent"] == opp])

    if not similar_rows:
        return None
    df    = pd.concat(similar_rows, ignore_index=True)
    w     = int((df["result"] == "W").sum())
    d     = int((df["result"] == "D").sum())
    l     = int((df["result"] == "L").sum())
    total = len(df)
    wr    = w / total if total > 0 else 0
    return {
        "type": "vs_similar_strength", "team": team_a, "reference_team": team_b,
        "wins": w, "draws": d, "losses": l, "total": total,
        "win_rate": round(wr, 3), "strength": round(min(total / 20, 1.0) * 0.65, 3),
        "narrative": (
            f"{team_a} vs opponents at {team_b}'s strength tier "
            f"(last {window_years}y): {w}W {d}D {l}L in {total} games ({wr:.0%})"
        ),
    }


def generate_matchup_facts(intl_df, matches_df, team_a, team_b,
                           ko_df=None, goals_df=None, geo_df=None, city=None):
    """Batch-generate ranked fact candidates for a specific matchup."""
    if ko_df is None:
        ko_df = knockout_stage_score(matches_df)

    facts = []

    # All-time H2H (every competition)
    facts.append(all_time_h2h(intl_df, team_a, team_b))

    # WC-only H2H
    h2h_wc = head_to_head_wc(intl_df, team_a, team_b)
    if h2h_wc:
        facts.append(h2h_wc)
    else:
        facts.append({
            "type": "no_wc_meeting", "teams": (team_a, team_b), "strength": 0.55,
            "narrative": f"{team_a} and {team_b} have NEVER met at a World Cup",
        })

    # Drought / never-won
    droughts = drought_since_last_title(matches_df, teams=[team_a, team_b])
    for _, row in droughts.iterrows():
        d = row.to_dict()
        if d["type"] == "drought" and d["value"] and d["value"] >= 8:
            facts.append(d)
        elif d["type"] == "no_title":
            d["narrative"] = f"{d['team']} has never won the World Cup"
            d["strength"]  = 0.7
            facts.append(d)

    # Best WC finish + appearances
    for team in [team_a, team_b]:
        bf = best_wc_finish(ko_df, team)
        if bf:
            facts.append(bf)
        app = wc_appearances(matches_df, team)
        if app["value"] > 0:
            facts.append(app)

    # Form trajectory
    for team in [team_a, team_b]:
        form = form_trajectory(intl_df, team)
        if form:
            facts.append(form)

    # Continental records
    for team in [team_a, team_b]:
        facts.extend(continental_record(intl_df, team))

    # Location performance (neutral ground = WC conditions)
    for team in [team_a, team_b]:
        loc = performance_by_location(intl_df, team)
        if loc:
            facts.append(loc)

    # Performance vs opponents at each other's strength tier
    for ta, tb in [(team_a, team_b), (team_b, team_a)]:
        sim = performance_vs_similar_strength(intl_df, ta, tb)
        if sim:
            facts.append(sim)

    # Geo / venue facts
    if geo_df is not None:
        dist = home_distance_advantage(geo_df, team_a, team_b, city=city)
        if dist:
            facts.append(dist)
    facts.append(altitude_context(team_a, team_b, city=city))

    # Player-level facts (requires goalscorers.csv)
    if goals_df is not None:
        scorers = top_scorers_in_matchup(goals_df, intl_df, team_a, team_b)
        for s in scorers:
            facts.append({
                "type": "h2h_top_scorer",
                "player": s["scorer"], "team": s["team"], "goals": s["goals"],
                "strength": min(s["goals"] / 5, 1.0),
                "narrative": f"{s['scorer']} ({s['team']}) scored {s['goals']} goal{'s' if s['goals'] > 1 else ''} in {team_a} vs {team_b} history",
            })

    return sorted(facts, key=lambda x: x.get("strength", 0), reverse=True)


# ── StatsBomb open data ───────────────────────────────────────────────────────
#
# FREE event data available:
#   competition_id=43  FIFA World Cup
#   season_id=3        2018 Russia  (64 matches, full event + freeze-frame)
#   season_id=106      2022 Qatar   (64 matches, full event + freeze-frame)
#
# Install: pip install statsbombpy
# Docs:    github.com/statsbomb/statsbombpy

# xT grid: 12×8 cells over a 120×80 pitch.
# Pre-computed values from Singh 2019 / mplsoccer literature.
# Higher cell = higher probability of scoring from that zone.
_XT_GRID = np.array([
    [0.00638, 0.00949, 0.01500, 0.02143, 0.02506, 0.02838, 0.03667, 0.06500],
    [0.00523, 0.00790, 0.01100, 0.01516, 0.01697, 0.02117, 0.03085, 0.06296],
    [0.00437, 0.00626, 0.00871, 0.01196, 0.01355, 0.01739, 0.02565, 0.05890],
    [0.00393, 0.00580, 0.00795, 0.01086, 0.01231, 0.01581, 0.02335, 0.05690],
    [0.00393, 0.00580, 0.00795, 0.01086, 0.01231, 0.01581, 0.02335, 0.05690],
    [0.00437, 0.00626, 0.00871, 0.01196, 0.01355, 0.01739, 0.02565, 0.05890],
    [0.00523, 0.00790, 0.01100, 0.01516, 0.01697, 0.02117, 0.03085, 0.06296],
    [0.00638, 0.00949, 0.01500, 0.02143, 0.02506, 0.02838, 0.03667, 0.06500],
    [0.00638, 0.00949, 0.01500, 0.02143, 0.02506, 0.02838, 0.03667, 0.06500],
    [0.00523, 0.00790, 0.01100, 0.01516, 0.01697, 0.02117, 0.03085, 0.06296],
    [0.00437, 0.00626, 0.00871, 0.01196, 0.01355, 0.01739, 0.02565, 0.05890],
    [0.00393, 0.00580, 0.00795, 0.01086, 0.01231, 0.01581, 0.02335, 0.05690],
])  # shape (12, 8) — x bins × y bins, attacking direction left→right


def _xy_to_xt(x, y):
    """Convert StatsBomb (x, y) → xT value. Pitch is 120×80. Returns 0 for own half."""
    if x < 60 or x > 120 or y < 0 or y > 80:
        return 0.0
    col = min(int((x - 60) / (60 / 12)), 11)
    row = min(int(y / (80 / 8)), 7)
    return float(_XT_GRID[col, row])


# Complete StatsBomb open-data WC season_id → year mapping
_SB_WC_SEASON_MAP = {
    269: 1958, 270: 1962, 272: 1970, 51: 1974,
    54: 1986, 55: 1990, 3: 2018, 106: 2022,
}


def load_statsbomb_wc(season_ids=None, cid=43):
    """
    Load StatsBomb open match list for the World Cup.
    Default: 2018 (sid=3) + 2022 (sid=106).
    Pass season_ids=list(_SB_WC_SEASON_MAP) for all 8 editions.
    """
    from statsbombpy import sb
    if season_ids is None:
        season_ids = [3, 106]
    frames = []
    for sid in season_ids:
        m = sb.matches(competition_id=cid, season_id=sid)
        m["season_id"] = sid
        m["wc_year"] = _SB_WC_SEASON_MAP.get(sid, sid)
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def load_statsbomb_all_wc():
    """
    Load ALL 8 StatsBomb WC editions with open event data:
    1958, 1962, 1970, 1974, 1986, 1990, 2018, 2022.
    NOTE: first run fetches ~300+ match JSON files. Use cache after that.
    """
    return load_statsbomb_wc(season_ids=list(_SB_WC_SEASON_MAP.keys()))


def _sb_events(match_id):
    """Thin wrapper — centralises the sb.events() call."""
    from statsbombpy import sb
    return sb.events(match_id=match_id, split=False, flatten_attrs=True)


# ── Per-team StatsBomb stats ──────────────────────────────────────────────────

def team_statsbomb_stats(sb_matches=None, season_ids=None, cache_path=None):
    """
    Aggregate per-team per-match stats from StatsBomb event data:
      xg_for, xg_against, xg_diff
      passes_completed, final_third_entries
      shots_on_target, possession_pct (approximated via carries+passes)
      xt_accumulated, xt_conceded

    cache_path: if provided, save/load result as CSV to avoid re-fetching.
    Returns DataFrame indexed by (team, match_id).

    NOTE: First run fetches ~128 match JSON files (~10–20 min on slow net).
          Subsequent runs use cache_path if provided.

    Usage:
        sb_m = load_statsbomb_wc()
        stats = team_statsbomb_stats(sb_matches=sb_m, cache_path="data/sb_stats.csv")
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    if sb_matches is None:
        sb_matches = load_statsbomb_wc(season_ids=season_ids)

    rows = []
    for _, match in sb_matches.iterrows():
        mid   = match["match_id"]
        home  = match["home_team"]
        away  = match["away_team"]
        year  = match.get("wc_year", None)

        try:
            ev = _sb_events(mid)
        except Exception:
            continue

        for team, opp in [(home, away), (away, home)]:
            t_ev  = ev[ev["team"] == team]
            o_ev  = ev[ev["team"] == opp]

            # xG
            shots     = t_ev[t_ev["type"] == "Shot"]
            opp_shots = o_ev[o_ev["type"] == "Shot"]
            xg_for  = shots["shot_statsbomb_xg"].fillna(0).sum()
            xg_agst = opp_shots["shot_statsbomb_xg"].fillna(0).sum()

            # Shots on target: outcome in (Saved, Goal, Saved To Post)
            sot_outcomes = {"Saved", "Goal", "Saved To Post"}
            sot = shots[shots["shot_outcome"].isin(sot_outcomes)].shape[0]

            # Passes completed
            passes  = t_ev[t_ev["type"] == "Pass"]
            pass_ok = passes[passes["pass_outcome"].isna()].shape[0]  # NaN = completed

            # Final third entries — carries/passes that END in x > 80
            carries = t_ev[t_ev["type"] == "Carry"]
            def _end_x(r):
                loc = r.get("carry_end_location") or r.get("pass_end_location")
                if isinstance(loc, list): return loc[0]
                return None
            # vectorised via carry_end_location column if present
            fte = 0
            if "carry_end_location" in t_ev.columns:
                c_ends = carries["carry_end_location"].dropna()
                fte += int(c_ends.apply(lambda l: isinstance(l, list) and l[0] > 80).sum())
            if "pass_end_location" in t_ev.columns:
                p_ends = passes["pass_end_location"].dropna()
                fte += int(p_ends.apply(lambda l: isinstance(l, list) and l[0] > 80).sum())

            # Ball possession % — ratio of team's on-ball events to total
            on_ball_types = {"Pass", "Carry", "Shot", "Dribble", "Ball Receipt*"}
            total_on_ball = ev[ev["type"].isin(on_ball_types)].shape[0]
            team_on_ball  = t_ev[t_ev["type"].isin(on_ball_types)].shape[0]
            poss_pct = round(team_on_ball / total_on_ball * 100, 1) if total_on_ball > 0 else 0.0

            # xT accumulated — passes + carries that move ball into higher-value zones
            xt_acc = 0.0
            for _, row in t_ev[t_ev["type"].isin({"Pass", "Carry"})].iterrows():
                try:
                    sx, sy = row["location"]
                    if row["type"] == "Pass":
                        ex, ey = row["pass_end_location"]
                    else:
                        ex, ey = row["carry_end_location"]
                    delta = _xy_to_xt(ex, ey) - _xy_to_xt(sx, sy)
                    if delta > 0:
                        xt_acc += delta
                except (TypeError, KeyError, ValueError):
                    continue

            xt_conc = 0.0
            for _, row in o_ev[o_ev["type"].isin({"Pass", "Carry"})].iterrows():
                try:
                    sx, sy = row["location"]
                    if row["type"] == "Pass":
                        ex, ey = row["pass_end_location"]
                    else:
                        ex, ey = row["carry_end_location"]
                    delta = _xy_to_xt(ex, ey) - _xy_to_xt(sx, sy)
                    if delta > 0:
                        xt_conc += delta
                except (TypeError, KeyError, ValueError):
                    continue

            rows.append({
                "wc_year":          year,
                "match_id":         mid,
                "team":             team,
                "opponent":         opp,
                "xg_for":           round(float(xg_for), 3),
                "xg_against":       round(float(xg_agst), 3),
                "xg_diff":          round(float(xg_for - xg_agst), 3),
                "shots_on_target":  sot,
                "passes_completed": pass_ok,
                "final_third_entries": fte,
                "possession_pct":   poss_pct,
                "xt_accumulated":   round(xt_acc, 4),
                "xt_conceded":      round(xt_conc, 4),
                "xt_diff":          round(xt_acc - xt_conc, 4),
            })

    df = pd.DataFrame(rows)
    if cache_path:
        df.to_csv(cache_path, index=False)
    return df


def team_sb_summary(stats_df, team=None, wc_year=None):
    """
    Aggregate team_statsbomb_stats() per team (avg per match).
    team: filter to specific team. None = all teams.
    wc_year: 2018 or 2022. None = both.

    Usage:
        sb_m   = load_statsbomb_wc()
        stats  = team_statsbomb_stats(sb_m, cache_path="data/sb_stats.csv")
        brazil = team_sb_summary(stats, team="Brazil")
        all_22 = team_sb_summary(stats, wc_year=2022)
    """
    df = stats_df.copy()
    if team:
        df = df[df["team"] == team]
    if wc_year:
        df = df[df["wc_year"] == wc_year]
    if df.empty:
        return None
    num_cols = ["xg_for", "xg_against", "xg_diff", "shots_on_target",
                "passes_completed", "final_third_entries", "possession_pct",
                "xt_accumulated", "xt_conceded", "xt_diff"]
    return (
        df.groupby("team")[num_cols]
        .mean()
        .round(3)
        .sort_values("xg_for", ascending=False)
    )


# ── Rolling form + GD trend (martj42 extended) ───────────────────────────────

def rolling_win_pct(intl_df, team, windows=(5, 10, 20), competition_filter=None):
    """
    Rolling win % at n=5, 10, 20 last matches.
    competition_filter: partial string e.g. "World Cup" to restrict to WC only.

    Usage:
        intl = load_international()
        rolling_win_pct(intl, "Brazil")
        rolling_win_pct(intl, "France", competition_filter="World Cup")
    """
    df = _team_results(intl_df, team).sort_values("date").reset_index(drop=True)
    if competition_filter:
        df = df[df["tournament"].str.contains(competition_filter, case=False, na=False)]
    if df.empty:
        return {}

    df["win"] = (df["result"] == "W").astype(int)
    out = {"team": team, "total_matches": len(df), "all_time_win_pct": round(df["win"].mean(), 3)}
    for w in windows:
        if len(df) >= w:
            out[f"last_{w}_win_pct"] = round(df["win"].tail(w).mean(), 3)
    return out


def goal_diff_rolling(intl_df, team, window=10, competition_filter=None):
    """
    Recent goal difference trend: avg GD over last N matches + cumulative GD series.
    Useful as proxy for "form quality" beyond just win/loss.

    Returns dict with avg_gd, last_N_matches with scores, running_gd_series.

    Usage:
        goal_diff_rolling(intl, "Argentina", window=10)
        goal_diff_rolling(intl, "England", window=5, competition_filter="World Cup")
    """
    df = _team_results(intl_df, team).sort_values("date").reset_index(drop=True)
    if competition_filter:
        df = df[df["tournament"].str.contains(competition_filter, case=False, na=False)]
    if df.empty:
        return {}

    df["gd"] = df["goals_for"] - df["goals_against"]
    recent   = df.tail(window)
    avg_gd   = round(recent["gd"].mean(), 2)
    total_gd = int(recent["gd"].sum())

    last_matches = recent[["date", "opponent", "goals_for", "goals_against", "result", "tournament"]].to_dict("records")
    gd_series    = df[["date", "gd"]].tail(window * 2).to_dict("records")

    trend = "positive" if avg_gd > 0.3 else ("negative" if avg_gd < -0.3 else "neutral")
    return {
        "team": team, "window": window,
        "avg_gd": avg_gd, "total_gd": total_gd, "trend": trend,
        "last_matches": last_matches,
        "gd_series": gd_series,
        "narrative": f"{team} avg GD last {window} games: {avg_gd:+.2f} ({trend})",
    }


def multi_team_form_table(intl_df, teams, window=10, competition_filter=None):
    """
    Build a comparative form table across multiple teams.
    Returns DataFrame sorted by last_N_win_pct descending.

    Usage:
        teams = ["Brazil", "Argentina", "France", "England", "Spain"]
        form_table = multi_team_form_table(intl, teams, window=10)
        print(form_table)
    """
    rows = []
    for team in teams:
        rwp = rolling_win_pct(intl_df, team, windows=(window,), competition_filter=competition_filter)
        gdr = goal_diff_rolling(intl_df, team, window=window, competition_filter=competition_filter)
        if not rwp:
            continue
        rows.append({
            "team":         team,
            "all_time_win_pct": rwp.get("all_time_win_pct"),
            f"last_{window}_win_pct": rwp.get(f"last_{window}_win_pct"),
            "avg_gd":       gdr.get("avg_gd"),
            "gd_trend":     gdr.get("trend"),
        })
    return pd.DataFrame(rows).sort_values(f"last_{window}_win_pct", ascending=False).reset_index(drop=True)


# ── Player-level StatsBomb stats ──────────────────────────────────────────────

def player_statsbomb_stats(sb_matches=None, season_ids=None, cache_path=None):
    """
    Per-player aggregated stats across all WC matches in StatsBomb open data.
    Includes: goals, assists, shots, shots_on_target, passes_completed,
              carries_into_final_third, xg_total, xt_generated.

    cache_path: saves/loads result as CSV.

    Usage:
        sb_m    = load_statsbomb_wc()
        pstats  = player_statsbomb_stats(sb_m, cache_path="data/sb_player_stats.csv")
        pstats[pstats["player"] == "Kylian Mbappé"]
        pstats.sort_values("xg_total", ascending=False).head(20)
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    if sb_matches is None:
        sb_matches = load_statsbomb_wc(season_ids=season_ids)

    player_rows = []
    for _, match in sb_matches.iterrows():
        mid  = match["match_id"]
        year = match.get("wc_year", None)
        try:
            ev = _sb_events(mid)
        except Exception:
            continue

        for player in ev["player"].dropna().unique():
            p_ev = ev[ev["player"] == player]
            team = p_ev["team"].iloc[0] if not p_ev.empty else None

            shots   = p_ev[p_ev["type"] == "Shot"]
            passes  = p_ev[p_ev["type"] == "Pass"]
            carries = p_ev[p_ev["type"] == "Carry"]

            goals   = int(shots[shots["shot_outcome"] == "Goal"].shape[0])
            assists = int(p_ev[p_ev["pass_goal_assist"] == True].shape[0]) if "pass_goal_assist" in p_ev.columns else 0
            sot     = int(shots[shots["shot_outcome"].isin({"Saved", "Goal", "Saved To Post"})].shape[0])
            xg      = round(float(shots["shot_statsbomb_xg"].fillna(0).sum()), 4)
            pass_ok = int(passes[passes["pass_outcome"].isna()].shape[0])

            # carries into final third
            c_ft = 0
            if "carry_end_location" in carries.columns:
                c_ft = int(carries["carry_end_location"].dropna().apply(
                    lambda l: isinstance(l, list) and l[0] > 80
                ).sum())

            # xT generated (passes + carries)
            xt = 0.0
            for _, row in p_ev[p_ev["type"].isin({"Pass", "Carry"})].iterrows():
                try:
                    sx, sy = row["location"]
                    ex, ey = (row["pass_end_location"] if row["type"] == "Pass"
                              else row["carry_end_location"])
                    delta = _xy_to_xt(ex, ey) - _xy_to_xt(sx, sy)
                    if delta > 0:
                        xt += delta
                except (TypeError, KeyError, ValueError):
                    continue

            player_rows.append({
                "wc_year":              year,
                "match_id":             mid,
                "player":               player,
                "team":                 team,
                "goals":                goals,
                "assists":              assists,
                "shots_on_target":      sot,
                "xg_total":             xg,
                "passes_completed":     pass_ok,
                "carries_into_final_third": c_ft,
                "xt_generated":         round(xt, 4),
            })

    df = pd.DataFrame(player_rows)
    # Aggregate across all matches per player
    agg_cols = ["goals", "assists", "shots_on_target", "xg_total",
                "passes_completed", "carries_into_final_third", "xt_generated"]
    result = (
        df.groupby(["wc_year", "player", "team"])[agg_cols]
        .sum()
        .reset_index()
        .sort_values("xg_total", ascending=False)
    )
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result


# ── FBRef player stats via soccerdata ────────────────────────────────────────

def load_fbref_player_stats(
    competitions=None,
    seasons=None,
    stat_type="standard",
    cache_path=None,
):
    """
    Player season stats from FBRef via soccerdata.

    competitions: FBRef league codes. Default: WC + Euro (covers most active intl players).
      Full list: ["INT-World Cup", "INT-European Championship"]
      Other options: "INT-Women's World Cup", etc.
    seasons: list of years e.g. [2022, 2024]. Default: [2018, 2022, 2024].
    stat_type: "standard" | "shooting" | "passing" | "goal_shot_creation" | "defense" | "misc"
    cache_path: saves/loads result as CSV.

    Returns DataFrame with player, team, season, and per-stat columns.
    Columns vary by stat_type — "standard" gives: goals, assists, shots, xg, xag, prgc, prgp, prgr.

    Usage:
        # All WC 2022 player standard stats
        df = load_fbref_player_stats(competitions=["INT-World Cup"], seasons=[2022])

        # Recent Euro + WC for active players
        df = load_fbref_player_stats(seasons=[2021, 2022, 2024])
        df[df["player"] == "Kylian Mbappé"]
    """
    import soccerdata as sd

    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    if competitions is None:
        competitions = ["INT-World Cup", "INT-European Championship"]
    if seasons is None:
        seasons = [2018, 2022, 2024]

    frames = []
    for comp in competitions:
        for season in seasons:
            try:
                fb = sd.FBref(leagues=[comp], seasons=[season])
                df = fb.read_player_season_stats(stat_type=stat_type)
                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join(filter(None, c)).strip("_") for c in df.columns]
                df = df.reset_index()
                df["competition"] = comp
                df["season"] = season
                frames.append(df)
            except Exception as exc:
                print(f"FBref {comp} {season} failed: {exc}")
                continue

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    if cache_path:
        result.to_csv(cache_path, index=False)
    return result


def active_players_recent(goals_df=None, since_year=2022):
    """
    Players who scored in international games since since_year.
    Derived from martj42 goalscorers.csv — records goals only (not full appearances).

    Returns DataFrame: player, team, goals_since, last_goal_date.
    Sorted by goals_since descending.

    Usage:
        g = load_goalscorers()
        active = active_players_recent(g, since_year=2022)
        active[active["team"] == "Brazil"]
    """
    if goals_df is None:
        goals_df = load_goalscorers()

    cutoff = pd.Timestamp(f"{since_year}-01-01")
    recent = goals_df[
        (goals_df["date"] >= cutoff) & (~goals_df["own_goal"].fillna(False))
    ].copy()

    if recent.empty:
        return pd.DataFrame(columns=["player", "team", "goals_since", "last_goal_date"])

    agg = (
        recent.groupby(["scorer", "team"])
        .agg(goals_since=("scorer", "count"), last_goal_date=("date", "max"))
        .reset_index()
        .rename(columns={"scorer": "player"})
        .sort_values("goals_since", ascending=False)
        .reset_index(drop=True)
    )
    return agg


# locomotor data (TDnoPosmin, TD21posmin) — removed, no free public source for WC.
# SkillCorner open data: only 10 A-League matches, no WC coverage.
# FIFA TSG PDF: manual extract, not automated. Not worth maintaining a stub.

# ── StatsBomb 360 — spatial pressure and xT ──────────────────────────────────
#
# 360 data: freeze-frame player positions at every event (pass, shot, carry, tackle).
# NOT continuous tracking — snapshots only. Available for WC 2018 + 2022.
# Use: pressure-adjusted xG, open space at shot, defensive shape.
# Does NOT give running distances or HSR thresholds.

def _sb_frames(match_id):
    from statsbombpy import sb
    return sb.frames(match_id=match_id)


def shot_pressure_stats(match_id):
    """
    For each shot in a match: how much defensive pressure was the shooter under?
    Uses StatsBomb 360 freeze frames.

    Returns DataFrame per shot:
      player, team, minute, xg, shot_outcome,
      defenders_in_3m, defenders_in_5m, nearest_defender_m, open_goal

    Usage:
        df = shot_pressure_stats(3788741)   # WC 2022 final
        df.sort_values("xg", ascending=False)
    """
    ev     = _sb_events(match_id)
    frames = _sb_frames(match_id)

    shots  = ev[ev["type"] == "Shot"].copy()
    if shots.empty or frames.empty:
        return pd.DataFrame()

    rows = []
    for _, shot in shots.iterrows():
        ff = frames[frames["id"] == shot["id"]]
        loc = shot.get("location")
        if not isinstance(loc, list) or ff.empty:
            continue

        sx, sy = loc
        freeze = ff.iloc[0].get("freeze_frame", [])
        if not isinstance(freeze, list):
            continue

        opponents = [p for p in freeze if not p.get("teammate", True) and p.get("actor") is not True]
        dists = []
        for p in opponents:
            pl = p.get("location")
            if isinstance(pl, list) and len(pl) == 2:
                dists.append(np.sqrt((pl[0] - sx)**2 + (pl[1] - sy)**2))

        rows.append({
            "match_id":           match_id,
            "player":             shot.get("player"),
            "team":               shot.get("team"),
            "minute":             shot.get("minute"),
            "xg":                 round(float(shot.get("shot_statsbomb_xg") or 0), 4),
            "shot_outcome":       shot.get("shot_outcome"),
            "defenders_in_3m":    sum(d < 3 for d in dists),
            "defenders_in_5m":    sum(d < 5 for d in dists),
            "nearest_defender_m": round(min(dists), 2) if dists else None,
            "open_goal":          len(dists) == 0,
        })

    return pd.DataFrame(rows)


def team_shot_pressure_summary(sb_matches=None, season_ids=None, cache_path=None):
    """
    Aggregate shot pressure stats per team across WC matches.
    Returns: team, avg_xg_per_shot, avg_defenders_in_5m, avg_nearest_defender,
             open_goal_shots, total_shots.

    cache_path: saves/loads as CSV.

    Usage:
        sb_m = load_statsbomb_wc()
        df   = team_shot_pressure_summary(sb_m, cache_path="data/sb_shot_pressure.csv")
        df.sort_values("avg_xg_per_shot", ascending=False)
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    if sb_matches is None:
        sb_matches = load_statsbomb_wc(season_ids=season_ids)

    all_rows = []
    for _, match in sb_matches.iterrows():
        try:
            df = shot_pressure_stats(match["match_id"])
            if not df.empty:
                df["wc_year"] = match.get("wc_year")
                all_rows.append(df)
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()

    shots = pd.concat(all_rows, ignore_index=True)
    summary = (
        shots.groupby(["wc_year", "team"])
        .agg(
            total_shots        = ("xg", "count"),
            avg_xg_per_shot    = ("xg", "mean"),
            avg_defenders_in_5m= ("defenders_in_5m", "mean"),
            avg_nearest_def_m  = ("nearest_defender_m", "mean"),
            open_goal_shots    = ("open_goal", "sum"),
        )
        .round(3)
        .reset_index()
        .sort_values("avg_xg_per_shot", ascending=False)
    )
    if cache_path:
        summary.to_csv(cache_path, index=False)
    return summary


def spatial_xt_pressure(match_id):
    """
    xT per pass/carry weighted by defensive pressure at origin.
    Pressure = number of opponents within 5m at the moment of the action.

    Returns DataFrame: player, team, action_type, xt_delta, pressure_count.
    Higher pressure + positive xt_delta = high-value action under pressure.

    Usage:
        df = spatial_xt_pressure(3788741)
        df[df["team"] == "Argentina"].sort_values("xt_delta", ascending=False).head(20)
    """
    ev     = _sb_events(match_id)
    frames = _sb_frames(match_id)

    actions = ev[ev["type"].isin({"Pass", "Carry"})].copy()
    if actions.empty or frames.empty:
        return pd.DataFrame()

    rows = []
    for _, act in actions.iterrows():
        loc = act.get("location")
        if not isinstance(loc, list):
            continue
        sx, sy = loc

        if act["type"] == "Pass":
            end = act.get("pass_end_location")
        else:
            end = act.get("carry_end_location")
        if not isinstance(end, list):
            continue
        ex, ey = end

        xt_delta = _xy_to_xt(ex, ey) - _xy_to_xt(sx, sy)

        # Pressure from freeze frame if available
        ff = frames[frames["id"] == act["id"]]
        pressure = 0
        if not ff.empty:
            freeze = ff.iloc[0].get("freeze_frame", [])
            if isinstance(freeze, list):
                for p in freeze:
                    if p.get("teammate") is False:
                        pl = p.get("location")
                        if isinstance(pl, list):
                            d = np.sqrt((pl[0] - sx)**2 + (pl[1] - sy)**2)
                            if d < 5:
                                pressure += 1

        rows.append({
            "match_id":    match_id,
            "player":      act.get("player"),
            "team":        act.get("team"),
            "minute":      act.get("minute"),
            "action_type": act["type"],
            "xt_delta":    round(xt_delta, 5),
            "pressure":    pressure,
        })

    return pd.DataFrame(rows)


# ── Correlation to winning ────────────────────────────────────────────────────

def build_feature_matrix(ko_df, group_pts_df, elo_at_wc, coach_exp_df=None):  # noqa: E501
    """
    Merges datasets into one row per team per WC year.

    Outcome columns:
      group_points  — total points in group stage (3/2/1/0 per W/D/L)
      ko_score      — knockout stage score (0–6, see KNOCKOUT_SCORE)
      champion      — binary: 1 if won tournament
      reached_final — binary: 1 if played the Final (score >= 5)
      reached_semi  — binary: 1 if reached SF or beyond (score >= 4)

    Teams that didn't reach knockout phase get ko_score = 0 (filled from NaN).
    """
    # Base: all teams that participated (union of both score systems)
    df = ko_df.merge(group_pts_df, on=["year", "team"], how="outer")
    df["ko_score"] = df["ko_score"].fillna(0).astype(int)

    # ELO feature
    df = df.merge(elo_at_wc, on=["year", "team"], how="left")

    # Coach experience feature
    if coach_exp_df is not None:
        df = df.merge(
            coach_exp_df[["team_name", "year", "prior_wc_count"]].rename(
                columns={"team_name": "team"}
            ),
            on=["year", "team"],
            how="left",
        )

    # Derived binary outcomes
    df["champion"]      = (df["ko_score"] == 6).astype(int)
    df["reached_final"] = (df["ko_score"] >= 5).astype(int)
    df["reached_semi"]  = (df["ko_score"] >= 4).astype(int)
    df["reached_qf"]    = (df["ko_score"] >= 3).astype(int)
    return df

def correlation_to_winning(feature_matrix, outcome_col="reached_final"):
    """
    Pearson + Spearman correlation of every numeric feature vs outcome.
    Returns ranked DataFrame of correlations.
    """
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    outcome_cols = {"ko_score", "group_points", "champion", "reached_final",
                    "reached_semi", "reached_qf", "year"}
    numeric_cols = [c for c in numeric_cols if c not in outcome_cols]

    results = []
    for col in numeric_cols:
        sub = feature_matrix[[col, outcome_col]].dropna()
        if len(sub) < 10:
            continue
        pearson_r, pearson_p = stats.pearsonr(sub[col], sub[outcome_col])
        spearman_r, spearman_p = stats.spearmanr(sub[col], sub[outcome_col])
        results.append({
            "feature": col,
            "pearson_r": round(pearson_r, 3),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 3),
            "spearman_p": round(spearman_p, 4),
            "abs_pearson": abs(pearson_r),
            "significant": pearson_p < 0.05,
        })

    return pd.DataFrame(results).sort_values("abs_pearson", ascending=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WC Shorts Fact Generator")
    parser.add_argument("--matchup", type=str, help='e.g. "Brazil vs Argentina"')
    parser.add_argument("--team",    type=str, help="Single team analysis")
    parser.add_argument("--correlations", action="store_true",
                        help="Run correlation_to_winning() and print results")
    parser.add_argument("--sb-team-stats", action="store_true",
                        help="Fetch StatsBomb xG/xT/passes/possession/SOT per team (slow on first run)")
    parser.add_argument("--sb-player-stats", action="store_true",
                        help="Fetch StatsBomb per-player stats (goals, xG, xT, passes, final-third carries)")
    parser.add_argument("--form-table", type=str, default=None,
                        help='Comma-separated teams for form table e.g. "Brazil,Argentina,France"')
    parser.add_argument("--gd", type=str, default=None,
                        help="Goal diff trend for a team e.g. --gd Brazil")
    parser.add_argument("--wc-year", type=int, default=None,
                        help="Filter StatsBomb stats to specific WC year (e.g. 1990, 2018, 2022)")
    parser.add_argument("--all-wc-sb", action="store_true",
                        help="Load all 8 StatsBomb WC editions (1958–2022) for team stats")
    parser.add_argument("--recent-intl", action="store_true",
                        help="Show form table for WC 2026 teams, last 4 years only")
    parser.add_argument("--fbref-players", action="store_true",
                        help="Fetch FBRef player stats (WC + Euro, 2018/2022/2024)")
    parser.add_argument("--active-players", action="store_true",
                        help="List players who scored internationally since 2022")
    parser.add_argument("--xg-today", action="store_true",
                        help="xG table for today's WC 2026 matches (shots + odds)")
    parser.add_argument("--xg-date", type=str, default=None,
                        help="xG table for specific date e.g. --xg-date 20260614")
    parser.add_argument("--xg-match", type=str, default=None,
                        help="xG for specific ESPN game_id e.g. --xg-match 760420")
    parser.add_argument("--club-stats", action="store_true",
                        help="Fetch FBref Top 5 club stats last 4 seasons (standard)")
    parser.add_argument("--club-shooting", action="store_true",
                        help="Fetch FBref shooting stats (xG, npxG) last 4 seasons")
    parser.add_argument("--player-profile", type=str, default=None,
                        help="Club profile for a player e.g. --player-profile 'Vinicius'")
    parser.add_argument("--top", type=int, default=10, help="Show top N facts")
    args = parser.parse_args()

    intl    = load_international()
    matches = load_matches()
    goals   = load_goalscorers()

    if args.xg_today or args.xg_date:
        d = args.xg_date or None
        label = args.xg_date or "today"
        print(f"\n=== WC 2026 xG Table — {label} ===\n")
        df = wc2026_xg_table(d, method="both")
        if not df.empty:
            cols = [c for c in ["home_team","score","away_team","state",
                                 "home_xg_shots","away_xg_shots",
                                 "home_xg_odds","away_xg_odds",
                                 "home_win_prob","draw_prob","away_win_prob"] if c in df.columns]
            print(df[cols].to_string(index=False))

    if args.xg_match:
        stats = fetch_espn_match_stats(args.xg_match)
        xg_s  = match_xg_from_shots(stats["home_shots"], stats["home_sot"],
                                      stats["away_shots"], stats["away_sot"])
        print(f"\n=== xG — {stats['home_team']} vs {stats['away_team']} ===")
        print(f"  Shots method:  {stats['home_team']} {xg_s['home_xg']} | {stats['away_team']} {xg_s['away_xg']}")
        print(f"  Shots:         {stats['home_shots']} ({stats['home_sot']} SOT) | {stats['away_shots']} ({stats['away_sot']} SOT)")
        print(f"  Possession:    {stats['home_poss']}% | {stats['away_poss']}%")

    if args.club_stats or args.club_shooting:
        st = "shooting" if args.club_shooting else "standard"
        cache = str(DATA / f"fbref_club_{st}.csv")
        print(f"Fetching FBref club {st} stats (Top 5, 2022–2025)...")
        cdf = load_fbref_club_stats(stat_type=st, cache_path=cache)
        if args.team:
            team_col = next((c for c in cdf.columns if "team" in c.lower() or "squad" in c.lower()), None)
            if team_col:
                cdf = cdf[cdf[team_col].fillna("").str.lower().str.contains(args.team.lower())]
        print(cdf.head(args.top).to_string(index=False))

    if args.player_profile:
        cache = str(DATA / "fbref_club_standard.csv")
        print(f"Building club profile for '{args.player_profile}'...")
        profile = player_club_profile(args.player_profile, cache_path=cache)
        if "error" in profile:
            print(f"Error: {profile['error']}")
        else:
            print(f"\n=== {profile['player']} — Club Stats ===\n")
            print(profile["club_seasons"].to_string(index=False))
            print("\n--- Totals ---")
            for k, v in profile.get("summary", {}).items():
                print(f"  {k}: {v}")

    if args.recent_intl:
        recent = load_recent_internationals(intl, years=4)
        wc26_teams = [t for t in WC_2026_TEAMS if t in
                      set(recent["home_team"]) | set(recent["away_team"])]
        table = multi_team_form_table(recent, wc26_teams[:20], window=10)
        print("\n=== WC 2026 Teams — Form Last 4 Years ===\n")
        print(table.to_string(index=False))

    if args.active_players:
        active = active_players_recent(goals, since_year=2022)
        if args.team:
            active = active[active["team"] == args.team]
        print(f"\n=== Active International Scorers Since 2022 ===\n")
        print(active.head(args.top).to_string(index=False))

    if args.fbref_players:
        print("Fetching FBRef player stats (WC + Euro 2018/2022/2024)...")
        pf = load_fbref_player_stats(
            cache_path=str(DATA / "fbref_player_stats.csv")
        )
        if args.team:
            pf = pf[pf.get("team", pf.get("squad", pf.columns[0])) == args.team] \
                 if not pf.empty else pf
        print(pf.head(args.top).to_string(index=False))

    if args.all_wc_sb:
        print("Fetching ALL StatsBomb WC editions 1958–2022 (slow on first run)...")
        sid_list = list(_SB_WC_SEASON_MAP.keys())
        if args.wc_year:
            sid_list = [s for s, y in _SB_WC_SEASON_MAP.items() if y == args.wc_year]
        sb_m  = load_statsbomb_wc(season_ids=sid_list)
        stats = team_statsbomb_stats(sb_m, cache_path=str(DATA / "sb_stats_all_wc.csv"))
        summary = team_sb_summary(stats, team=args.team, wc_year=args.wc_year)
        if summary is not None:
            print(summary.to_string())
        else:
            print(stats.head(20).to_string())

    if args.sb_team_stats:
        print("Fetching StatsBomb WC event data (cached after first run)...")
        sb_m   = load_statsbomb_wc(season_ids=[3, 106] if not args.wc_year
                                   else ([3] if args.wc_year == 2018 else [106]))
        stats  = team_statsbomb_stats(sb_m, cache_path=str(DATA / "sb_stats.csv"))
        summary = team_sb_summary(stats, team=args.team, wc_year=args.wc_year)
        if summary is not None:
            print(summary.to_string())
        else:
            print(stats.head(20).to_string())

    if args.sb_player_stats:
        print("Fetching StatsBomb player event data (cached after first run)...")
        sb_m  = load_statsbomb_wc()
        pstat = player_statsbomb_stats(sb_m, cache_path=str(DATA / "sb_player_stats.csv"))
        if args.team:
            pstat = pstat[pstat["team"] == args.team]
        print(pstat.head(30).to_string(index=False))

    if args.form_table:
        teams_list = [t.strip() for t in args.form_table.split(",")]
        table = multi_team_form_table(intl, teams_list, window=10)
        print(f"\n=== Form Table (last 10 matches) ===\n")
        print(table.to_string(index=False))

    if args.gd:
        result = goal_diff_rolling(intl, args.gd, window=10)
        print(f"\n{result['narrative']}")
        for m in result.get("last_matches", []):
            print(f"  {m['date'].strftime('%Y-%m-%d') if hasattr(m['date'], 'strftime') else m['date']} "
                  f"vs {m['opponent']}: {m['goals_for']}–{m['goals_against']} ({m['result']})")

    if args.matchup:
        parts = [t.strip() for t in args.matchup.split("vs")]
        if len(parts) != 2:
            print("Format: --matchup 'Team A vs Team B'")
            return
        ko_df = knockout_stage_score(matches)
        facts = generate_matchup_facts(intl, matches, parts[0], parts[1], ko_df=ko_df, goals_df=goals)
        print(f"\n=== Facts: {parts[0]} vs {parts[1]} ===\n")
        for i, f in enumerate(facts[:args.top], 1):
            strength = f.get("strength", 0)
            print(f"[{strength:.2f}] {f['narrative']}")

    if args.team:
        ko_df = knockout_stage_score(matches)
        ksr   = knockout_stage_win_rate(ko_df, args.team)
        print(f"\n=== {args.team} — Knockout Advance Rates ===\n")
        for stage, data in ksr.items():
            print(f"  {stage}: {data['advanced']}/{data['appearances']} advanced "
                  f"({data['advance_rate']*100:.0f}%)")

    if args.correlations:
        elo_df      = load_elo()
        ko_df       = knockout_stage_score(matches)
        group_pts   = group_stage_points(matches)
        elo_wc      = elo_at_wc_start(elo_df)
        feature_matrix = build_feature_matrix(ko_df, group_pts, elo_wc)

        for outcome in ["champion", "reached_final", "reached_semi", "ko_score", "group_points"]:
            corr = correlation_to_winning(feature_matrix, outcome_col=outcome)
            print(f"\n=== Correlation → {outcome} ===\n")
            print(corr.head(8).to_string(index=False))

if __name__ == "__main__":
    main()
