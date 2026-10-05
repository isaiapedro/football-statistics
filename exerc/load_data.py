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


# FBref club scraping via soccerdata is blocked by Cloudflare on headless Chrome.
# Replaced by Understat-based functions (no browser required).
# Manual workaround: download CSV from fbref.com directly in a browser and pass
# as cache_path — load_fbref_club_stats() will read it without scraping.

def load_fbref_club_stats(cache_path=None, **_kwargs):
    """
    FBref club scraping blocked (Cloudflare). Two options:
      1. Pass cache_path pointing to a CSV you manually downloaded from fbref.com
      2. Use load_understat_team_stats() for automated xG/goals data

    Manual download steps:
      fbref.com → Big 5 Stats → Squad Standard Stats → Share & Export → CSV
      Save as data/fbref_club_standard.csv, then call:
        load_fbref_club_stats(cache_path='data/fbref_club_standard.csv')
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)
    print(
        "FBref blocked (Cloudflare). Options:\n"
        "  1. Manual: download from fbref.com → save CSV → pass cache_path\n"
        "  2. Automated: use load_understat_team_stats() for xG data"
    )
    return pd.DataFrame()


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


def load_understat_team_stats(league=None, seasons=None, cache_path=None):
    """
    Team-level xG/goals from Understat for Top 5 leagues.
    Automated alternative to load_fbref_club_stats() (which is Cloudflare-blocked).

    league: Understat league key. None = all leagues in _UNDERSTAT_LEAGUES.
    seasons: list of year strings e.g. ["2022","2023","2024","2025"]. Default: last 4.
    cache_path: save/load as CSV.

    Returns DataFrame: league, season, team, scored, missed, xG, xGA, wins, draws, loses, pts.

    Usage:
        df = load_understat_team_stats()
        df[df["team"] == "Manchester City"]
        df.groupby(["team","season"])[["xG","xGA"]].mean()
    """
    import asyncio
    from understat import Understat

    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    if seasons is None:
        seasons = ["2022", "2023", "2024", "2025"]

    leagues_to_fetch = {league: _UNDERSTAT_LEAGUES[league]} if league else _UNDERSTAT_LEAGUES

    async def _fetch():
        frames = []
        async with Understat() as u:
            for league_key, ustat_name in leagues_to_fetch.items():
                for season in seasons:
                    try:
                        teams = await u.get_league_table(ustat_name, int(season))
                        df = pd.DataFrame(teams)
                        df["league"] = league_key
                        df["season"] = season
                        frames.append(df)
                    except Exception as exc:
                        print(f"Understat {league_key} {season}: {exc}")
                        continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    try:
        result = asyncio.run(_fetch())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_fetch())

    if not result.empty and cache_path:
        result.to_csv(cache_path, index=False)
    return result


def player_club_profile(player_name, fbref_df=None, cache_path=None):
    """
    Club stats for a player across last 4 seasons.
    Primary: Understat xG (automated). Fallback: FBref CSV (manual download).

    fbref_df: pre-loaded FBref DataFrame. If None + cache_path exists, loads from CSV.
    If neither available, falls back to Understat only.

    Returns dict:
      - player: name
      - understat_xg: DataFrame (season, goals, shots, xG, xA, ...)
      - club_seasons: DataFrame from FBref if available, else empty
      - summary: understat aggregated totals

    Usage:
        profile = player_club_profile("Kylian Mbappé")
        profile["understat_xg"]   # season xG (always available)
        profile["club_seasons"]   # FBref data (only if cache_path CSV exists)
    """
    # Always try Understat (automated, no scraping)
    try:
        ust_df = load_understat_player_xg(player_name)
    except Exception as exc:
        print(f"Understat lookup failed for '{player_name}': {exc}")
        ust_df = pd.DataFrame()

    # FBref: only if manual cache CSV present
    club_df = pd.DataFrame()
    if fbref_df is None and cache_path and Path(cache_path).exists():
        fbref_df = pd.read_csv(cache_path)

    if fbref_df is not None and not fbref_df.empty:
        player_col = next((c for c in fbref_df.columns if "player" in c.lower()), None)
        if player_col:
            mask = fbref_df[player_col].fillna("").str.lower().str.contains(player_name.lower())
            club_df = fbref_df[mask].copy()

    # Summary from Understat
    summary: dict = {}
    if not ust_df.empty:
        num_cols = ust_df.select_dtypes(include=[np.number]).columns.tolist()
        summary = ust_df[num_cols].sum().round(3).to_dict() if num_cols else {}

    return {
        "player": player_name,
        "understat_xg": ust_df,
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

def load_fbref_player_stats(cache_path=None, **_kwargs):
    """
    FBref player stats (international tournaments) — Cloudflare-blocked.

    Manual workaround:
      fbref.com → World Cup → Player Standard Stats → Share & Export → CSV
      Save as data/fbref_player_stats.csv, then:
        load_fbref_player_stats(cache_path='data/fbref_player_stats.csv')

    For WC player xG use load_statsbomb_player_stats() (fully automated).
    For club xG use load_understat_player_xg(player_name).
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)
    print(
        "FBref blocked (Cloudflare). Manual download required.\n"
        "  fbref.com → World Cup → Player Stats → Export CSV → save as data/fbref_player_stats.csv\n"
        "  Alternative: use load_statsbomb_player_stats() for WC xG data."
    )
    return pd.DataFrame()


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

