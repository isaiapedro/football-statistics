Loaders

  from pipeline import *

  # ── International results (martj42, all-time) ─────────────────────────────────
  intl   = load_international()          # all matches 1872–now
  goals  = load_goalscorers()            # all goals 1872–now
  recent = load_recent_internationals(intl, years=4)  # 2022–2026 only

  # ── WC structural data (jfjelstul, 1930–2018) ─────────────────────────────────
  matches  = load_matches()              # requires data/matches.csv
  standings = load_group_standings()     # requires data/group_standings.csv
  mgr_apps  = load_manager_appearances() # requires data/manager_appearances.csv

  # ── ELO ───────────────────────────────────────────────────────────────────────
  elo = load_elo()                       # Kaggle auto-download

  # ── StatsBomb event data ───────────────────────────────────────────────────────
  sb_m     = load_statsbomb_wc()                    # 2018 + 2022 (default)
  sb_all   = load_statsbomb_all_wc()                # all 8: 1958/62/70/74/86/90/2018/22
  sb_1990  = load_statsbomb_wc(season_ids=[55])     # single edition

  # ── FBref (international tournaments) ─────────────────────────────────────────
  fbref_intl = load_fbref_player_stats(
      competitions=["INT-World Cup", "INT-European Championship"],
      seasons=[2018, 2022, 2024],
      stat_type="standard",              # "shooting" | "passing" | "defense" | "misc"
      cache_path="data/fbref_intl.csv",
  )

  # ── FBref (club, Top 5 leagues) ───────────────────────────────────────────────
  fbref_club = load_fbref_club_stats(
      stat_type="standard",              # "shooting" | "passing" | "defense"
      seasons=[2022, 2023, 2024, 2025],
      cache_path="data/fbref_club_standard.csv",
  )
  fbref_shoot = load_fbref_club_stats(stat_type="shooting", cache_path="data/fbref_club_shooting.csv")

  # ── Active players since 2022 (martj42 goalscorers) ───────────────────────────
  active = active_players_recent(goals, since_year=2022)

  # ── WC 2026 live (ESPN, no key) ───────────────────────────────────────────────
  today     = fetch_espn_wc2026()                   # today's matches
  june14    = fetch_espn_wc2026("20260614")          # specific date
  xg_table  = wc2026_xg_table()                     # xG for today (shots + odds)
  xg_table  = wc2026_xg_table("20260614", method="odds")  # pre-match only
  match_stats = fetch_espn_match_stats("760420")    # shots/SOT/poss for one match

  ---
  Aggregations & Analysis

  # ── Team WC stats (StatsBomb) ─────────────────────────────────────────────────
  stats   = team_statsbomb_stats(sb_m, cache_path="data/sb_stats.csv")
  # cols: team, match_id, wc_year, xg_for, xg_against, xg_diff,
  #       shots_on_target, passes_completed, final_third_entries,
  #       possession_pct, xt_accumulated, xt_conceded, xt_diff

  summary = team_sb_summary(stats)                   # avg per match, all teams
  summary = team_sb_summary(stats, team="Brazil")    # one team
  summary = team_sb_summary(stats, wc_year=2022)     # one edition

  # ── Player WC stats (StatsBomb) ───────────────────────────────────────────────
  pstats  = player_statsbomb_stats(sb_m, cache_path="data/sb_player_stats.csv")
  # cols: wc_year, player, team, goals, assists, shots_on_target,
  #       xg_total, passes_completed, carries_into_final_third, xt_generated

  # ── Shot pressure / 360 (StatsBomb) ──────────────────────────────────────────
  pressure = shot_pressure_stats(3788741)
  # cols: player, team, minute, xg, shot_outcome,
  #       defenders_in_3m, defenders_in_5m, nearest_defender_m, open_goal

  pressure_summary = team_shot_pressure_summary(sb_m, cache_path="data/sb_shot_pressure.csv")
  # cols: wc_year, team, total_shots, avg_xg_per_shot,
  #       avg_defenders_in_5m, avg_nearest_def_m, open_goal_shots

  spatial = spatial_xt_pressure(3788741)
  # cols: player, team, minute, action_type, xt_delta, pressure (defenders in 5m)

  # ── Rolling form (martj42) ────────────────────────────────────────────────────
  rolling_win_pct(intl, "Brazil", windows=(5, 10, 20))
  rolling_win_pct(recent, "Morocco", windows=(5, 10))         # last 4 years only
  goal_diff_rolling(intl, "Argentina", window=10)
  goal_diff_rolling(intl, "England", competition_filter="World Cup")

  multi_team_form_table(intl, ["Brazil","France","Spain","Argentina"], window=10)
  multi_team_form_table(recent, WC_2026_TEAMS[:20], window=10) # recent only

  # ── Match xG (manual odds input) ─────────────────────────────────────────────
  match_xg_from_odds(-150, 275, 450)    # home_ml, draw_ml, away_ml
  match_xg_from_shots(9, 4, 2, 1)      # home_shots, home_sot, away_shots, away_sot

  # ── Player club profile ───────────────────────────────────────────────────────
  player_club_profile("Kylian Mbappé", fbref_df=fbref_club)
  active[active["team"] == "Brazil"]

  # ── Historical match facts ────────────────────────────────────────────────────
  ko_df   = knockout_stage_score(matches)
  gp_df   = group_stage_points(matches)
  elo_wc  = elo_at_wc_start(elo)

  head_to_head_wc(intl, "Brazil", "Argentina")
  all_time_h2h(intl, "France", "Morocco")
  generate_matchup_facts(intl, matches, "Brazil", "France", ko_df=ko_df, goals_df=goals)
  best_wc_finish(ko_df, "Morocco")
  wc_appearances(matches, "Japan")
  form_trajectory(intl, "Spain", n=15)
  continental_record(intl, "Senegal")

  ---
  CLI

  # WC 2026 live xG
  python pipeline.py --xg-today
  python pipeline.py --xg-date 20260614
  python pipeline.py --xg-match 760420

  # Form & GD
  python pipeline.py --form-table "Brazil,Argentina,France,England,Spain"
  python pipeline.py --recent-intl          # all WC 2026 teams, last 4 years
  python pipeline.py --gd "Morocco"

  # StatsBomb WC stats
  python pipeline.py --sb-team-stats                         # 2018 + 2022
  python pipeline.py --sb-team-stats --wc-year 2022 --team "France"
  python pipeline.py --all-wc-sb                             # all 8 editions
  python pipeline.py --sb-player-stats --wc-year 2022

  # FBref
  python pipeline.py --fbref-players        # WC + Euro 2018/22/24
  python pipeline.py --club-stats           # Top 5 clubs, 4 seasons
  python pipeline.py --club-shooting
  python pipeline.py --player-profile "Vinicius"

  # Active players
  python pipeline.py --active-players
  python pipeline.py --active-players --team "Brazil"

  # Historical matchup
  python pipeline.py --matchup "Brazil vs Argentina"
  python pipeline.py --team "Germany"
  python pipeline.py --correlations
