# World Cup Shorts — Project Plan

> Created: 2026-06-13 | Status: Active  
> Linked: [[new-angle]] | [[short-form-bilingual-strategy]] | [[content-production]]

---

## Objective

Produce 60-second data-narrative shorts (TikTok + IG Reels, EN + PT) for key World Cup 2026 matches.  
Format: voice-over (Boya BY-M1) over animated data visuals. No talking head.

---

## Data Infrastructure

### Required Downloads

Place all files in `data/` folder relative to this plan.

| File | Source | URL |
|---|---|---|
| `international_results.csv` | Kaggle (martj42) | kaggle.com/martj42/international-football-results-from-1872-to-2017 |
| `matches.csv` | GitHub (jfjelstul) | github.com/jfjelstul/worldcup → data-csv/ |
| `group_standings.csv` | GitHub (jfjelstul) | same repo |
| `squads.csv` | GitHub (jfjelstul) | same repo |
| `managers.csv` | GitHub (jfjelstul) | same repo |
| `manager_appearances.csv` | GitHub (jfjelstul) | same repo |
| `players.csv` | GitHub (jfjelstul) | same repo |
| `player_appearances.csv` | GitHub (jfjelstul) | same repo |
| `elo_ratings.csv` | Kaggle (afonsofernandescruz) | kaggle.com/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings |

### Club Stats (Last 4 Seasons)

| Stat | Source | Function | Notes |
|---|---|---|---|
| Goals, assists, shots, xG, xA | FBref (Top 5) | `load_fbref_club_stats()` | stat_type="standard" or "shooting" |
| Progressive carries (prgc) | FBref (Top 5) | `load_fbref_club_stats()` | Best free xT proxy (~r=0.78 with xT) |
| Key passes, pass completion | FBref (Top 5) | `load_fbref_club_stats(stat_type="passing")` | |
| Match-level xG | Understat | `load_understat_player_xg(player)` | Top 5 + Russian league |
| xT from club events | socceraction + WhoScored | `load_club_xt_note()` | ⚠️ broken Python 3.13 / NumPy 2.0 |

**xT workaround:** Use FBref `prgc` (progressive carries) as proxy. `prgc` = carries ≥10 yards forward toward goal. Covers same "threat progression" concept at no cost.

**socceraction status:** Broken on Python 3.13 due to `np.string_` removal in NumPy 2.0. Fix requires pinning Python 3.11 + NumPy 1.26 in a separate venv.

---

### WC 2026 Live xG (ESPN Public API)

No API key. Two approaches combined:

| Method | What it uses | When available |
|---|---|---|
| **Shots-based** | shots + shots on target from ESPN boxscore | Match started (in/post) |
| **Poisson/odds-based** | Moneyline odds → implied win probs → solve λ₁, λ₂ | Always (pre/live/post) |

```python
from pipeline import wc2026_xg_table, fetch_espn_match_stats, match_xg_from_odds

# xG for all today's matches
df = wc2026_xg_table()                          # both methods
df = wc2026_xg_table(method='odds')             # pre-match only
df = wc2026_xg_table('20260614', method='both') # specific date

# Shot stats + xG for one match (need game_id from fetch_espn_wc2026())
stats = fetch_espn_match_stats("760420")
# → home_shots, home_sot, away_shots, away_sot, possession, saves, corners

# Manual odds → xG (any match with moneyline)
xg = match_xg_from_odds(home_ml=-150, draw_ml=275, away_ml=450)
# → {'home_xg': 1.56, 'away_xg': 0.73, 'home_win_prob': 0.57, ...}
```

CLI:
```bash
python pipeline.py --xg-today               # today's matches
python pipeline.py --xg-date 20260614       # specific date
python pipeline.py --xg-match 760420        # one match by ESPN game_id
```

---

### Stat → Source Map

| Stat | Source | Available? | Pipeline function |
|---|---|---|---|
| Rolling win % | martj42 | ✅ | `rolling_win_pct()` |
| Recent goal difference | martj42 | ✅ | `goal_diff_rolling()` |
| Expected goals (xG) | StatsBomb open data | ✅ 2018 + 2022 WC | `team_statsbomb_stats()` → `xg_for / xg_against` |
| xT accumulation | StatsBomb (computed) | ✅ 2018 + 2022 WC | `team_statsbomb_stats()` → `xt_accumulated` |
| Passes completed | StatsBomb open data | ✅ 2018 + 2022 WC | `team_statsbomb_stats()` → `passes_completed` |
| Final third entries | StatsBomb open data | ✅ 2018 + 2022 WC | `team_statsbomb_stats()` → `final_third_entries` |
| Ball possession % | StatsBomb (computed) | ✅ 2018 + 2022 WC | `team_statsbomb_stats()` → `possession_pct` |
| Shots on target | StatsBomb open data | ✅ 2018 + 2022 WC | `team_statsbomb_stats()` → `shots_on_target` |
| Distance out-of-possession (TDnoPosmin) | FIFA TSG Report (PDF) | ⚠️ Manual extract | `load_locomotor_wc()` stub |
| HSR >21 km/h in-possession (TD21posmin) | FIFA TSG Report (PDF) | ⚠️ Manual extract | `load_locomotor_wc()` stub |
| Scout features (percentile rankings) | FBRef | ✅ via `soccerdata` | Use `soccerdata.FBref` |

**StatsBomb WC open data — ALL 8 editions:**
| season_id | Year |
|---|---|
| 269 | 1958 Sweden |
| 270 | 1962 Chile |
| 272 | 1970 Mexico |
| 51 | 1974 West Germany |
| 54 | 1986 Mexico |
| 55 | 1990 Italy |
| 3 | 2018 Russia |
| 106 | 2022 Qatar |
- Default `load_statsbomb_wc()` → 2018+2022. `load_statsbomb_all_wc()` → all 8.
- First full run downloads ~300+ match JSON files. Use `cache_path="data/sb_stats_all_wc.csv"`.
- `_SB_WC_SEASON_MAP` dict in pipeline.py maps all season_ids.

**FBRef scout features (via soccerdata):**
```python
# Via pipeline function (handles caching + multi-competition)
from pipeline import load_fbref_player_stats
df = load_fbref_player_stats(
    competitions=["INT-World Cup", "INT-European Championship"],
    seasons=[2018, 2022, 2024],
    stat_type="standard",   # "shooting" | "passing" | "defense" | "misc"
    cache_path="data/fbref_player_stats.csv",
)
```
Available `stat_type` values: `standard`, `shooting`, `passing`, `goal_shot_creation`, `defense`, `misc`.

**Locomotor (GPS/tracking) data:**
- NOT publicly available for WC matches.
- FIFA Technical Study Group report: https://digitalhub.fifa.com → search "Technical Report"
- Extract tables from PDF → save as `data/locomotor_wc.csv`
- Column names follow Castellano et al. 2026: `tdmin, tdnoposmin, tdposmin, td21min, td21posmin, td21noposmin, tdoffmin`

---

### What Each Dataset Gives You

**martj42 (49k+ international matches):**
- Head-to-head records between any two teams, WC only or all-time
- Win rate per team per tournament type
- Goal scoring patterns

**jfjelstul (27 tables, 1930–2018 WC):**
- Stage reached per team per edition → basis for `correlation_to_winning()`
- Squad composition, player appearances, manager experience
- Group standings with GF/GA/GD

**ELO ratings:**
- Continuous strength rating per team — join on year to get pre-tournament ELO
- Most predictive single feature for WC success (per Bayesian research)

---

## What I Need to Complete `correlation_to_winning()`

Before running the pipeline, I need you to check these column names after downloading:

### 1. ELO dataset (`elo_ratings.csv`)
Run `pd.read_csv("data/elo_ratings.csv").head(3)` and tell me:
- Column name for team name
- Column name for date / year
- Column name for ELO value

### 2. jfjelstul `matches.csv`
Run `pd.read_csv("data/matches.csv").columns.tolist()` and confirm:
- Column that identifies which team is being described (home/away vs team-oriented)
- Column for match stage (e.g. "stage", "round")
- Column for match outcome (score columns, or win/loss indicator)

### 3. jfjelstul `manager_appearances.csv`
Check if there's a column for WC year and `tournament_id` or `year` — needed to count coach prior WC appearances.

Once you share those, I'll complete `correlation_to_winning()` with exact joins.

---

## Content Formula

```
[0–4s]   HOOK    — surprising historical fact on screen + voice
[4–12s]  CONTEXT — why this pattern exists
[12–40s] DATA    — chart builds as you narrate
[40–52s] NOW     — "here's what this means for [team] in 2026"
[52–60s] CTA     — "I'll post the result after [match date]" (pre) | "full breakdown on YouTube" (post)
```

---

## Match Coverage Plan

| Round | Dates | Post? | Timing |
|---|---|---|---|
| Group stage — standard | Jun 12–26 | Only if giant killer | Reactive, within 1h |
| Round of 16 | Jun 28–Jul 4 | Yes | Pre: -24h / Post: +2h |
| Quarterfinals | Jul 5–8 | Yes | Pre + post |
| Semifinals | Jul 9–10 | Yes | Pre + post |
| Final | Jul 13 | Yes | Pre + during + post |

---

## Narrative Angles Bank

Seed list — add new ones as you run the pipeline.

| Angle type | Example |
|---|---|
| Drought | "Brazil: 24 years without a title — longest drought among 5-time champions" |
| Curse | "No defending champion has won back-to-back since Italy 1938. Argentina just won." |
| Home continent bias | "England has 0 WC wins outside home continent in 3 attempts" |
| Travel fatigue (2026 unique) | "2026 teams travel 40% more than Qatar — here's who's most exposed" |
| Coach effect | "First-time WC coaches win 23% fewer knockout games. 3 favorites have one." |
| H2H dominance | "Brazil leads WC head-to-head vs [X] — but lost the last 2" |
| Odds vs history gap | "Spain is +450 favorite. Favorites win only 18-22% of the time." |

---

## Production Workflow (per short)

1. Run `pipeline.py --matchup "TeamA vs TeamB"` → get ranked fact candidates
2. Pick top fact → write 60-sec voice script (HOOK → CONTEXT → DATA → NOW → CTA)
3. Build chart in matplotlib (30 min) or Manim (60 min)
4. Record EN voice-over → immediately record PT take (Boya → iPhone 15)
5. Assemble in CapCut: video + audio + auto-captions + export 9:16 1080×1920
6. Post EN to TikTok `@equalrightsblog` → Repurpose.io → IG Reels
7. Post PT to personal accounts (remove watermark first)

---

## Key Specialist Sources (for narrative context)

- [Nate Silver WC 2026 analysis](https://www.natesilver.net/p/world-cup-2026-odds-predictions)
- [Sportmonks data-driven predictions](https://www.sportmonks.com/blogs/world-cup-2026-predictions-who-is-going-to-win-based-on-football-data/)
- [StatsBomb blog](https://statsbomb.com/news/) — match analysis
- [Tifo Football](https://www.youtube.com/@TifoFootball) — team tactical context
- [FBRef scouting reports](https://fbref.com) — player percentile rankings vs peers
- [11v11.com](https://www.11v11.com) — deep historical match records
- [Transfermarkt](https://www.transfermarkt.com) — squad values + career stats

---

## CLI Commands

```bash
# Form table — last 10 matches for 5 teams
python pipeline.py --form-table "Brazil,Argentina,France,England,Spain"

# Form table — all WC 2026 teams, last 4 years only
python pipeline.py --recent-intl

# Goal difference trend for one team
python pipeline.py --gd "Morocco"

# StatsBomb xG/xT/passes/possession/SOT — 2018+2022 (default)
python pipeline.py --sb-team-stats
python pipeline.py --sb-team-stats --wc-year 2022 --team "France"

# StatsBomb ALL 8 WC editions (1958–2022) — slow first run
python pipeline.py --all-wc-sb
python pipeline.py --all-wc-sb --wc-year 1990 --team "Germany"

# Player stats via StatsBomb (WC event data)
python pipeline.py --sb-player-stats --wc-year 2022
python pipeline.py --sb-player-stats --team "Argentina"

# Player stats via FBRef (WC + Euro, 2018/2022/2024)
python pipeline.py --fbref-players
python pipeline.py --fbref-players --team "France"

# Players who scored internationally since 2022 (active pool)
python pipeline.py --active-players
python pipeline.py --active-players --team "Brazil"

# Club stats — Top 5 leagues, last 4 seasons (slow first run)
python pipeline.py --club-stats
python pipeline.py --club-stats --team "Real Madrid"

# Club shooting stats (xG, npxG per player)
python pipeline.py --club-shooting

# Player club profile — club seasons + totals
python pipeline.py --player-profile "Kylian Mbappé"
python pipeline.py --player-profile "Vinicius"

# Matchup facts (existing)
python pipeline.py --matchup "Brazil vs Argentina"
```

## Notebook — how to query by team or player

```python
from pipeline import *

intl  = load_international()
goals = load_goalscorers()

# ── Scope 1: All WC editions (StatsBomb event data 1958–2022) ─────────────────
sb_all = load_statsbomb_all_wc()   # all 8 editions
# or just 2018+2022:
sb_m   = load_statsbomb_wc()

stats_all = team_statsbomb_stats(sb_all, cache_path="data/sb_stats_all_wc.csv")

# Best xG per match across all WC editions
team_sb_summary(stats_all).sort_values("xg_for", ascending=False).head(10)

# France across all available WC editions
team_sb_summary(stats_all, team="France")

# Single edition
team_sb_summary(stats_all, wc_year=1990)

# ── Scope 2: All international games last 4 years (martj42 results) ───────────
recent = load_recent_internationals(intl, years=4)   # 2022–2026, all competitions

# Form for WC 2026 teams, last 4 years
wc26 = [t for t in WC_2026_TEAMS if t in set(recent["home_team"]) | set(recent["away_team"])]
form = multi_team_form_table(recent, wc26, window=10)

# Rolling win % for Morocco, recent internationals only
rolling_win_pct(recent, "Morocco", windows=(5, 10, 20))

# GD trend — qualifier matches only
goal_diff_rolling(recent, "Spain", window=10, competition_filter="Qualifier")

# ── Scope 3: Active WC 2026 players, last 4 years ─────────────────────────────
# Option A — martj42 goalscorers (goals only, all internationals since 2022)
active = active_players_recent(goals, since_year=2022)
active[active["team"] == "Brazil"]
active[active["player"].str.contains("Mbapp")]   # fuzzy search

# Option B — FBRef (WC + Euro 2022–2024, full stats per player)
fbref = load_fbref_player_stats(
    competitions=["INT-World Cup", "INT-European Championship"],
    seasons=[2021, 2022, 2024],
    stat_type="standard",
    cache_path="data/fbref_player_stats.csv",
)
# shooting stats
fbref_sh = load_fbref_player_stats(stat_type="shooting", seasons=[2022, 2024],
                                    cache_path="data/fbref_shooting.csv")

# Option C — StatsBomb WC 2022 player event data (xG, xT, passes, carries)
sb_m   = load_statsbomb_wc()
pstats = player_statsbomb_stats(sb_m, cache_path="data/sb_player_stats.csv")
pstats[pstats["wc_year"] == 2022].sort_values("xg_total", ascending=False).head(20)
pstats[pstats["player"] == "Kylian Mbappé"]
pstats.sort_values("xt_generated", ascending=False).head(20)
pstats.sort_values("carries_into_final_third", ascending=False).head(20)

# ── Scope 4: Club stats last 4 seasons (FBref + Understat) ───────────────────
# FBref club standard stats — goals, assists, xG, xA, prgc (xT proxy), prgp, prgr
club = load_fbref_club_stats(cache_path="data/fbref_club_standard.csv")

# Find any WC player by name
club[club["player"].str.contains("Mbapp", na=False)]

# Shooting stats — xG, npxG, shots on target
shooting = load_fbref_club_stats(stat_type="shooting", cache_path="data/fbref_club_shooting.csv")

# Full player club profile (club seasons + totals)
profile = player_club_profile("Vinicius", fbref_df=club)
profile["club_seasons"][["season_start", "squad", "goals", "assists", "xg", "prgc"]]

# Understat match-level xG for a player (async, requires network)
xg_history = load_understat_player_xg("Erling Haaland")
# Returns: season, goals, shots, xG, xA, key_passes, position, team

# ── Cross-scope: StatsBomb 2018+2022 team stats ───────────────────────────────
stats  = team_statsbomb_stats(sb_m, cache_path="data/sb_stats.csv")
brazil = stats[stats["team"] == "Brazil"]
rolling_win_pct(intl, "Brazil", windows=(5, 10, 20))
goal_diff_rolling(intl, "Argentina", window=10, competition_filter="World Cup")
```

## Files in This Project

```
worldcup-shorts/
├── plan.md              ← this file
├── pipeline.py          ← fact generator + correlation analysis
├── exploration.ipynb    ← dataset visualization notebook
└── data/
    ├── *.csv            ← downloaded datasets (gitignore)
    ├── sb_stats.csv          ← StatsBomb team stats 2018+2022 (auto-generated)
    ├── sb_stats_all_wc.csv   ← StatsBomb team stats all 8 WC editions
    ├── sb_player_stats.csv   ← StatsBomb player stats 2018+2022
    ├── fbref_player_stats.csv    ← FBRef intl player stats WC+Euro (auto-generated)
    ├── fbref_club_standard.csv   ← FBRef club standard stats Top 5 (auto-generated)
    ├── fbref_club_shooting.csv   ← FBRef club shooting stats (optional)
    └── locomotor_wc.csv          ← manual extract from FIFA TSG PDF (optional)
```
