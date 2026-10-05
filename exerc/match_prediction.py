
"""
match_predictor.py

Simple World Cup match predictor built to sit alongside pipeline.py and cohesion.py.

Usage:
    from match_predictor import predict_world_cup_match

    result = predict_world_cup_match(
        "Brazil",
        "Argentina",
        elo_df=elo_df
    )
"""

from scipy.stats import poisson


def _poisson_probs(home_xg: float, away_xg: float, max_goals: int = 10):
    home_win = draw = away_win = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)

            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

    return home_win, draw, away_win


def _expected_goals_from_elo(home_elo: float, away_elo: float):
    """
    Convert ELO difference into expected goals.
    Baseline WC match ≈ 1.35 goals/team.
    """
    elo_diff = home_elo - away_elo

    home_xg = max(0.2, 1.35 + elo_diff / 400.0)
    away_xg = max(0.2, 1.35 - elo_diff / 400.0)

    return round(home_xg, 3), round(away_xg, 3)


def predict_world_cup_match(
    team_a: str,
    team_b: str,
    elo_df,
    cohesion_a: float | None = None,
    cohesion_b: float | None = None,
):
    """
    Predict win probabilities and expected goals.

    Parameters
    ----------
    team_a, team_b
        Team names.

    elo_df
        DataFrame containing:
            team
            elo

        Typically output from load_elo().

    cohesion_a, cohesion_b
        Optional squad cohesion scores (0-8) from cohesion.py.
        Small adjustment applied to expected goals.

    Returns
    -------
    dict
    """

    row_a = elo_df[elo_df["team"] == team_a]
    row_b = elo_df[elo_df["team"] == team_b]

    if row_a.empty:
        raise ValueError(f"ELO not found for {team_a}")

    if row_b.empty:
        raise ValueError(f"ELO not found for {team_b}")

    elo_a = float(row_a.iloc[-1]["elo"])
    elo_b = float(row_b.iloc[-1]["elo"])

    xg_a, xg_b = _expected_goals_from_elo(elo_a, elo_b)

    if cohesion_a is not None:
        xg_a += (cohesion_a - 4.0) * 0.05

    if cohesion_b is not None:
        xg_b += (cohesion_b - 4.0) * 0.05

    xg_a = max(0.2, xg_a)
    xg_b = max(0.2, xg_b)

    win_a, draw, win_b = _poisson_probs(xg_a, xg_b)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "expected_goals": {
            team_a: round(xg_a, 2),
            team_b: round(xg_b, 2),
        },
        "win_probability": {
            team_a: round(win_a, 4),
            "draw": round(draw, 4),
            team_b: round(win_b, 4),
        },
        "elo": {
            team_a: round(elo_a, 1),
            team_b: round(elo_b, 1),
        },
    }