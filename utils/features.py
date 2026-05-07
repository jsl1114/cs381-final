import pandas as pd
import numpy as np


def _long_form_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert matches to long form (one row per team per match).

    Expects columns: date, home_team, away_team, home_score, away_score
    """
    left = matches[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    # Home row
    home = pd.DataFrame({
        "date": left["date"],
        "team": left["home_team"],
        "opponent": left["away_team"],
        "goals_for": left["home_score"],
        "goals_against": left["away_score"],
        "is_home": 1,
    })

    # Away row
    away = pd.DataFrame({
        "date": left["date"],
        "team": left["away_team"],
        "opponent": left["home_team"],
        "goals_for": left["away_score"],
        "goals_against": left["home_score"],
        "is_home": 0,
    })

    long = pd.concat([home, away], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    return long


def compute_recent_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute recent-form features and attach them to matches DataFrame.

    Adds columns (home_ and away_ prefixes): lastN_wins, lastN_draws, lastN_losses,
    goals_lastN, conceded_lastN, recent_win_pct.

    This function is conservative and does not modify original match ordering.
    """
    m = matches.copy()
    long = _long_form_matches(m)

    def outcome_row(row):
        if row["goals_for"] > row["goals_against"]:
            return "win"
        elif row["goals_for"] < row["goals_against"]:
            return "loss"
        else:
            return "draw"

    long["outcome"] = long.apply(outcome_row, axis=1)

    # For each team compute rolling counts excluding current match (shift by 1)
    agg_list = []
    grouped = long.groupby("team", sort=False)
    for name, grp in grouped:
        grp = grp.sort_values("date").copy()
        grp["wins_cum"] = (grp["outcome"] == "win").cumsum()
        grp["draws_cum"] = (grp["outcome"] == "draw").cumsum()
        grp["losses_cum"] = (grp["outcome"] == "loss").cumsum()
        grp["goals_for_cum"] = grp["goals_for"].cumsum()
        grp["goals_against_cum"] = grp["goals_against"].cumsum()

        # To get last N, use rolling on a reversed index window via expanding then difference
        wins = (grp["outcome"] == "win").astype(int).rolling(window=window, min_periods=0).sum().shift(1)
        draws = (grp["outcome"] == "draw").astype(int).rolling(window=window, min_periods=0).sum().shift(1)
        losses = (grp["outcome"] == "loss").astype(int).rolling(window=window, min_periods=0).sum().shift(1)
        goals_for = grp["goals_for"].rolling(window=window, min_periods=0).sum().shift(1)
        goals_against = grp["goals_against"].rolling(window=window, min_periods=0).sum().shift(1)

        grp["lastN_wins"] = wins.fillna(0).astype(int)
        grp["lastN_draws"] = draws.fillna(0).astype(int)
        grp["lastN_losses"] = losses.fillna(0).astype(int)
        grp["goals_lastN"] = goals_for.fillna(0).astype(int)
        grp["conceded_lastN"] = goals_against.fillna(0).astype(int)
        grp["recent_win_pct"] = (grp["lastN_wins"] / window).fillna(0)

        agg_list.append(grp)

    long_feats = pd.concat(agg_list, ignore_index=True)

    # Pivot back to match-level: for each match date and teams, get the lastN features
    # Create a key to help merge: team,date,is_home
    long_feats["match_key"] = long_feats["team"] + "__" + long_feats["date"].astype(str) + "__" + long_feats["is_home"].astype(str)

    # Build mapping for home and away
    # Home key
    m = m.copy()
    m["home_key"] = m["home_team"] + "__" + m["date"].astype(str) + "__1"
    m["away_key"] = m["away_team"] + "__" + m["date"].astype(str) + "__0"

    feats = long_feats.set_index("match_key")[ ["lastN_wins", "lastN_draws", "lastN_losses", "goals_lastN", "conceded_lastN", "recent_win_pct"] ]

    # Helper to safe get
    def _get_feats_for_key(key):
        try:
            return feats.loc[key]
        except Exception:
            # return zeros
            return pd.Series({"lastN_wins":0, "lastN_draws":0, "lastN_losses":0, "goals_lastN":0, "conceded_lastN":0, "recent_win_pct":0})

    # Apply to matches
    for idx, row in m.iterrows():
        hk = row["home_key"]
        ak = row["away_key"]
        h = _get_feats_for_key(hk)
        a = _get_feats_for_key(ak)
        for c in ["lastN_wins", "lastN_draws", "lastN_losses", "goals_lastN", "conceded_lastN", "recent_win_pct"]:
            m.at[idx, f"home_{c}"] = int(h[c]) if c != "recent_win_pct" else float(h[c])
            m.at[idx, f"away_{c}"] = int(a[c]) if c != "recent_win_pct" else float(a[c])

    # Cast types
    for c in ["lastN_wins", "lastN_draws", "lastN_losses", "goals_lastN", "conceded_lastN"]:
        m[f"home_{c}"] = m[f"home_{c}"].fillna(0).astype(int)
        m[f"away_{c}"] = m[f"away_{c}"].fillna(0).astype(int)
    m["home_recent_win_pct"] = m["home_recent_win_pct"].fillna(0).astype(float)
    m["away_recent_win_pct"] = m["away_recent_win_pct"].fillna(0).astype(float)

    return m


def label_upset(row, gap_threshold: int = 10, prob_threshold: float = 0.35):
    """Label upset with a probabilistic proxy.

    Returns tuple: (upset_binary, upset_severity, p_home, p_away)
    """
    # Prefer using points if available
    home_points = row.get("home_points", None)
    away_points = row.get("away_points", None)
    home_rank = row.get("home_rank", None)
    away_rank = row.get("away_rank", None)

    p_home = None
    if pd.notna(home_points) and pd.notna(away_points):
        try:
            # Use a logistic-ish transform on points difference
            diff = float(away_points) - float(home_points)
            p_home = 1.0 / (1.0 + 10 ** (diff / 400.0))
        except Exception:
            p_home = None

    if p_home is None and pd.notna(home_rank) and pd.notna(away_rank):
        try:
            diff = float(away_rank) - float(home_rank)
            p_home = 1.0 / (1.0 + 10 ** (diff / 400.0))
        except Exception:
            p_home = 0.5

    if p_home is None:
        p_home = 0.5

    p_away = 1.0 - p_home

    winner = None
    if row.get("result") == "home_win":
        winner = "home"
    elif row.get("result") == "away_win":
        winner = "away"
    else:
        winner = None

    upset_binary = 0
    upset_severity = 0.0
    if winner == "home":
        prob_winner = p_home
        # lower-probability win
        if prob_winner < prob_threshold:
            upset_binary = 1
            upset_severity = 1.0 - prob_winner
        # rank gap rule
        elif pd.notna(row.get("rank_gap")) and row.get("rank_gap") >= gap_threshold:
            upset_binary = 1
            upset_severity = max(upset_severity, (row.get("rank_gap") / 100.0))
    elif winner == "away":
        prob_winner = p_away
        if prob_winner < prob_threshold:
            upset_binary = 1
            upset_severity = 1.0 - prob_winner
        elif pd.notna(row.get("rank_gap")) and row.get("rank_gap") >= gap_threshold:
            upset_binary = 1
            upset_severity = max(upset_severity, (row.get("rank_gap") / 100.0))

    return int(upset_binary), float(upset_severity), float(p_home), float(p_away)
