from __future__ import annotations
import numpy as np
import pandas as pd

def implied_prob(decimal_odds: float) -> float:
    if decimal_odds is None or pd.isna(decimal_odds) or decimal_odds <= 1e-9:
        return np.nan
    return 1.0 / decimal_odds

def ev_decimal(p: float, odds: float) -> float:
    # Expected value per 1 unit stake using decimal odds
    return p * (odds - 1.0) - (1.0 - p)

def kelly_fraction(p: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * (b + 1.0) - 1.0) / b
    return float(max(0.0, f))

def build_value_bets(odds_best: pd.DataFrame, model: pd.DataFrame, bankroll: float, kelly_mult: float, min_edge: float) -> pd.DataFrame:
    if odds_best.empty or model.empty:
        return pd.DataFrame()

    # Merge on fixture identity
    key_cols = ["competition","commence_time","home_team","away_team"]
    m = model[key_cols + ["p_home","p_draw","p_away","p_btts_yes","p_over_2_5","exp_cards","exp_fouls"]].copy()
    o = odds_best.copy()

    # Map odds markets to model probabilities
    def pick_prob(row):
        mk = row["market"]
        sel = str(row["selection"]).lower()
        pt = row.get("point")
        if mk == "h2h":
            if sel == str(row["home_team"]).lower(): return row.get("_p_home")
            if sel == "draw": return row.get("_p_draw")
            if sel == str(row["away_team"]).lower(): return row.get("_p_away")
        if mk == "btts":
            if sel in ("yes","y"): return row.get("_p_btts_yes")
            if sel in ("no","n"): return 1.0 - row.get("_p_btts_yes")
        if mk == "totals":
            # outcomes usually named Over/Under, with point line
            if pt is None or pd.isna(pt): return np.nan
            if sel == "over":
                if float(pt) == 2.5: return row.get("_p_over_2_5")
                # approximate using Poisson totals by scaling from 2.5 (rough)
                # keep simple: fallback nan for other lines unless you extend
                return np.nan
            if sel == "under":
                if float(pt) == 2.5: return 1.0 - row.get("_p_over_2_5")
                return np.nan
        if mk == "alternate_totals_cards":
            if pt is None or pd.isna(pt) or row.get("_exp_cards") is None: return np.nan
            mean = row.get("_exp_cards")
            if pd.isna(mean): return np.nan
            # normal approx
            from .model import total_over_prob
            if sel == "over": return total_over_prob(float(mean), float(pt))
            if sel == "under": return 1.0 - total_over_prob(float(mean), float(pt))
        return np.nan

    merged = o.merge(m, on=key_cols, how="left", suffixes=("","_m"))
    # stash fixture team names for home/away comparisons
    merged["_p_home"] = merged["p_home"]; merged["_p_draw"]=merged["p_draw"]; merged["_p_away"]=merged["p_away"]
    merged["_p_btts_yes"] = merged["p_btts_yes"]; merged["_p_over_2_5"]=merged["p_over_2_5"]
    merged["_exp_cards"] = merged["exp_cards"]; merged["_exp_fouls"]=merged["exp_fouls"]

    merged["model_p"] = merged.apply(pick_prob, axis=1)
    merged = merged.dropna(subset=["model_p","price"])
    merged["implied_p"] = merged["price"].apply(implied_prob)
    merged["edge"] = merged["model_p"] - merged["implied_p"]
    merged["ev"] = merged.apply(lambda r: ev_decimal(float(r["model_p"]), float(r["price"])), axis=1)
    merged["kelly_raw"] = merged.apply(lambda r: kelly_fraction(float(r["model_p"]), float(r["price"])), axis=1)
    merged["stake"] = (bankroll * kelly_mult * merged["kelly_raw"]).clip(lower=0)

    # filters
    merged = merged[(merged["edge"] >= min_edge) & (merged["stake"] > 0)]
    merged = merged.sort_values(["competition","commence_time","edge","ev"], ascending=[True, True, False, False])

    keep = key_cols + ["market","selection","point","price","model_p","implied_p","edge","ev","stake","bookmaker"]
    for c in keep:
        if c not in merged.columns:
            merged[c] = pd.NA
    return merged[keep]
