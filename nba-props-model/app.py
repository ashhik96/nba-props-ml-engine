import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime

# make sure we can import from utils/
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.data_fetcher import (
    get_player_id,
    get_opponent_recent_games,
    get_head_to_head_history,
    get_player_position,
    get_team_defense_rank_vs_position,
    get_players_by_team,
    get_upcoming_games,
    fetch_fanduel_lines,
    get_event_id_for_game,
    get_player_fanduel_line,
)

from utils.cached_data_fetcher import (
    get_player_game_logs_cached_db,
    get_team_stats_cached_db,
    scrape_defense_vs_position_cached_db,
)

from utils.database import get_cache_stats, clear_old_seasons

from utils.features import build_enhanced_feature_vector
from utils.model import PlayerPropModel


# ─────────────────────────────
# Page config
# ─────────────────────────────
st.set_page_config(
    page_title="NBA Player Props Projection Model",
    page_icon="🏀",
    layout="wide",
)

# GLOBAL store for custom user lines
if "custom_lines" not in st.session_state:
    st.session_state["custom_lines"] = {}


# ─────────────────────────────
# Season helpers
# ─────────────────────────────
def get_current_nba_season():
    now = datetime.now()
    yr = now.year
    mo = now.month
    if mo >= 10:
        return f"{yr}-{str(yr+1)[2:]}"
    else:
        return f"{yr-1}-{str(yr)[2:]}"


def get_prior_nba_season():
    cur = get_current_nba_season()
    start_year = int(cur.split('-')[0])
    return f"{start_year-1}-{str(start_year)[2:]}"


current_season = get_current_nba_season()
prior_season = get_prior_nba_season()


# ─────────────────────────────
# Model
# ─────────────────────────────
@st.cache_resource
def load_model():
    return PlayerPropModel(alpha=1.0)

model = load_model()


# ─────────────────────────────
# Stat dropdown options
# ─────────────────────────────
STAT_OPTIONS = {
    "Points": "PTS",
    "Assists": "AST",
    "Rebounds": "REB",
    "Three-Pointers Made": "FG3M",
    "Points + Rebounds + Assists (PRA)": "PRA",
    "Double-Double Probability": "DD",
}


# ─────────────────────────────
# Helper functions
# ─────────────────────────────
def defense_emoji(rank_num: int) -> str:
    if rank_num <= 10:
        return "🔴"
    elif rank_num <= 20:
        return "🟡"
    else:
        return "🟢"


def calc_hit_rate(game_logs: pd.DataFrame, stat_col: str, line_value: float, window: int = 10):
    if game_logs is None or game_logs.empty:
        return None
    if line_value is None:
        return None

    recent = game_logs.head(window).copy()

    if stat_col == "PRA":
        if not {"PTS", "REB", "AST"}.issubset(recent.columns):
            return None
        recent_vals = recent["PTS"] + recent["REB"] + recent["AST"]
    else:
        if stat_col not in recent.columns:
            return None
        recent_vals = recent[stat_col]

    if len(recent_vals) == 0:
        return None

    hits = (recent_vals > line_value).sum()
    rate = hits / len(recent_vals)
    return rate * 100.0


def calc_edge(prediction: float, line_value: float):
    if line_value is None:
        return ("—", "No line", "—")

    if line_value == 0:
        diff = prediction
        pct = 0.0
    else:
        diff = prediction - line_value
        pct = (diff / line_value) * 100.0

    if abs(diff) < 1.5:
        rec_text = "⚪ No clear edge"
        ou_short = "—"
    elif diff > 1.5:
        rec_text = "✅ OVER looks good"
        ou_short = "OVER"
    else:
        rec_text = "❌ UNDER looks good"
        ou_short = "UNDER"

    edge_str = f"{diff:+.1f} ({pct:+.1f}%)"
    return (edge_str, rec_text, ou_short)


# ─────────────────────────────
# Player Detail Panel (expander body)
# ─────────────────────────────
def render_player_detail_body(pdata, cur_season, prev_season, unique_suffix="0_0"):
    player_name = pdata["player_name"]
    team_abbrev = pdata["team_abbrev"]
    player_pos = pdata["player_pos"]
    opponent_abbrev = pdata["opponent_abbrev"]

    current_logs = pdata["current_logs"]
    prior_logs = pdata["prior_logs"]
    h2h_history = pdata["h2h_history"]
    opp_def_rank = pdata["opp_def_rank"]
    features = pdata["features"]

    prediction = pdata["prediction"]
    stat_code = pdata["stat_code"]
    stat_display = pdata["stat_display"]

    fd_line_val = pdata["fd_line_val"]
    base_line_val = pdata["base_line_val"]

    override_key = f"{player_name}|{team_abbrev}|{stat_code}"
    active_line_val = st.session_state["custom_lines"].get(
        override_key,
        base_line_val
    )

    has_current = not current_logs.empty
    has_prior = not prior_logs.empty
    current_games = len(current_logs) if has_current else 0
    prior_games = len(prior_logs) if has_prior else 0
    h2h_games = 0 if h2h_history is None or h2h_history.empty else len(h2h_history)

    st.subheader(f"📊 Projections for {player_name} ↩")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(f"{cur_season} Games", current_games)
    with colB:
        st.metric(f"{prev_season} Games", prior_games)
    with colC:
        st.metric(f"vs {opponent_abbrev} History", h2h_games)

    if current_games < 5:
        st.info(
            f"Only {current_games} games in {cur_season}. "
            f"We're leaning more on {prev_season} + H2H."
        )

    # Opponent matchup context
    st.markdown("---")
    st.subheader(
        f"{opponent_abbrev} Defense vs {player_pos}s "
        f"({ 'PG/SG' if player_pos=='G' else 'SF/PF' if player_pos=='F' else 'C' })"
    )

    col1, col2, col3 = st.columns(3)

    rank_val = opp_def_rank.get("rank", 15)
    rating_text = opp_def_rank.get("rating", "Average")
    percentile = opp_def_rank.get("percentile", 50.0)

    rating_lower = str(rating_text).lower()
    if "elite" in rating_lower or "above" in rating_lower:
        diff_emoji = "🔴"
    elif "average" in rating_lower and "above" not in rating_lower:
        diff_emoji = "🟡"
    else:
        diff_emoji = "🟢"

    with col1:
        st.metric(
            "Defensive Rank vs Position",
            f"{diff_emoji} #{rank_val} of 30",
            help=f"How {opponent_abbrev} guards this archetype ({player_pos})"
        )
    with col2:
        st.metric(
            "Matchup Difficulty",
            rating_text,
            help="Elite / Above Avg = tough. Below Avg = soft / target spot."
        )
    with col3:
        st.metric(
            "Defense Percentile",
            f"{percentile:.0f}%",
            help="Higher percentile = stronger defense overall."
        )

    if "elite" in rating_lower or "above" in rating_lower:
        st.info(
            f"🔴 Tough matchup: {opponent_abbrev} defends {player_pos} well. "
            "Unders / caution."
        )
    elif "below" in rating_lower:
        st.success(
            f"🟢 Favorable matchup: {opponent_abbrev} struggles vs {player_pos}. "
            "Overs become more viable."
        )

    # Projection / Recency / Context
    st.markdown("---")
    colP, colR, colCxt = st.columns([2, 2, 1])

    with colP:
        st.subheader("🎯 Model Projection")
        if stat_code == "DD":
            st.metric("Double-Double Probability", f"{prediction:.1f}%")
        else:
            st.metric(f"Projected {stat_display}", f"{prediction:.1f}")

        if stat_code != "DD":
            if opp_def_rank and "pts_allowed" in opp_def_rank:
                st.caption(
                    f"🛡️ Opp vs {player_pos}: "
                    f"{opp_def_rank['pts_allowed']:.1f} pts allowed"
                )
            else:
                st.caption("🛡️ Opponent defense data unavailable")

            if h2h_games > 0:
                h2h_avg = features.get(f"h2h_{stat_code}_avg", 0)
                st.caption(
                    f"📊 vs {opponent_abbrev} Avg: {h2h_avg:.1f} "
                    f"({h2h_games} games)"
                )

    with colR:
        st.subheader("📈 Recent Performance")
        season_avg = features.get(f"{stat_code}_avg", 0)
        last5 = features.get(f"{stat_code}_last5", season_avg)
        last10 = features.get(f"{stat_code}_last10", season_avg)

        if stat_code != "DD":
            st.write(f"**Season Average:** {season_avg:.1f}")
            st.write(f"**Last 5 Games:** {last5:.1f}")
            st.write(f"**Last 10 Games:** {last10:.1f}")

            wc = features.get("weight_current", 0)
            wp = features.get("weight_prior", 1)
            st.caption(
                f"Blend: {wc*100:.0f}% {cur_season}, "
                f"{wp*100:.0f}% {prev_season}"
            )
        else:
            st.write(f"Chance at DD: {prediction:.1f}% (model)")

    with colCxt:
        st.subheader("🏀 Context")
        rest_days = features.get("rest_days", 3)
        is_b2b = features.get("is_back_to_back", 0)
        st.write(f"**Rest Days:** {rest_days}")
        st.write(f"**Back-to-Back:** {'Yes' if is_b2b else 'No'}")
        st.write(f"**Opponent:** {opponent_abbrev}")

    # Line / Hit Rate
    st.markdown("---")
    st.subheader("📊 Sportsbook Line & Hit Rate")

    colL, colH = st.columns(2)

    with colL:
        st.markdown("**Line / Edge**")

        if stat_code == "DD":
            st.write("Most books don't post DD props as a standard line.")
        else:
            form_key = f"lineform_{override_key}__{unique_suffix}"
            num_key = f"lineinput_{override_key}__{unique_suffix}"

            with st.form(key=form_key):
                new_val = st.number_input(
                    "Your line (adjust this)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(active_line_val),
                    step=0.5,
                    key=num_key,
                )
                submitted = st.form_submit_button("Apply line")

            if submitted:
                st.session_state["custom_lines"][override_key] = float(new_val)
                active_line_val = float(new_val)

            if fd_line_val is not None:
                st.write(f"FanDuel line: **{fd_line_val}**")
            else:
                st.write("FanDuel line: —")

            custom_edge_str, custom_rec_text, _ou_short = calc_edge(
                float(prediction),
                float(active_line_val),
            )
            st.write(
                f"Edge vs Your Line ({active_line_val}): "
                f"**{custom_edge_str}**"
            )
            st.caption(custom_rec_text)

    with colH:
        st.markdown("**Hit Rate (Last 10 Games)**")

        if stat_code == "DD":
            st.write("Hit%: —")
            st.caption("N/A for DD market here.")
        else:
            recent_logs_df = current_logs if not current_logs.empty else prior_logs
            if recent_logs_df is None or recent_logs_df.empty:
                custom_hit_pct = None
            else:
                custom_hit_pct = calc_hit_rate(
                    recent_logs_df,
                    stat_code,
                    float(active_line_val),
                    window=10
                )

            if custom_hit_pct is None:
                st.write("Hit%: —")
                st.caption("Not enough recent games.")
            else:
                st.write(f"Hit%: **{custom_hit_pct:.0f}%**")
                st.caption(
                    "Hit% = % of recent games over your line. "
                    "Historical only."
                )

    # Head-to-head trend section
    if h2h_games > 0 and stat_code != "DD":
        st.markdown("---")
        st.subheader(f"🔥 Head-to-Head vs {opponent_abbrev}")

        h2h_avg = features.get(f"h2h_{stat_code}_avg", 0)
        h2h_trend = features.get(f"h2h_{stat_code}_trend", 0)

        colH2H1, colH2H2 = st.columns(2)
        with colH2H1:
            st.markdown("**Average vs Opponent**")
            st.markdown(f"### Avg: {h2h_avg:.1f} ({h2h_games} games)")
            diff = h2h_avg - features.get(f"{stat_code}_avg", 0)
            clr = "green" if diff > 0 else "red"
            st.markdown(f":{clr}[{diff:+.1f} vs season avg]")

        with colH2H2:
            st.markdown("**Recent Trend**")
            if abs(h2h_trend) > 1:
                trending_up = (h2h_trend > 0)
                trend_text = "📈 Trending UP" if trending_up else "📉 Trending DOWN"
                st.markdown(f"### {trend_text}")
                st.markdown(
                    f":{('green' if trending_up else 'red')}[{h2h_trend:+.1f}]"
                )
            else:
                st.markdown("### ➡️ Consistent")

        if not h2h_history.empty:
            st.markdown("**Recent Games vs Opponent:**")
            base_cols = ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M"]
            show_cols = [c for c in base_cols if c in h2h_history.columns]
            if show_cols:
                h2h_recent = h2h_history.head(5)[show_cols].copy()
                if {"PTS","REB","AST"}.issubset(h2h_recent.columns):
                    h2h_recent["PRA"] = (
                        h2h_recent["PTS"] +
                        h2h_recent["REB"] +
                        h2h_recent["AST"]
                    )
                st.dataframe(h2h_recent, use_container_width=True)

    # Recent Logs
    st.markdown("---")
    st.subheader("📋 Recent Game Log (Last 10 Games)")

    has_current_logs = (current_logs is not None and not current_logs.empty)
    has_prior_logs = (prior_logs is not None and not prior_logs.empty)

    if has_current_logs and len(current_logs) >= 10:
        last10_logs = current_logs.head(10)
        label_season = cur_season
    elif has_current_logs and len(current_logs) < 10:
        need = 10 - len(current_logs)
        last10_logs = pd.concat(
            [current_logs, prior_logs.head(need)], ignore_index=True
        ).head(10)
        label_season = f"{cur_season} + {prev_season}"
    elif has_prior_logs:
        last10_logs = prior_logs.head(10)
        label_season = prev_season
    else:
        last10_logs = pd.DataFrame()
        label_season = "N/A"

    st.caption(f"Showing games from: {label_season}")

    if not last10_logs.empty:
        display_cols = [
            "GAME_DATE","MATCHUP","MIN","PTS","REB","AST",
            "FG3M","FGA","FG_PCT"
        ]
        cols_avail = [c for c in display_cols if c in last10_logs.columns]
        if cols_avail:
            preview_df = last10_logs[cols_avail].copy()
            if {"PTS","REB","AST"}.issubset(preview_df.columns):
                preview_df["PRA"] = (
                    preview_df["PTS"] +
                    preview_df["REB"] +
                    preview_df["AST"]
                )
            st.dataframe(preview_df, use_container_width=True)


# ─────────────────────────────
# Matchup View (streaming / incremental render)
# ─────────────────────────────
def build_matchup_view(
    selected_game,
    stat_code,
    stat_display,
    cur_season,
    prev_season,
    model_obj,
):
    if not selected_game:
        st.title("🏀 NBA Player Props Projection Model")
        st.markdown(
            "Advanced predictions using historical data, matchup analysis, "
            "and head-to-head history"
        )
        st.markdown("---")
        st.subheader("👋 How to use this tool")
        st.markdown(
            """
1. Pick a matchup on the left  
2. Pick a stat (Points, PRA, Double-Double, etc.)  
3. We'll start streaming projections player-by-player  
4. Open any player below and set **your own line**  
   → that becomes the line in the table above
            """
        )
        return

    home_team = selected_game["home"]
    away_team = selected_game["away"]

    st.title("🏀 NBA Player Props Projection Model")
    st.markdown(
        "Advanced predictions using historical data, matchup analysis, "
        "and head-to-head history"
    )

    st.subheader("🏟 Matchup Board")
    st.caption(
        "Projection, your line (or FanDuel if available), model lean, hit rate, and defensive matchup. "
        "Edit lines in any player below."
    )

    show_full = st.checkbox(
        "Show full roster (include deep bench / fringe players)",
        value=False,
        help=(
            "Off = just core rotation / relevant players from each team "
            "(fast load). On = entire roster including low-minute guys."
        )
    )

    table_placeholder = st.empty()
    status_placeholder = st.empty()
    expanders_placeholder = st.empty()

    init_cols = [
        "Player", "Team/Pos", "Proj", "Line",
        "O/U", "Hit%", "Opp Def Rank vs Position"
    ]
    table_placeholder.dataframe(
        pd.DataFrame(columns=init_cols),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("📂 Player Breakdowns (click any name below to expand)")
    st.caption(
        "Adjust a player's line and click 'Apply line'. "
        "Your number becomes the line in the table above."
    )

    # Shared data
    def_vs_pos_df = scrape_defense_vs_position_cached_db()
    team_stats = get_team_stats_cached_db(season=prev_season)

    # Get rosters for both teams
    home_roster = get_players_by_team(home_team, season=cur_season)
    if home_roster.empty:
        home_roster = get_players_by_team(home_team, season=prev_season)
    if not home_roster.empty:
        home_roster = home_roster.copy()
        home_roster["team_abbrev"] = home_team

    away_roster = get_players_by_team(away_team, season=cur_season)
    if away_roster.empty:
        away_roster = get_players_by_team(away_team, season=prev_season)
    if not away_roster.empty:
        away_roster = away_roster.copy()
        away_roster["team_abbrev"] = away_team

    if home_roster.empty and away_roster.empty:
        st.error("Couldn't load rosters for this matchup.")
        return

    # Dedupe inside each roster
    home_roster = home_roster.drop_duplicates(subset=["player_id"])
    away_roster = away_roster.drop_duplicates(subset=["player_id"])

    # --- Helper: safe average minutes ---
    def _safe_avg_minutes(df, n):
        if df is None or df.empty or "MIN" not in df.columns:
            return 0.0
        sl = df.head(min(n, len(df))).copy()
        mins = pd.to_numeric(sl["MIN"], errors="coerce")
        mins = mins.dropna()
        if mins.empty:
            return 0.0
        return float(mins.mean())

    # --- Helper: build a "core table" for a team's players with core_score ---
    def _rank_team_players(roster_df, team_abbrev):
        rows = []
        for _, r in roster_df.iterrows():
            pid = r["player_id"]
            pname = r["full_name"]

            # pull logs for current + prior season (from cache)
            cur_logs = get_player_game_logs_cached_db(pid, pname, season=cur_season)
            prev_logs = get_player_game_logs_cached_db(pid, pname, season=prev_season)

            cur5_avg_min = _safe_avg_minutes(cur_logs, 5)     # recent usage
            prev10_avg_min = _safe_avg_minutes(prev_logs, 10) # historical usage

            # "core_score" keeps injured stars in the mix:
            # if you played big minutes last year, you still rate high
            core_score = max(cur5_avg_min, prev10_avg_min * 0.9)

            rows.append({
                "player_id": pid,
                "full_name": pname,
                "team_abbrev": team_abbrev,
                "core_score": core_score,
            })

        if rows:
            out = pd.DataFrame(rows)
        else:
            out = pd.DataFrame(columns=["player_id", "full_name", "team_abbrev", "core_score"])

        # high score first
        out = out.sort_values("core_score", ascending=False).reset_index(drop=True)
        return out

    home_ranked = _rank_team_players(home_roster, home_team)
    away_ranked = _rank_team_players(away_roster, away_team)

    # how many "core" players per team we consider part of the fast batch
    CORE_PER_TEAM = 8

    home_core = home_ranked.head(CORE_PER_TEAM)
    away_core = away_ranked.head(CORE_PER_TEAM)

    home_rest = home_ranked.iloc[CORE_PER_TEAM:]
    away_rest = away_ranked.iloc[CORE_PER_TEAM:]

    # interleave helper
    def _interleave(df_a, df_b):
        rows = []
        i = j = 0
        while i < len(df_a) or j < len(df_b):
            if i < len(df_a):
                rows.append(df_a.iloc[i])
                i += 1
            if j < len(df_b):
                rows.append(df_b.iloc[j])
                j += 1
        if rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame(columns=df_a.columns if len(df_a.columns) > 0 else df_b.columns)

    core_interleaved = _interleave(home_core, away_core)

    rest_interleaved = _interleave(home_rest, away_rest)

    # final ordering:
    # - If show_full is OFF: just core_interleaved
    # - If show_full is ON: core first, then the rest
    if show_full:
        ordered_players = pd.concat(
            [core_interleaved, rest_interleaved],
            ignore_index=True
        )
    else:
        ordered_players = core_interleaved.copy()

    # De-dupe just in case
    ordered_players = ordered_players.drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    # We'll stream through ordered_players
    max_players = len(ordered_players) if show_full else len(ordered_players)

    # pull odds data once for this specific game
    event_id = get_event_id_for_game(home_team, away_team)
    if event_id:
        odds_data = fetch_fanduel_lines(event_id)
    else:
        odds_data = {}

    opponent_recent_cache = {}
    def get_recent_for_team(team_abbrev_lookup):
        if team_abbrev_lookup not in opponent_recent_cache:
            opponent_recent_cache[team_abbrev_lookup] = get_opponent_recent_games(
                team_abbrev_lookup,
                season=prev_season,
                last_n=10
            )
        return opponent_recent_cache[team_abbrev_lookup]

    player_payloads = []

    # STREAM LOOP
    for loop_idx, prow in enumerate(ordered_players.itertuples(index=False)):
        if loop_idx >= max_players:
            break

        pid = prow.player_id
        player_name = prow.full_name
        team_abbrev = prow.team_abbrev
        opponent_abbrev = away_team if team_abbrev == home_team else home_team

        player_pos = get_player_position(pid, season=prev_season)

        current_logs = get_player_game_logs_cached_db(
            pid, player_name, season=cur_season
        )
        prior_logs = get_player_game_logs_cached_db(
            pid, player_name, season=prev_season
        )

        opponent_recent = get_recent_for_team(opponent_abbrev)

        h2h_history = get_head_to_head_history(
            pid,
            opponent_abbrev,
            seasons=[prev_season, "2023-24"]
        )

        opp_def_rank_info = get_team_defense_rank_vs_position(
            opponent_abbrev,
            player_pos,
            def_vs_pos_df
        )

        feat = build_enhanced_feature_vector(
            current_logs,
            opponent_abbrev,
            team_stats,
            prior_season_logs=prior_logs,
            opponent_recent_games=opponent_recent,
            head_to_head_games=h2h_history,
            player_position=player_pos
        )

        if stat_code == "DD":
            raw_pct = model_obj.predict_double_double(feat) * 100.0
            pred_val = raw_pct
            if raw_pct >= 5.0:
                proj_display = f"{raw_pct:.1f}%"
            elif raw_pct >= 1.0:
                proj_display = "<5%"
            else:
                proj_display = "—"
        else:
            pred_val = model_obj.predict(feat, stat_code)
            proj_display = f"{pred_val:.1f}"

        if stat_code == "DD":
            fd_line_val = None
        else:
            fd_info = get_player_fanduel_line(player_name, stat_code, odds_data)
            fd_line_val = fd_info["line"] if fd_info else None

        if stat_code == "DD":
            base_line_val = None
        else:
            base_line_val = (
                fd_line_val if fd_line_val is not None
                else round(float(pred_val), 1)
            )

        pdata = {
            "player_name": player_name,
            "team_abbrev": team_abbrev,
            "player_pos": player_pos,
            "opponent_abbrev": opponent_abbrev,

            "current_logs": current_logs,
            "prior_logs": prior_logs,
            "h2h_history": h2h_history,
            "opp_def_rank": opp_def_rank_info,
            "features": feat,

            "prediction": pred_val,
            "proj_display": proj_display,
            "stat_code": stat_code,
            "stat_display": stat_display,

            "fd_line_val": fd_line_val,
            "base_line_val": base_line_val,
        }

        player_payloads.append(pdata)

        # update expanders for all loaded so far
        with expanders_placeholder.container():
            for idx_render, info in enumerate(player_payloads):
                header_label = (
                    f"{info['player_name']} "
                    f"({info['team_abbrev']} · {info['player_pos']})"
                )
                with st.expander(header_label, expanded=False):
                    render_player_detail_body(
                        info,
                        cur_season,
                        prev_season,
                        unique_suffix=f"{loop_idx}_{idx_render}",
                    )

        # build table rows so far
        table_rows = []
        for info in player_payloads:
            p_name = info["player_name"]
            t_abbr = info["team_abbrev"]
            p_pos = info["player_pos"]
            pred_val_local = info["prediction"]
            proj_disp_local = info["proj_display"]
            stat_code_local = info["stat_code"]
            cur_logs_local = info["current_logs"]
            prior_logs_local = info["prior_logs"]
            opp_def_rank_local = info["opp_def_rank"]
            base_line_local = info["base_line_val"]

            override_key = f"{p_name}|{t_abbr}|{stat_code_local}"
            if stat_code_local == "DD":
                final_line_val = None
            else:
                final_line_val = st.session_state["custom_lines"].get(
                    override_key,
                    base_line_local
                )

            if stat_code_local == "DD":
                hit_pct_val = None
                ou_short = "—"
            else:
                recent_logs_df = cur_logs_local if not cur_logs_local.empty else prior_logs_local
                if (
                    recent_logs_df is not None
                    and not recent_logs_df.empty
                    and final_line_val is not None
                ):
                    hit_pct_val = calc_hit_rate(
                        recent_logs_df,
                        stat_code_local,
                        float(final_line_val),
                        window=10
                    )
                else:
                    hit_pct_val = None

                _edge_str, _rec_text, ou_short = calc_edge(
                    float(pred_val_local),
                    float(final_line_val) if final_line_val is not None else None
                )

            rank_num = opp_def_rank_local.get("rank", 15)
            rating_txt = opp_def_rank_local.get("rating", "Average")
            d_emoji = defense_emoji(rank_num)
            opp_def_display = f"{d_emoji} #{rank_num} ({rating_txt})"

            table_rows.append({
                "Player": p_name,
                "Team/Pos": f"{t_abbr} · {p_pos}",
                "Proj": proj_disp_local,
                "Line": "—" if final_line_val is None else final_line_val,
                "O/U": ou_short,
                "Hit%": "—" if hit_pct_val is None else f"{hit_pct_val:.0f}%",
                "Opp Def Rank vs Position": opp_def_display,
            })

        table_df = pd.DataFrame(table_rows)
        table_placeholder.dataframe(table_df, use_container_width=True)

        status_placeholder.write(
            f"Loaded {len(player_payloads)}/{max_players} players..."
        )

        time.sleep(0.05)

    status_placeholder.success("✅ Done.")


# ─────────────────────────────
# Sidebar UI
# ─────────────────────────────
st.sidebar.header("⚙️ Settings")

st.sidebar.write(f"**Current Season:** {current_season}")
st.sidebar.write(f"**Prior Season:** {prior_season}")

with st.sidebar.expander("💾 Cache Stats"):
    cache_stats = get_cache_stats()
    st.write(f"**Players cached:** {cache_stats.get('total_players', 0)}")
    st.write(f"**Games cached:** {cache_stats.get('total_games', 0):,}")
    st.write(f"**DB Size:** {cache_stats.get('db_size_mb', 0):.1f} MB")

    if st.button("🗑️ Clear Old Seasons"):
        clear_old_seasons([current_season, prior_season])
        st.success("Old seasons cleared!")
        st.rerun()

st.sidebar.subheader("📅 Select Upcoming Game")

upcoming_games = get_upcoming_games(days=7)
selected_game = None
game_map = {}

if upcoming_games:
    sidebar_options = ["-- Select a Game --"]
    for g in upcoming_games:
        date_disp = g.get("date_display", "")
        tm = f" ({g['time_display']})" if g.get("time_display") else ""
        label = f"{date_disp} - {g['away']} @ {g['home']}{tm}"
        sidebar_options.append(label)
        game_map[label] = g

    picked_label = st.sidebar.selectbox(
        f"Upcoming games (next 7 days) - {len(upcoming_games)} found",
        options=sidebar_options,
        index=0,
    )

    if picked_label != "-- Select a Game --":
        selected_game = game_map[picked_label]
        st.sidebar.info(
            f"Matchup: {selected_game['away']} @ {selected_game['home']}"
        )
else:
    st.sidebar.warning("⚠️ No upcoming games in next 7 days.")
    picked_label = None
    selected_game = None

st.sidebar.subheader("📊 Stat to Project")
stat_display_list = list(STAT_OPTIONS.keys())
stat_display_choice = st.sidebar.selectbox(
    "Choose stat to preview on the board",
    options=stat_display_list,
    index=0,
)
stat_code_choice = STAT_OPTIONS[stat_display_choice]

# ─────────────────────────────
# Render main view
# ─────────────────────────────
build_matchup_view(
    selected_game=selected_game,
    stat_code=stat_code_choice,
    stat_display=stat_display_choice,
    cur_season=current_season,
    prev_season=prior_season,
    model_obj=model,
)

st.markdown("---")
st.markdown(
    "**Data Sources:** NBA.com (via nba_api) | "
    "**Model:** Enhanced Ridge Regression  \n"
    "**Features:** Season blending, H2H history, opponent recent form, "
    "positional defense  \n"
    "**Sportsbook Lines:** FanDuel via The Odds API  \n"
    "**Note:** Projections are informational only. "
    "Always verify lines and context."
)
