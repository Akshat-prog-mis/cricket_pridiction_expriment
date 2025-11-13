"""
Complete Integrated Cricket ML Analytics Dashboard
===================================================
Full Streamlit app with predictions, simulations, and analytics
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="🏏 Cricket ML Analytics Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .simulation-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">🏏 Cricket ML Analytics Pro</p>', unsafe_allow_html=True)
st.markdown("**🤖 ML Predictions • 🎲 Monte Carlo Simulation • 📊 Advanced Analytics • 🏆 Team Comparison**")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cricket.png", width=80)
    st.title("⚡ Navigation")
    
    page = st.radio(
        "Select Page",
        [
            "🏠 Overview",
            "🔮 Live Predictor",
            "🎲 Monte Carlo Simulator", 
            "⚔️ Team Comparison",
            "👤 Player Analysis",
            "🏟️ Venue Intelligence",
            "📈 Match Analytics",
            "🎯 Pressure Analyzer"
        ]
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** All predictions use advanced ML models trained on historical IPL data")
    
    # API Status
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        if response.status_code == 200:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Error")
    except:
        st.error("🔴 API Offline")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
def fetch_overview_stats():
    try:
        response = requests.get(f"{API_BASE}/stats/overview")
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

@st.cache_data(ttl=300)
def fetch_venues():
    try:
        response = requests.get(f"{API_BASE}/venues/list")
        return response.json().get("venues", []) if response.status_code == 200 else []
    except:
        return []

@st.cache_data(ttl=300)
def fetch_teams():
    try:
        response = requests.get(f"{API_BASE}/teams/list")
        return response.json().get("teams", []) if response.status_code == 200 else []
    except:
        return []

@st.cache_data(ttl=300)
def fetch_players():
    try:
        response = requests.get(f"{API_BASE}/players/list", params={"limit": 200})
        return response.json().get("players", []) if response.status_code == 200 else []
    except:
        return []

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.header("📊 Database Overview")
    
    with st.spinner("Loading statistics..."):
        stats = fetch_overview_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏏 Total Matches", f"{stats.get('total_matches', 0):,}")
        with col2:
            st.metric("⚾ Total Deliveries", f"{stats.get('total_deliveries', 0):,}")
        with col3:
            st.metric("👤 Unique Players", f"{stats.get('unique_players', 0):,}")
        with col4:
            st.metric("🏟️ Venues", f"{stats.get('unique_venues', 0):,}")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            date_range = stats.get('date_range', ['N/A', 'N/A'])
            st.info(f"**📅 Date Range:** {date_range[0]} to {date_range[1]}")
        with col_b:
            st.info(f"**🏆 Total Series:** {stats.get('total_series', 0):,}")
        
        st.success("✅ Database loaded successfully!")
        
        # Quick stats
        st.subheader("🎯 System Capabilities")
        
        capabilities = {
            "🔮 Ball-by-Ball Predictions": "Predict next ball outcome with probabilities",
            "🎲 Monte Carlo Simulations": "Run 5000+ simulations to predict match outcomes",
            "📊 Win Probability Tracking": "Real-time win probability calculations",
            "⚔️ Team Comparisons": "Compare custom XIs at any venue",
            "👤 Player Head-to-Head": "Detailed batter vs bowler matchups",
            "🏟️ Venue Analysis": "Comprehensive venue statistics and trends"
        }
        
        for capability, description in capabilities.items():
            st.markdown(f"**{capability}:** {description}")
    
    else:
        st.error("⚠️ Unable to fetch statistics. Make sure the API is running.")

# ============================================================================
# PAGE: LIVE PREDICTOR
# ============================================================================

elif page == "🔮 Live Predictor":
    st.header("🔮 Live Match Predictor")
    st.markdown("**Enter current match state to get instant predictions**")
    
    # Load data for dropdowns
    venues = fetch_venues()
    teams = fetch_teams()
    players = fetch_players()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Match Setup")
        
        match_id = st.text_input("Match ID", value="live_match_001")
        venue = st.selectbox("🏟️ Venue", [v['venue'] for v in venues] if venues else ["Wankhede Stadium"])
        city = st.text_input("🏙️ City", value="Mumbai")
        
        inning = st.selectbox("Innings", [1, 2])
        
        batting_team = st.selectbox("🏏 Batting Team", teams if teams else ["Team A"])
        bowling_team = st.selectbox("⚾ Bowling Team", teams if teams else ["Team B"])
        
        if inning == 2:
            target = st.number_input("🎯 Target", min_value=1, max_value=300, value=180)
        else:
            target = None
    
    with col2:
        st.subheader("🎯 Current State")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            over = st.number_input("Over", min_value=0, max_value=19, value=10)
            ball_in_over = st.number_input("Ball", min_value=0, max_value=5, value=3)
        
        with col_b:
            runs_so_far = st.number_input("Runs", min_value=0, max_value=300, value=85)
            wickets_so_far = st.number_input("Wickets", min_value=0, max_value=10, value=3)
        
        st.markdown("---")
        
        batter = st.selectbox("🏏 Batter", players if players else ["Virat Kohli"])
        non_striker = st.selectbox("🏃 Non-Striker", players if players else ["AB de Villiers"])
        bowler = st.selectbox("⚾ Bowler", players if players else ["Jasprit Bumrah"])
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 PREDICT NEXT BALL", type="primary"):
        with st.spinner("Running ML prediction..."):
            try:
                # Prepare request
                match_state = {
                    "match_id": match_id,
                    "venue": venue,
                    "city": city,
                    "inning": inning,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over,
                    "ball_in_over": ball_in_over,
                    "runs_so_far": runs_so_far,
                    "wickets_so_far": wickets_so_far,
                    "batter": batter,
                    "non_striker": non_striker,
                    "bowler": bowler,
                    "target": target
                }
                
                # Get prediction
                response = requests.post(f"{API_BASE}/predict/ball", json=match_state)
                
                if response.status_code == 200:
                    prediction = response.json()
                    
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Most Likely Outcome", 
                                f"{prediction['most_likely']} runs" if prediction['most_likely'] < 7 else "Wicket")
                    with col2:
                        st.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
                    with col3:
                        st.metric("Wicket Probability", f"{prediction['wicket_probability']*100:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.subheader("📊 Outcome Probabilities")
                    
                    probs_data = {
                        'Outcome': ['0', '1', '2', '3', '4', '6', 'Wicket', 'Extra'],
                        'Probability': [
                            prediction['runs_probabilities']['0'],
                            prediction['runs_probabilities']['1'],
                            prediction['runs_probabilities']['2'],
                            prediction['runs_probabilities']['3'],
                            prediction['runs_probabilities']['4'],
                            prediction['runs_probabilities']['6'],
                            prediction['runs_probabilities']['wicket'],
                            prediction['runs_probabilities']['extra']
                        ]
                    }
                    
                    fig = px.bar(probs_data, x='Outcome', y='Probability',
                               title="Next Ball Outcome Probabilities",
                               color='Probability',
                               color_continuous_scale='Viridis')
                    fig.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Context
                    st.info(f"**Match Context:** {prediction['match_context']['phase']} phase • "
                           f"Pressure: {prediction['match_context']['pressure']}")
                    
                    # Next over prediction
                    if st.button("🔮 Predict Next Over"):
                        with st.spinner("Predicting next 6 balls..."):
                            over_response = requests.post(f"{API_BASE}/predict/over", json=match_state)
                            
                            if over_response.status_code == 200:
                                over_pred = over_response.json()
                                
                                st.success(f"**Expected Runs in Next Over:** {over_pred['expected_runs']} "
                                         f"(Range: {over_pred['range']['min']}-{over_pred['range']['max']})")
                                
                                # Ball by ball
                                ball_df = pd.DataFrame(over_pred['ball_by_ball'])
                                st.dataframe(ball_df, use_container_width=True)
                
                else:
                    st.error(f"Prediction failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE: MONTE CARLO SIMULATOR
# ============================================================================

elif page == "🎲 Monte Carlo Simulator":
    st.header("🎲 Monte Carlo Match Simulator")
    st.markdown("**Run thousands of simulations to predict match outcomes**")
    
    # Load data
    venues = fetch_venues()
    teams = fetch_teams()
    players = fetch_players()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⚙️ Match Info")
        venue = st.selectbox("Venue", [v['venue'] for v in venues] if venues else ["Eden Gardens"])
        inning = st.selectbox("Innings", [1, 2], key="sim_inning")
        batting_team = st.selectbox("Batting", teams if teams else ["Team A"], key="sim_bat")
        bowling_team = st.selectbox("Bowling", teams if teams else ["Team B"], key="sim_bowl")
    
    with col2:
        st.subheader("🎯 Current State")
        over = st.number_input("Over", 0, 19, 12, key="sim_over")
        ball = st.number_input("Ball", 0, 5, 3, key="sim_ball")
        runs = st.number_input("Runs", 0, 300, 95, key="sim_runs")
        wickets = st.number_input("Wickets", 0, 10, 4, key="sim_wickets")
    
    with col3:
        st.subheader("👥 Players")
        batter = st.selectbox("Batter", players if players else ["Player A"], key="sim_batter")
        non_striker = st.selectbox("Non-Striker", players if players else ["Player B"], key="sim_ns")
        bowler = st.selectbox("Bowler", players if players else ["Bowler X"], key="sim_bowler")
    
    if inning == 2:
        target = st.number_input("🎯 Target", 1, 300, 165, key="sim_target")
    else:
        target = None
    
    n_sims = st.slider("🎲 Number of Simulations", 1000, 10000, 5000, step=1000)
    
    st.markdown("---")
    
    if st.button("🚀 RUN SIMULATION", type="primary"):
        with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
            try:
                # Prepare request
                sim_request = {
                    "match_state": {
                        "match_id": "simulation",
                        "venue": venue,
                        "city": "Mumbai",
                        "inning": inning,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "over": over,
                        "ball_in_over": ball,
                        "runs_so_far": runs,
                        "wickets_so_far": wickets,
                        "batter": batter,
                        "non_striker": non_striker,
                        "bowler": bowler,
                        "target": target
                    },
                    "n_simulations": n_sims
                }
                
                response = requests.post(f"{API_BASE}/simulate", json=sim_request)
                
                if response.status_code == 200:
                    results = response.json()['simulation_results']
                    
                    st.markdown('<div class="simulation-box">', unsafe_allow_html=True)
                    st.subheader("🎯 Simulation Results")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Key metrics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Expected Score", 
                                f"{results['score_distribution']['median']:.0f}")
                    with col_b:
                        st.metric("Most Likely Range",
                                f"{results['score_distribution']['q25']:.0f}-{results['score_distribution']['q75']:.0f}")
                    with col_c:
                        if 'win_probability' in results:
                            st.metric("Win Probability",
                                    f"{results['win_probability']*100:.1f}%")
                        else:
                            st.metric("Max Score", f"{results['score_distribution']['max']}")
                    with col_d:
                        st.metric("Avg Wickets Lost",
                                f"{results['wickets_distribution']['mean']:.1f}")
                    
                    # Score distribution
                    st.subheader("📊 Score Distribution")
                    
                    score_dist = results['score_distribution']
                    
                    fig = go.Figure()
                    
                    # Add percentile ranges
                    fig.add_trace(go.Box(
                        y=[score_dist['min'], score_dist['q05'], score_dist['q25'],
                           score_dist['median'], score_dist['q75'], score_dist['q95'], score_dist['max']],
                        name="Score Range",
                        boxmean='sd'
                    ))
                    
                    fig.update_layout(
                        title="Final Score Distribution",
                        yaxis_title="Runs",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed stats
                    st.subheader("📈 Detailed Statistics")
                    
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '5th %ile', '95th %ile'],
                        'Score': [
                            f"{score_dist['mean']:.1f}",
                            f"{score_dist['median']:.1f}",
                            f"{score_dist['std']:.1f}",
                            score_dist['min'],
                            score_dist['max'],
                            f"{score_dist['q05']:.1f}",
                            f"{score_dist['q95']:.1f}"
                        ]
                    })
                    
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    # Win probability gauge (if 2nd innings)
                    if 'win_probability' in results:
                        st.subheader("🎯 Win Probability")
                        
                        win_prob = results['win_probability'] * 100
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=win_prob,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"{batting_team} Win Probability"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgray"},
                                    {'range': [25, 50], 'color': "gray"},
                                    {'range': [50, 75], 'color': "lightgreen"},
                                    {'range': [75, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"**Target:** {results['target']} | "
                               f"**Runs Needed:** {results['runs_needed']} | "
                               f"**Required RR:** {results['required_run_rate']:.2f}")
                
                else:
                    st.error(f"Simulation failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE: TEAM COMPARISON
# ============================================================================

elif page == "⚔️ Team Comparison":
    st.header("⚔️ Team Comparison & Head-to-Head")
    
    tab1, tab2 = st.tabs(["📊 Historical H2H", "🎯 Custom XI Comparison"])
    
    teams = fetch_teams()
    venues = fetch_venues()
    players = fetch_players()
    
    with tab1:
        st.subheader("📊 Historical Head-to-Head")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            team1 = st.selectbox("Team 1", teams if teams else ["Team A"], key="h2h_t1")
        with col2:
            team2 = st.selectbox("Team 2", teams if teams else ["Team B"], key="h2h_t2")
        with col3:
            venue_filter = st.selectbox("Venue", ["All Venues"] + [v['venue'] for v in venues] if venues else ["All Venues"])
        
        if st.button("⚔️ Compare Teams"):
            if team1 == team2:
                st.error("Please select different teams")
            else:
                with st.spinner("Analyzing head-to-head..."):
                    try:
                        params = {"team1": team1, "team2": team2}
                        if venue_filter != "All Venues":
                            params["venue"] = venue_filter
                        
                        response = requests.get(f"{API_BASE}/teams/vs", params=params)
                        
                        if response.status_code == 200:
                            data = response.json()
                            stats = data['stats']
                            
                            st.success(f"**📍 Venue:** {data.get('venue', 'All Venues')}")
                            
                            col_a, col_b, col_c, col_d = st.columns(4)
                            
                            with col_a:
                                st.metric("Total Matches", stats['total_matches'])
                            with col_b:
                                st.metric(f"{team1} Wins", stats['team1_wins'])
                            with col_c:
                                st.metric(f"{team2} Wins", stats['team2_wins'])
                            with col_d:
                                st.metric("Draws/NR", stats['draws'])
                            
                            # Win distribution pie chart
                            if stats['total_matches'] > 0:
                                fig = go.Figure(data=[go.Pie(
                                    labels=[team1, team2, 'Draw/NR'],
                                    values=[stats['team1_wins'], stats['team2_wins'], stats['draws']],
                                    hole=.4,
                                    marker_colors=['#1f77b4', '#ff7f0e', '#d62728']
                                )])
                                fig.update_layout(title="Win Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.info(f"📅 **First Match:** {stats.get('first_match', 'N/A')} | **Last Match:** {stats.get('last_match', 'N/A')}")
                            
                            st.markdown("---")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Fours", data.get('fours', 0))
                            with col2:
                                st.metric("Sixes", data.get('sixes', 0))
                            with col3:
                                st.metric("Dots", data.get('dots', 0))
                            with col4:
                                st.metric("Avg RPB", f"{data.get('avg_runs_per_ball', 0):.2f}")
                            
                            # Visualization
                            if 'fours' in data and 'sixes' in data and 'dots' in data:
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=['Fours', 'Sixes', 'Dots'],
                                        y=[data['fours'], data['sixes'], data['dots']],
                                        marker_color=['#1f77b4', '#2ca02c', '#d62728']
                                    )
                                ])
                                fig.update_layout(
                                    title="Ball Distribution",
                                    yaxis_title="Count",
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Performance assessment
                            if 'strike_rate' in data:
                                sr = data['strike_rate']
                                dismissal_pct = (data.get('dismissals', 0) / data.get('balls_faced', 1) * 100) if data.get('balls_faced', 0) > 0 else 0
                                
                                if sr > 150:
                                    performance = "🔥 Dominant"
                                elif sr > 120:
                                    performance = "✅ Good"
                                elif sr > 100:
                                    performance = "⚖️ Even"
                                else:
                                    performance = "😰 Struggling"
                                
                                st.info(f"**Performance Assessment:** {performance} | Dismissal Rate: {dismissal_pct:.1f}%")
                        else:
                            st.error(f"Request failed: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("🎯 Custom XI Comparison")
        st.markdown("Select 11 players for each team and compare at a venue")
        
        venue_select = st.selectbox("Select Venue", [v['venue'] for v in venues] if venues else ["Wankhede Stadium"], key="xi_venue")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**🔵 Team 1 XI**")
            team1_xi = st.multiselect(
                "Select 11 Players",
                players if players else [],
                key="team1_xi",
                max_selections=11
            )
        
        with col_right:
            st.markdown("**🔴 Team 2 XI**")
            team2_xi = st.multiselect(
                "Select 11 Players",
                players if players else [],
                key="team2_xi",
                max_selections=11
            )
        
        if st.button("⚔️ Compare XIs"):
            if len(team1_xi) != 11 or len(team2_xi) != 11:
                st.error("Please select exactly 11 players for each team")
            else:
                with st.spinner("Comparing custom XIs..."):
                    try:
                        payload = {
                            "team1_players": team1_xi,
                            "team2_players": team2_xi,
                            "venue": venue_select
                        }
                        
                        response = requests.post(f"{API_BASE}/teams/compare", json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            st.success(f"**🏟️ {data['venue']}** | Avg Score: **{data['venue_avg_score']}**")
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("### 🔵 Team 1")
                                st.metric("Total Experience", f"{data['team1']['total_experience']:,} balls")
                                st.metric("Avg Strike Rate", f"{data['team1']['avg_strike_rate']:.2f}")
                                st.dataframe(pd.DataFrame(data['team1']['stats']), use_container_width=True, hide_index=True)
                            
                            with col_b:
                                st.markdown("### 🔴 Team 2")
                                st.metric("Total Experience", f"{data['team2']['total_experience']:,} balls")
                                st.metric("Avg Strike Rate", f"{data['team2']['avg_strike_rate']:.2f}")
                                st.dataframe(pd.DataFrame(data['team2']['stats']), use_container_width=True, hide_index=True)
                            
                            # Comparison chart
                            fig = go.Figure(data=[
                                go.Bar(name='Experience', x=['Team 1', 'Team 2'], 
                                      y=[data['team1']['total_experience'], data['team2']['total_experience']]),
                                go.Bar(name='Avg SR', x=['Team 1', 'Team 2'], 
                                      y=[data['team1']['avg_strike_rate'], data['team2']['avg_strike_rate']])
                            ])
                            fig.update_layout(barmode='group', title="Team Comparison")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info(f"**Advantage:** {data['advantage']}")
                        else:
                            st.error(f"Request failed: {response.status_code}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE: VENUE INTELLIGENCE
# ============================================================================

elif page == "🏟️ Venue Intelligence":
    st.header("🏟️ Venue Intelligence & Analysis")
    
    venues = fetch_venues()
    
    tab1, tab2 = st.tabs(["📊 All Venues Stats", "🔍 Detailed Analysis"])
    
    with tab1:
        st.subheader("📊 All Venue Statistics")
        
        with st.spinner("Loading venue stats..."):
            try:
                response = requests.get(f"{API_BASE}/venues/stats")
                
                if response.status_code == 200:
                    venue_df = pd.DataFrame(response.json())
                    
                    if not venue_df.empty:
                        st.dataframe(venue_df, use_container_width=True, hide_index=True)
                        
                        # Visualization
                        fig = px.scatter(
                            venue_df,
                            x='venue_avg_first_innings',
                            y='venue_boundary_rate',
                            size='matches_played',
                            hover_data=['venue', 'city'],
                            title="Venue Characteristics: Avg Score vs Boundary Rate",
                            labels={
                                'venue_avg_first_innings': 'Avg First Innings Score',
                                'venue_boundary_rate': 'Boundary Rate (%)'
                            },
                            color='venue_avg_first_innings',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No venue statistics available")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("🔍 Detailed Venue Analysis")
        
        selected_venue = st.selectbox(
            "Select Venue",
            [v['venue'] for v in venues] if venues else []
        )
        
        if st.button("🔍 Analyze Venue", type="primary"):
            with st.spinner(f"Analyzing {selected_venue}..."):
                try:
                    response = requests.get(f"{API_BASE}/venues/analysis/{selected_venue}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.success(f"**🏟️ {data['venue']}**")
                        
                        # Venue stats
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Matches", data['stats']['total_matches'])
                        with col2:
                            st.metric("Different Winners", data['stats']['different_winners'])
                        with col3:
                            st.metric("Toss-Win %", f"{data['stats']['toss_win_correlation']:.1f}%")
                        
                        st.info(f"**🏆 Most Successful Team:** {data['stats']['most_successful_team']}")
                        
                        # Team performance
                        st.subheader("📊 Team Performance at Venue")
                        
                        if data['team_performance']:
                            team_df = pd.DataFrame(data['team_performance'])
                            
                            fig = px.bar(
                                team_df,
                                x='team',
                                y='wins',
                                color='win_pct',
                                title=f"Team Wins at {selected_venue}",
                                labels={'wins': 'Wins', 'team': 'Team', 'win_pct': 'Win %'},
                                color_continuous_scale='Viridis',
                                text='wins'
                            )
                            fig.update_traces(textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(team_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No team performance data available")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE: MATCH ANALYTICS
# ============================================================================

elif page == "📈 Match Analytics":
    st.header("📈 Match-Level Analytics")
    
    st.markdown("**Analyze historical match data and trends**")
    
    # Get sample match IDs
    try:
        conn = requests.get(f"{API_BASE}/stats/overview")
        if conn.status_code == 200:
            st.info("Enter a match ID to analyze (e.g., match_001, match_100)")
    except:
        pass
    
    match_id = st.text_input("🆔 Enter Match ID", value="335982")
    
    tab1, tab2 = st.tabs(["📊 Win Probability Tracker", "⚡ Pressure Moments"])
    
    with tab1:
        st.subheader("📊 Win Probability Tracker")
        
        if st.button("📈 Track Win Probability"):
            with st.spinner("Tracking win probability..."):
                try:
                    response = requests.get(
                        f"{API_BASE}/analysis/win-probability-tracker",
                        params={"match_id": match_id}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.success(f"**Match ID:** {data['match_id']} | **Target:** {data['target']}")
                        
                        # Convert to dataframe
                        tracker_df = pd.DataFrame(data['tracker'])
                        
                        if not tracker_df.empty:
                            # Line chart
                            fig = px.line(
                                tracker_df,
                                x='over',
                                y='win_probability',
                                title="Win Probability Throughout Match",
                                labels={'over': 'Over', 'win_probability': 'Win Probability (%)'},
                                markers=True
                            )
                            fig.add_hline(y=50, line_dash="dash", line_color="red", 
                                        annotation_text="50% - Even")
                            fig.update_layout(yaxis_range=[0, 100])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Data table
                            st.dataframe(tracker_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No tracking data available for this match")
                    
                    else:
                        st.error("Match not found or error occurred")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("⚡ Pressure Moments")
        
        if st.button("⚡ Find Pressure Moments"):
            with st.spinner("Analyzing pressure moments..."):
                try:
                    response = requests.get(
                        f"{API_BASE}/analysis/pressure-moments",
                        params={"match_id": match_id}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.success(f"**Match ID:** {data['match_id']}")
                        
                        if data['pressure_moments']:
                            st.markdown("### 🔥 Top Pressure Moments")
                            
                            for i, moment in enumerate(data['pressure_moments'], 1):
                                col1, col2, col3 = st.columns([1, 2, 1])
                                
                                with col1:
                                    st.markdown(f"**#{i}**")
                                    st.markdown(f"Over {moment['over']}.{moment['ball']}")
                                
                                with col2:
                                    st.markdown(f"**{moment['description']}**")
                                    st.markdown(f"Innings {moment['inning']}")
                                
                                with col3:
                                    st.metric("Pressure Score", moment['pressure_score'])
                                
                                st.markdown("---")
                        else:
                            st.warning("No significant pressure moments found")
                    
                    else:
                        st.error("Match not found or error occurred")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE: PRESSURE ANALYZER
# ============================================================================

elif page == "🎯 Pressure Analyzer":
    st.header("🎯 Match Pressure & Momentum Analyzer")
    
    st.markdown("**Analyze how pressure affects performance**")
    
    teams = fetch_teams()
    venues = fetch_venues()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Scenario Setup")
        
        venue = st.selectbox("Venue", [v['venue'] for v in venues] if venues else ["Wankhede Stadium"], key="pressure_venue")
        
        scenario = st.selectbox(
            "Pressure Scenario",
            [
                "High Pressure - Need 60 from 30 balls",
                "Medium Pressure - Need 40 from 30 balls",
                "Low Pressure - Need 20 from 30 balls",
                "Death Overs - Last 3 overs",
                "Powerplay - First 6 overs"
            ]
        )
    
    with col2:
        st.subheader("📊 Expected Outcomes")
        
        # Show scenario-specific predictions
        if "High Pressure" in scenario:
            st.metric("Required Run Rate", "12.00")
            st.metric("Success Probability", "25%")
            st.metric("Expected Wickets", "2-3")
        elif "Medium Pressure" in scenario:
            st.metric("Required Run Rate", "8.00")
            st.metric("Success Probability", "55%")
            st.metric("Expected Wickets", "1-2")
        elif "Low Pressure" in scenario:
            st.metric("Required Run Rate", "4.00")
            st.metric("Success Probability", "85%")
            st.metric("Expected Wickets", "0-1")
        elif "Death Overs" in scenario:
            st.metric("Expected Run Rate", "10-12")
            st.metric("Wicket Probability", "15-20%")
            st.metric("Boundary Rate", "25-30%")
        else:  # Powerplay
            st.metric("Expected Run Rate", "7-8")
            st.metric("Wicket Probability", "5-8%")
            st.metric("Boundary Rate", "20-25%")
    
    st.markdown("---")
    
    # Pressure insights
    st.subheader("💡 Pressure Insights")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**🔥 High Pressure Factors**")
        st.markdown("""
        - Required RR > 12
        - < 30 balls remaining
        - 6+ wickets down
        - Death overs (16-20)
        - Quality bowlers
        """)
    
    with col_b:
        st.markdown("**⚖️ Momentum Shifters**")
        st.markdown("""
        - Boundaries in death overs
        - Quick wickets
        - Dot ball sequences
        - Partnership breaks
        - Strategic timeouts
        """)
    
    with col_c:
        st.markdown("**✅ Success Indicators**")
        st.markdown("""
        - Partnership stability
        - Wickets in hand
        - Batsman strike rate
        - Bowler economy
        - Match awareness
        """)
    
    # Pressure distribution chart
    st.subheader("📊 Pressure Distribution by Phase")
    
    phase_data = {
        'Phase': ['Powerplay', 'Middle Overs', 'Death Overs'],
        'Avg Pressure': [3.5, 5.0, 8.5],
        'Wicket Risk': [5, 4, 7],
        'Scoring Rate': [7.5, 8.0, 10.5]
    }
    
    phase_df = pd.DataFrame(phase_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Avg Pressure',
        x=phase_df['Phase'],
        y=phase_df['Avg Pressure'],
        marker_color='#ff7f0e'
    ))
    
    fig.add_trace(go.Bar(
        name='Wicket Risk',
        x=phase_df['Phase'],
        y=phase_df['Wicket Risk'],
        marker_color='#d62728'
    ))
    
    fig.add_trace(go.Bar(
        name='Scoring Rate',
        x=phase_df['Phase'],
        y=phase_df['Scoring Rate'],
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        barmode='group',
        title="Match Phase Characteristics",
        yaxis_title="Score (Normalized)",
        legend_title="Metric"
    )
    
    st.plotly_chart(fig, width=True)

# ============================================================================
# PAGE: PLAYER ANALYSIS
# ============================================================================

elif page == "👤 Player Analysis":
    st.header("👤 Player Head-to-Head Analysis")
    
    players = fetch_players()
    
    col1, col2 = st.columns(2)
    
    with col1:
        batter = st.selectbox("🏏 Select Batter", players if players else ["Player A"])
    
    with col2:
        bowler = st.selectbox("⚾ Select Bowler", players if players else ["Bowler X"])
    
    if st.button("⚔️ Analyze Matchup", type="primary"):
        with st.spinner(f"Analyzing {batter} vs {bowler}..."):
            try:
                response = requests.get(
                    f"{API_BASE}/players/head-to-head",
                    params={"batter": batter, "bowler": bowler}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'message' in data:
                        st.warning(data['message'])
                    else:
                        st.success(f"**🏏 {batter}** vs **⚾ {bowler}**")
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            st.metric("Balls Faced", data.get('balls_faced', 0))
                        with col_b:
                            st.metric("Runs Scored", data.get('runs_scored', 0))
                        with col_c:
                            st.metric("Strike Rate", f"{data.get('strike_rate', 0):.2f}")
                        with col_d:
                            st.metric("Dismissals", data.get('dismissals', 0))
                        
                        # Additional stats if available
                        if 'average' in data:
                            col_e, col_f = st.columns(2)
                            with col_e:
                                st.metric("Average", f"{data['average']:.2f}")
                            with col_f:
                                st.metric("Boundaries", data.get('boundaries', 0))
                        
                        # Visualization if we have enough data
                        if data.get('balls_faced', 0) > 0:
                            outcome_data = {
                                'Outcome': ['Dot', '1', '2', '3', '4', '6', 'Wicket'],
                                'Count': [
                                    data.get('dots', 0),
                                    data.get('ones', 0),
                                    data.get('twos', 0),
                                    data.get('threes', 0),
                                    data.get('fours', 0),
                                    data.get('sixes', 0),
                                    data.get('dismissals', 0)
                                ]
                            }
                            
                            outcome_df = pd.DataFrame(outcome_data)
                            
                            fig = px.bar(
                                outcome_df,
                                x='Outcome',
                                y='Count',
                                title=f"{batter} vs {bowler} - Outcome Distribution",
                                color='Count',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Request failed: {response.status_code}")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>🏏 Cricket ML Analytics Pro</b> • Powered by FastAPI + DuckDB + Streamlit</p>
    <p>Advanced ML Models • Monte Carlo Simulation • Real-time Analytics</p>
    <p>Built with ❤️ for cricket analytics enthusiasts</p>
</div>
""", unsafe_allow_html=True)