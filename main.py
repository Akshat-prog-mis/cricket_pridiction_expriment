"""
Complete Integrated Cricket ML API
===================================
Full integration with trained models, Monte Carlo simulation, and all endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import logging
from pathlib import Path
import lightgbm as lgb
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="🏏 Cricket ML Analytics API",
    description="Advanced Cricket Analytics with ML Predictions & Monte Carlo Simulation",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

DB_PATH = "IPL_cricket.db"

def get_db():
    """Get database connection"""
    return duckdb.connect(DB_PATH, read_only=False)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class MatchState(BaseModel):
    """Match state for predictions"""
    match_id: str
    venue: str
    city: str
    inning: int
    batting_team: str
    bowling_team: str
    over: int
    ball_in_over: int
    runs_so_far: int
    wickets_so_far: int
    batter: str
    non_striker: str
    bowler: str
    target: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "match_id": "match_001",
                "venue": "Wankhede Stadium",
                "city": "Mumbai",
                "inning": 2,
                "batting_team": "Mumbai Indians",
                "bowling_team": "Chennai Super Kings",
                "over": 15,
                "ball_in_over": 3,
                "runs_so_far": 125,
                "wickets_so_far": 4,
                "batter": "Rohit Sharma",
                "non_striker": "Hardik Pandya",
                "bowler": "Ravindra Jadeja",
                "target": 180
            }
        }

class SimulationRequest(BaseModel):
    """Monte Carlo simulation request"""
    match_state: MatchState
    n_simulations: int = Field(default=5000, ge=100, le=10000)

class CustomXIRequest(BaseModel):
    """Custom XI comparison request"""
    team1_players: List[str] = Field(..., min_items=11, max_items=11)
    team2_players: List[str] = Field(..., min_items=11, max_items=11)
    venue: str

class PlayerH2HRequest(BaseModel):
    """Player head-to-head request"""
    batter: str
    bowler: str

# ============================================================================
# MODEL MANAGER (Singleton)
# ============================================================================

class ModelManager:
    """Manages ML models and provides predictions"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models = {}
            self.feature_stats = {}
            self.calibrators = {}
            self._load_feature_stats()
            self._load_models()
            ModelManager._initialized = True
    
    def _load_feature_stats(self):
        """Load feature statistics for default values"""
        try:
            conn = get_db()
            # Calculate stats directly from deliveries table (more reliable)
            query = """
            WITH first_inn_totals AS (
                SELECT 
                    d.match_id,
                    SUM(d.runs_total) as first_inn_total
                FROM deliveries d
                JOIN matches m ON d.match_id = m.match_id
                WHERE d.inning = 1
                GROUP BY d.match_id
            )
            SELECT 
                AVG(d.runs_batter) * 100 as avg_batsman_sr,
                AVG(d.runs_total) * 6 as avg_bowler_econ,
                AVG(f.first_inn_total) as avg_venue_score,
                AVG(CASE WHEN d.runs_batter >= 4 THEN 1.0 ELSE 0.0 END) * 100 as avg_boundary_rate
            FROM deliveries d
            LEFT JOIN first_inn_totals f ON d.match_id = f.match_id
            WHERE d.runs_batter IS NOT NULL
            LIMIT 1
            """
            result = conn.execute(query).fetchone()
            if result and result[0] is not None:
                self.feature_stats = {
                    'avg_batsman_sr': float(result[0]) if result[0] else 120.0,
                    'avg_bowler_econ': float(result[1]) if result[1] else 8.0,
                    'avg_venue_score': float(result[2]) if result[2] else 165.0,
                    'avg_boundary_rate': float(result[3]) if result[3] else 15.0
                }
            else:
                self.feature_stats = {
                    'avg_batsman_sr': 120.0,
                    'avg_bowler_econ': 8.0,
                    'avg_venue_score': 165.0,
                    'avg_boundary_rate': 15.0
                }
            conn.close()
            logger.info(f"✅ Loaded feature stats: {self.feature_stats}")
        except Exception as e:
            logger.warning(f"Could not load feature stats: {e}. Using defaults.")
            self.feature_stats = {
                'avg_batsman_sr': 120.0,
                'avg_bowler_econ': 8.0,
                'avg_venue_score': 165.0,
                'avg_boundary_rate': 15.0
            }
    
    def _load_models(self):
        """Load all trained models and calibrators"""
        try:
            models_path = Path("models")
            if not models_path.exists():
                logger.warning("Models directory not found. Using rule-based predictions.")
                return
            
            model_files = list(models_path.glob("*.txt"))
            
            # Load runs_class model
            runs_class_file = models_path / "runs_class.txt"
            if runs_class_file.exists():
                self.models['runs_class'] = lgb.Booster(model_file=str(runs_class_file))
                logger.info("✅ Loaded runs_class model")
            
            # Load wicket_flag model
            wicket_flag_file = models_path / "wicket_flag.txt"
            if wicket_flag_file.exists():
                self.models['wicket_flag'] = lgb.Booster(model_file=str(wicket_flag_file))
                logger.info("✅ Loaded wicket_flag model")
            
            # Load quantile models for runs_next_over
            runs_over_models = {}
            for q_file in models_path.glob("runs_next_over_*.txt"):
                q_name = q_file.stem.replace('runs_next_over_', '')
                runs_over_models[q_name] = lgb.Booster(model_file=str(q_file))
            if runs_over_models:
                self.models['runs_next_over'] = runs_over_models
                logger.info(f"✅ Loaded {len(runs_over_models)} runs_next_over quantile models")
            
            # Load quantile models for runs_batsman_next_over
            batsman_runs_models = {}
            for q_file in models_path.glob("runs_batsman_next_over_*.txt"):
                q_name = q_file.stem.replace('runs_batsman_next_over_', '')
                batsman_runs_models[q_name] = lgb.Booster(model_file=str(q_file))
            if batsman_runs_models:
                self.models['runs_batsman_next_over'] = batsman_runs_models
                logger.info(f"✅ Loaded {len(batsman_runs_models)} runs_batsman_next_over quantile models")
            
            # Load calibrators
            calibrators_file = models_path / "calibrators.pkl"
            if calibrators_file.exists():
                with open(calibrators_file, 'rb') as f:
                    self.calibrators = pickle.load(f)
                logger.info("✅ Loaded calibrators")
            
            logger.info(f"✅ Models loaded: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}. Using rule-based predictions.")
            self.models = {}
    
    def create_match_state_df(self, state: MatchState) -> pd.DataFrame:
        """Convert MatchState to DataFrame with all required features"""
        
        # Calculate derived values
        balls_elapsed = state.over * 6 + state.ball_in_over
        balls_remaining = 120 - balls_elapsed
        wickets_remaining = 10 - state.wickets_so_far
        
        current_run_rate = (state.runs_so_far / balls_elapsed * 6) if balls_elapsed > 0 else 0
        
        # Calculate required run rate for 2nd innings
        if state.inning == 2 and state.target:
            runs_needed = state.target - state.runs_so_far
            required_run_rate = (runs_needed / balls_remaining * 6) if balls_remaining > 0 else 0
            run_rate_pressure = required_run_rate - current_run_rate
        else:
            required_run_rate = 0
            run_rate_pressure = 0
        
        # Phase classification
        if state.over < 6:
            phase = "Powerplay"
            is_powerplay, is_middle_over, is_death_over = 1, 0, 0
        elif state.over >= 16:
            phase = "Death"
            is_powerplay, is_middle_over, is_death_over = 0, 0, 1
        else:
            phase = "Middle"
            is_powerplay, is_middle_over, is_death_over = 0, 1, 0
        
        # Get venue-specific features from database
        try:
            conn = get_db()
            # Get venue stats - calculate first innings totals per match
            venue_query = """
            WITH first_inn_totals AS (
                SELECT 
                    d.match_id,
                    SUM(d.runs_total) as first_inn_total
                FROM deliveries d
                JOIN matches m ON d.match_id = m.match_id
                WHERE m.venue = ? AND d.inning = 1
                GROUP BY d.match_id
            )
            SELECT 
                AVG(f.first_inn_total) as venue_avg_first_innings,
                AVG(d.runs_total) as venue_avg_rpb,
                AVG(CASE WHEN d.runs_batter >= 4 THEN 1.0 ELSE 0.0 END) * 100 as venue_boundary_rate,
                AVG(CASE WHEN d.wicket THEN 1.0 ELSE 0.0 END) * 100 as venue_wicket_rate
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            LEFT JOIN first_inn_totals f ON d.match_id = f.match_id
            WHERE m.venue = ?
            """
            venue_stats = conn.execute(venue_query, [state.venue, state.venue]).fetchone()
            
            if venue_stats and venue_stats[0]:
                venue_avg_first_innings = float(venue_stats[0])
                venue_avg_rpb = float(venue_stats[1])
                venue_boundary_rate = float(venue_stats[2])
                venue_wicket_rate = float(venue_stats[3])
            else:
                venue_avg_first_innings = self.feature_stats['avg_venue_score']
                venue_avg_rpb = 0.6
                venue_boundary_rate = self.feature_stats['avg_boundary_rate']
                venue_wicket_rate = 5.0
            
            # Get player stats with EMA-like calculations (simplified)
            batsman_query = """
            SELECT 
                AVG(runs_batter) as avg_rpb,
                AVG(runs_batter) * 100 as avg_sr,
                AVG(CASE WHEN runs_batter >= 4 THEN 1 ELSE 0 END) * 100 as boundary_pct,
                AVG(CASE WHEN runs_total = 0 THEN 1 ELSE 0 END) * 100 as dot_pct
            FROM deliveries
            WHERE batter = ?
            """
            batsman_stats = conn.execute(batsman_query, [state.batter]).fetchone()
            
            if batsman_stats and batsman_stats[0]:
                batsman_rpb = float(batsman_stats[0]) if batsman_stats[0] else 1.0
                batsman_sr = float(batsman_stats[1]) if batsman_stats[1] else 120.0
                batsman_boundary_pct = float(batsman_stats[2]) if batsman_stats[2] else 12.0
                batsman_dot_pct = float(batsman_stats[3]) if batsman_stats[3] else 35.0
            else:
                batsman_rpb = 1.0
                batsman_sr = self.feature_stats['avg_batsman_sr']
                batsman_boundary_pct = 12.0
                batsman_dot_pct = 35.0
            
            bowler_query = """
            SELECT 
                AVG(runs_total) as avg_rpb,
                AVG(runs_total) * 6 as avg_econ,
                AVG(CASE WHEN wicket THEN 1 ELSE 0 END) * 100 as wicket_rate,
                AVG(CASE WHEN runs_total = 0 THEN 1 ELSE 0 END) * 100 as dot_pct
            FROM deliveries
            WHERE bowler = ?
            """
            bowler_stats = conn.execute(bowler_query, [state.bowler]).fetchone()
            
            if bowler_stats and bowler_stats[0]:
                bowler_rpb = float(bowler_stats[0]) if bowler_stats[0] else 1.0
                bowler_econ = float(bowler_stats[1]) if bowler_stats[1] else 8.0
                bowler_wicket_rate = float(bowler_stats[2]) if bowler_stats[2] else 3.0
                bowler_dot_pct = float(bowler_stats[3]) if bowler_stats[3] else 40.0
            else:
                bowler_rpb = 1.0
                bowler_econ = self.feature_stats['avg_bowler_econ']
                bowler_wicket_rate = 3.0
                bowler_dot_pct = 40.0
            
            # Get head-to-head stats
            h2h_query = """
            SELECT 
                COUNT(*) as h2h_balls,
                AVG(runs_total) as h2h_rpb,
                AVG(CASE WHEN wicket THEN 1 ELSE 0 END) * 100 as h2h_dismissal_rate,
                AVG(runs_batter) * 100 as h2h_strike_rate
            FROM deliveries
            WHERE batter = ? AND bowler = ?
            """
            h2h_stats = conn.execute(h2h_query, [state.batter, state.bowler]).fetchone()
            
            if h2h_stats and h2h_stats[0]:
                h2h_balls = int(h2h_stats[0])
                h2h_rpb = float(h2h_stats[1]) if h2h_stats[1] else 1.0
                h2h_dismissal_rate = float(h2h_stats[2]) if h2h_stats[2] else 0.0
                h2h_strike_rate = float(h2h_stats[3]) if h2h_stats[3] else 100.0
            else:
                h2h_balls = 0
                h2h_rpb = 1.0
                h2h_dismissal_rate = 0.0
                h2h_strike_rate = 100.0
            
            # Get current inning stats for batsman and bowler
            batsman_innings_query = """
            SELECT COUNT(*) as balls_faced_this_inning
            FROM deliveries
            WHERE batter = ? AND match_id = ? AND inning = ?
            """
            batsman_innings_stats = conn.execute(batsman_innings_query, [state.batter, state.match_id, state.inning]).fetchone()
            batsman_balls_faced_this_inning = int(batsman_innings_stats[0]) if batsman_innings_stats and batsman_innings_stats[0] else 0
            
            bowler_innings_query = """
            SELECT COUNT(*) as balls_bowled_this_inning
            FROM deliveries
            WHERE bowler = ? AND match_id = ? AND inning = ?
            """
            bowler_innings_stats = conn.execute(bowler_innings_query, [state.bowler, state.match_id, state.inning]).fetchone()
            bowler_balls_bowled_this_inning = int(bowler_innings_stats[0]) if bowler_innings_stats and bowler_innings_stats[0] else 0
            
            # Calculate partnership runs (simplified - use current runs)
            partnership_runs = state.runs_so_far
            
            # Calculate runs and RR in last 3 overs (simplified)
            runs_last_3_overs = max(0, state.runs_so_far - (state.runs_so_far * 0.8))
            wickets_last_3_overs = min(state.wickets_so_far, 2)
            rr_last_3_overs = current_run_rate * 1.1 if balls_elapsed > 18 else current_run_rate
            rr_acceleration = 0.0  # Simplified
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Error fetching stats: {e}. Using defaults.")
            venue_avg_first_innings = self.feature_stats['avg_venue_score']
            venue_avg_rpb = 1.0
            venue_boundary_rate = self.feature_stats['avg_boundary_rate']
            venue_wicket_rate = 5.0
            batsman_rpb = 1.0
            batsman_sr = self.feature_stats['avg_batsman_sr']
            batsman_boundary_pct = 12.0
            batsman_dot_pct = 35.0
            bowler_rpb = 1.0
            bowler_econ = self.feature_stats['avg_bowler_econ']
            bowler_wicket_rate = 3.0
            bowler_dot_pct = 40.0
            h2h_balls = 0
            h2h_rpb = 1.0
            h2h_dismissal_rate = 0.0
            h2h_strike_rate = 100.0
            batsman_balls_faced_this_inning = 0
            bowler_balls_bowled_this_inning = 0
            partnership_runs = state.runs_so_far
            runs_last_3_overs = 0
            wickets_last_3_overs = 0
            rr_last_3_overs = current_run_rate
            rr_acceleration = 0.0
        
        # Create feature dictionary
        data = {
            # Match state
            'over': state.over,
            'ball_in_over': state.ball_in_over,
            'inning': state.inning,
            'balls_elapsed': balls_elapsed,
            'balls_remaining': balls_remaining,
            'runs_so_far': state.runs_so_far,
            'wickets_so_far': state.wickets_so_far,
            'wickets_remaining': wickets_remaining,
            'current_run_rate': current_run_rate,
            'required_run_rate': required_run_rate,
            'run_rate_pressure': run_rate_pressure,
            
            # Phase
            'phase': phase,
            'is_powerplay': is_powerplay,
            'is_middle_over': is_middle_over,
            'is_death_over': is_death_over,
            'late_in_over': 1 if state.ball_in_over >= 4 else 0,
            
            # Venue
            'venue': state.venue,
            'city': state.city,
            'venue_avg_first_innings': venue_avg_first_innings,
            'venue_avg_rpb': venue_avg_rpb,
            'venue_boundary_rate': venue_boundary_rate,
            'venue_wicket_rate': venue_wicket_rate,
            
            # Players
            'batter': state.batter,
            'bowler': state.bowler,
            'non_striker': state.non_striker,
            
            # Enhanced temporal features (defaults)
            'is_modern_cricket': 1,
            'venue_season_avg_rpb': venue_avg_rpb,
            'venue_season_rpb_volatility': 0.15,
            'venue_season_wicket_rate': venue_wicket_rate,
            'venue_season_boundary_rate': venue_boundary_rate,
            'venue_evolution_runs': 0.0,
            'venue_evolution_wickets': 0.0,
            
            # Career phase
            'batsman_career_balls': 500,
            'bowler_career_balls': 300,
            'batsman_maturity': 0.6,
            'bowler_maturity': 0.5,
            'batsman_career_phase': 'Experienced',
            'bowler_career_phase': 'Experienced',
            'cricket_era': 'Modern_T20',
            
            # Form & fatigue
            'batsman_days_since_last_match': 5,
            'bowler_days_since_last_match': 5,
            'batsman_rest_quality': 0.6,
            'bowler_rest_quality': 0.6,
            'batsman_workload_30d': 5,
            'bowler_workload_30d': 5,
            'batsman_form_score': batsman_sr,
            'bowler_form_score': 100 - bowler_econ * 10,
            
            # EMA rolling stats - Model expects names WITHOUT "_ema_" prefix
            'batsman_rpb_short': batsman_rpb,
            'batsman_rpb_medium': batsman_rpb,
            'batsman_rpb_long': batsman_rpb,
            'batsman_sr_short': batsman_sr,
            'batsman_sr_medium': batsman_sr,
            'batsman_sr_long': batsman_sr,
            'batsman_boundary_pct_short': batsman_boundary_pct,
            'batsman_boundary_pct_medium': batsman_boundary_pct,
            'batsman_dot_pct_short': batsman_dot_pct,
            'batsman_dot_pct_medium': batsman_dot_pct,
            
            'bowler_econ_short': bowler_econ,
            'bowler_econ_medium': bowler_econ,
            'bowler_econ_long': bowler_econ,
            'bowler_wicket_rate_short': bowler_wicket_rate,
            'bowler_wicket_rate_medium': bowler_wicket_rate,
            'bowler_dot_pct_short': bowler_dot_pct,
            'bowler_dot_pct_medium': bowler_dot_pct,
            
            # Head-to-head features
            'h2h_balls': h2h_balls,
            'h2h_rpb': h2h_rpb,
            'h2h_dismissal_rate': h2h_dismissal_rate,
            'h2h_strike_rate': h2h_strike_rate,
            
            # Current inning features
            'runs_last_3_overs': runs_last_3_overs,
            'wickets_last_3_overs': wickets_last_3_overs,
            'rr_last_3_overs': rr_last_3_overs,
            'rr_acceleration': rr_acceleration,
            'batsman_balls_faced_this_inning': batsman_balls_faced_this_inning,
            'bowler_balls_bowled_this_inning': bowler_balls_bowled_this_inning,
            'partnership_runs': partnership_runs
        }
        
        df = pd.DataFrame([data])
        
        # Get model's expected features if model is loaded (for validation)
        expected_features = None
        if 'runs_class' in self.models:
            try:
                expected_features = self.models['runs_class'].feature_name()
                logger.debug(f"Model expects {len(expected_features)} features")
            except:
                pass
        
        # Ensure categorical features are strings (but don't include them in prediction)
        # Note: Model uses only numerical features, categoricals are for reference only
        categorical_cols = ['venue', 'city', 'phase', 'batter', 'bowler', 'non_striker', 
                           'batsman_career_phase', 'bowler_career_phase', 'cricket_era']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Validate features if model is loaded
        if expected_features:
            missing_features = [f for f in expected_features if f not in df.columns]
            if missing_features:
                logger.error(f"❌ Missing features: {missing_features}")
                # Add missing features with default values
                for feat in missing_features:
                    df[feat] = 0.0
                    logger.warning(f"Added missing feature '{feat}' with default value 0.0")
            
            extra_features = [f for f in df.columns if f not in expected_features and f not in categorical_cols]
            if extra_features:
                logger.debug(f"Extra features (will be ignored): {extra_features}")
        
        return df
    
    def monte_carlo_simulation(self, state: MatchState, n_sims: int = 5000) -> Dict:
        """
        Run Monte Carlo simulation from current match state
        
        Uses simple probabilistic model based on current conditions
        """
        logger.info(f"Running {n_sims} Monte Carlo simulations...")
        
        # Get current state
        balls_remaining = 120 - (state.over * 6 + state.ball_in_over)
        wickets_remaining = 10 - state.wickets_so_far
        current_score = state.runs_so_far
        
        # Calculate base run rate from phase
        if state.over < 6:
            base_rr = 7.5
        elif state.over >= 16:
            base_rr = 10.0
        else:
            base_rr = 8.5
        
        # Adjust for wickets lost
        wicket_penalty = state.wickets_so_far * 0.3
        adjusted_rr = max(base_rr - wicket_penalty, 5.0)
        
        # Simulate final scores
        final_scores = []
        wickets_lost_list = []
        
        for _ in range(n_sims):
            sim_score = current_score
            sim_wickets = state.wickets_so_far
            sim_balls = balls_remaining
            
            while sim_balls > 0 and sim_wickets < 10:
                # Wicket probability (increases with overs bowled)
                wicket_prob = 0.05 + (sim_wickets * 0.01)
                
                if np.random.random() < wicket_prob:
                    sim_wickets += 1
                    runs = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
                else:
                    # Runs distribution
                    runs = np.random.choice(
                        [0, 1, 2, 3, 4, 6],
                        p=[0.35, 0.30, 0.15, 0.05, 0.10, 0.05]
                    )
                
                sim_score += runs
                sim_balls -= 1
                
                # Early exit for 2nd innings if target reached
                if state.inning == 2 and state.target and sim_score >= state.target:
                    break
            
            final_scores.append(sim_score)
            wickets_lost_list.append(sim_wickets)
        
        final_scores = np.array(final_scores)
        wickets_lost = np.array(wickets_lost_list)
        
        # Calculate statistics
        results = {
            'n_simulations': n_sims,
            'current_score': current_score,
            'current_wickets': state.wickets_so_far,
            'balls_remaining': balls_remaining,
            'score_distribution': {
                'mean': float(np.mean(final_scores)),
                'median': float(np.median(final_scores)),
                'std': float(np.std(final_scores)),
                'min': int(np.min(final_scores)),
                'max': int(np.max(final_scores)),
                'q05': float(np.percentile(final_scores, 5)),
                'q25': float(np.percentile(final_scores, 25)),
                'q75': float(np.percentile(final_scores, 75)),
                'q95': float(np.percentile(final_scores, 95))
            },
            'wickets_distribution': {
                'mean': float(np.mean(wickets_lost)),
                'median': float(np.median(wickets_lost))
            }
        }
        
        # Win probability for 2nd innings
        if state.inning == 2 and state.target:
            wins = np.sum(final_scores >= state.target)
            win_prob = wins / n_sims
            results['win_probability'] = float(win_prob)
            results['target'] = state.target
            results['runs_needed'] = max(0, state.target - current_score)
            results['required_run_rate'] = (results['runs_needed'] / balls_remaining * 6) if balls_remaining > 0 else 0
        
        logger.info(f"Simulation complete. Expected score: {results['score_distribution']['median']:.0f}")
        
        return results

# Initialize model manager
model_manager = ModelManager()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "🏏 Cricket ML Analytics API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "stats": "/stats/*",
            "predictions": "/predict/*",
            "simulation": "/simulate",
            "analysis": "/analysis/*",
            "teams": "/teams/*",
            "players": "/players/*",
            "venues": "/venues/*"
        }
    }

# ============================================================================
# STATS ENDPOINTS
# ============================================================================

@app.get("/stats/overview")
async def get_overview_stats():
    """Get database overview statistics"""
    try:
        conn = get_db()
        
        stats = {
            'total_matches': conn.execute("SELECT COUNT(DISTINCT match_id) FROM matches").fetchone()[0],
            'total_deliveries': conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0],
            'unique_players': conn.execute("SELECT COUNT(DISTINCT batter) FROM deliveries").fetchone()[0],
            'unique_venues': conn.execute("SELECT COUNT(DISTINCT venue) FROM matches").fetchone()[0],
            'total_series': conn.execute("SELECT COUNT(DISTINCT series_name) FROM matches").fetchone()[0],
            'date_range': conn.execute("SELECT MIN(date), MAX(date) FROM matches").fetchone()
        }
        
        conn.close()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/season/{season}")
async def get_season_stats(season: str):
    """Get statistics for a specific season"""
    try:
        conn = get_db()
        
        query = """
        SELECT 
            COUNT(DISTINCT match_id) as matches,
            COUNT(DISTINCT winner) as teams,
            SUM(CASE WHEN margin LIKE '%wickets%' THEN 1 ELSE 0 END) as batting_first_wins,
            SUM(CASE WHEN margin LIKE '%runs%' THEN 1 ELSE 0 END) as chasing_wins
        FROM matches
        WHERE season = ?
        """
        
        result = conn.execute(query, [season]).fetchone()
        conn.close()
        
        if result:
            return {
                'season': season,
                'total_matches': result[0],
                'teams_participated': result[1],
                'batting_first_wins': result[2],
                'chasing_wins': result[3]
            }
        else:
            raise HTTPException(status_code=404, detail=f"Season {season} not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/predict/ball")
async def predict_next_ball(state: MatchState):
    """
    Predict outcome of next ball
    
    Returns probabilities for different run outcomes and wicket probability
    """
    try:
        # Create feature dataframe
        state_df = model_manager.create_match_state_df(state)
        
        # Use ML models if available, otherwise fall back to rule-based
        if 'runs_class' in model_manager.models and 'wicket_flag' in model_manager.models:
            try:
                # Get model predictions
                runs_class_model = model_manager.models['runs_class']
                wicket_model = model_manager.models['wicket_flag']
                
                # Prepare features for prediction (ensure correct order)
                model_features = runs_class_model.feature_name()
                feature_dict = state_df.iloc[0].to_dict()
                
                # Validate feature count
                if len(model_features) != len([f for f in model_features if f in feature_dict]):
                    missing = [f for f in model_features if f not in feature_dict]
                    logger.error(f"❌ Missing features in prediction: {missing}")
                    logger.error(f"Available features: {list(feature_dict.keys())[:10]}...")
                
                # Extract features in exact order model expects
                feature_vector = []
                for f in model_features:
                    if f in feature_dict:
                        val = feature_dict[f]
                        # Convert to float, handle None
                        if val is None:
                            val = 0.0
                        elif isinstance(val, (str, bool)):
                            val = 1.0 if val else 0.0
                        feature_vector.append(float(val))
                    else:
                        logger.warning(f"Feature '{f}' not found, using 0.0")
                        feature_vector.append(0.0)
                
                feature_array = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
                
                # Validate array shape
                if feature_array.shape[1] != len(model_features):
                    raise ValueError(f"Feature array shape {feature_array.shape} doesn't match expected {len(model_features)} features")
                
                # Predict runs class probabilities (9 classes: 0-6, wicket, extra)
                # LightGBM multiclass returns probabilities directly
                runs_probs_raw = runs_class_model.predict(feature_array, predict_disable_shape_check=True)
                if len(runs_probs_raw.shape) > 1:
                    runs_probs = runs_probs_raw[0]
                else:
                    runs_probs = runs_probs_raw
                
                # Predict wicket probability separately
                wicket_probs_raw = wicket_model.predict(feature_array, predict_disable_shape_check=True)
                if len(wicket_probs_raw.shape) > 1:
                    wicket_probs = wicket_probs_raw[0]
                else:
                    wicket_probs = wicket_probs_raw
                # Wicket model is binary: [no_wicket_prob, wicket_prob]
                wicket_prob = float(wicket_probs[1]) if len(wicket_probs) > 1 else float(wicket_probs[0]) if len(wicket_probs) == 1 else 0.05
                
                # Normalize probabilities
                runs_probs = np.clip(runs_probs, 0, 1)
                runs_probs = runs_probs / (runs_probs.sum() + 1e-10)
                
                # Map to outcome probabilities (classes: 0, 1, 2, 3, 4, 5, 6, 7=wicket, 8=extra)
                # Note: Class 5 is 5 runs (rare in cricket), we'll merge it with 4 or 6
                outcome_probs = {
                    '0': float(runs_probs[0]) if len(runs_probs) > 0 else 0.0,
                    '1': float(runs_probs[1]) if len(runs_probs) > 1 else 0.0,
                    '2': float(runs_probs[2]) if len(runs_probs) > 2 else 0.0,
                    '3': float(runs_probs[3]) if len(runs_probs) > 3 else 0.0,
                    '4': float(runs_probs[4]) + (float(runs_probs[5]) * 0.5 if len(runs_probs) > 5 else 0.0),  # Merge 5 runs into 4
                    '6': float(runs_probs[6]) + (float(runs_probs[5]) * 0.5 if len(runs_probs) > 5 else 0.0),  # Merge 5 runs into 6
                    'wicket': float(runs_probs[7]) if len(runs_probs) > 7 else wicket_prob,
                    'extra': float(runs_probs[8]) if len(runs_probs) > 8 else 0.02
                }
                
                # Combine wicket probabilities
                outcome_probs['wicket'] = max(outcome_probs.get('wicket', 0), wicket_prob * 0.5)
                
                # Normalize
                total = sum(outcome_probs.values())
                if total > 0:
                    outcome_probs = {k: v / total for k, v in outcome_probs.items()}
                
                most_likely_idx = np.argmax(runs_probs[:7])
                most_likely = int(most_likely_idx) if most_likely_idx < 7 else 0
                confidence = float(max(outcome_probs['0'], outcome_probs['1'], outcome_probs['2'], 
                                     outcome_probs['3'], outcome_probs['4'], outcome_probs['6']))
                model_used = 'ML'
                
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}. Using rule-based fallback.")
                outcome_probs = None
                model_used = 'Rule-based (fallback)'
        else:
            outcome_probs = None
            model_used = 'Rule-based'
        
        # Fallback to rule-based prediction
        if outcome_probs is None:
            if state.over < 6:  # Powerplay
                probs = [0.30, 0.25, 0.15, 0.05, 0.15, 0.05, 0.03, 0.02]
            elif state.over >= 16:  # Death overs
                probs = [0.25, 0.20, 0.12, 0.03, 0.20, 0.12, 0.05, 0.03]
            else:  # Middle overs
                probs = [0.35, 0.28, 0.15, 0.05, 0.10, 0.04, 0.02, 0.01]
            
            if state.inning == 2 and state.target:
                runs_needed = state.target - state.runs_so_far
                balls_left = 120 - (state.over * 6 + state.ball_in_over)
                if balls_left > 0:
                    rrr = runs_needed / balls_left * 6
                    if rrr > 12:
                        probs[7] *= 1.5
                        probs = [p / sum(probs) for p in probs]
            
            outcome_probs = {
                '0': probs[0],
                '1': probs[1],
                '2': probs[2],
                '3': probs[3],
                '4': probs[4],
                '6': probs[5],
                'wicket': probs[6] if len(probs) > 6 else 0.05,
                'extra': probs[7] if len(probs) > 7 else 0.02
            }
            most_likely = int(np.argmax(probs[:6]))
            confidence = max(probs[:6])
            wicket_prob = outcome_probs['wicket']
        else:
            wicket_prob = outcome_probs['wicket']
        
        return {
            'runs_probabilities': outcome_probs,
            'most_likely': most_likely,
            'confidence': confidence,
            'wicket_probability': wicket_prob,
            'match_context': {
                'phase': 'Powerplay' if state.over < 6 else ('Death' if state.over >= 16 else 'Middle'),
                'pressure': 'High' if state.inning == 2 and state.target and (state.target - state.runs_so_far) / max(1, 120 - state.over * 6 - state.ball_in_over) * 6 > 12 else 'Normal',
                'model_used': model_used
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/over")
async def predict_next_over(state: MatchState):
    """Predict runs in next 6 balls"""
    try:
        # Simulate next 6 balls
        total_expected = 0
        ball_predictions = []
        
        for ball_num in range(6):
            ball_state = MatchState(
                **{**state.dict(), 
                   'over': state.over if state.ball_in_over + ball_num < 6 else state.over + 1,
                   'ball_in_over': (state.ball_in_over + ball_num) % 6}
            )
            
            ball_pred = await predict_next_ball(ball_state)
            expected = sum(int(k) * v for k, v in ball_pred['runs_probabilities'].items() if k.isdigit())
            total_expected += expected
            
            ball_predictions.append({
                'ball_number': ball_num + 1,
                'expected_runs': round(expected, 2),
                'wicket_probability': ball_pred['wicket_probability']
            })
        
        return {
            'expected_runs': round(total_expected, 1),
            'range': {
                'min': max(0, total_expected - 6),
                'max': min(36, total_expected + 8)
            },
            'ball_by_ball': ball_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

@app.post("/simulate")
async def run_simulation(request: SimulationRequest):
    """
    Run Monte Carlo simulation from current match state
    
    Returns expected final score distribution and win probability
    """
    try:
        results = model_manager.monte_carlo_simulation(
            request.match_state,
            request.n_simulations
        )
        
        return {
            'success': True,
            'simulation_results': results,
            'interpretation': {
                'expected_score': f"{results['score_distribution']['median']:.0f}",
                'likely_range': f"{results['score_distribution']['q25']:.0f} - {results['score_distribution']['q75']:.0f}",
                'win_probability': f"{results.get('win_probability', 0) * 100:.1f}%" if 'win_probability' in results else None
            }
        }
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TEAM & PLAYER ENDPOINTS
# ============================================================================

@app.get("/teams/list")
async def get_teams():
    """Get list of all teams"""
    try:
        conn = get_db()
        query = """
        SELECT DISTINCT team
        FROM (
            SELECT UNNEST(teams) as team
            FROM matches
        ) t
        WHERE team IS NOT NULL
        ORDER BY team
        """
        teams = [row[0] for row in conn.execute(query).fetchall()]
        conn.close()
        
        return {'teams': teams}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teams/vs")
async def team_head_to_head(team1: str, team2: str, venue: Optional[str] = None):
    """Get head-to-head statistics between two teams"""
    try:
        conn = get_db()
        
        venue_filter = f"AND venue = '{venue}'" if venue else ""
        
        query = f"""
        SELECT 
            COUNT(*) as total_matches,
            SUM(CASE WHEN winner = '{team1}' THEN 1 ELSE 0 END) as team1_wins,
            SUM(CASE WHEN winner = '{team2}' THEN 1 ELSE 0 END) as team2_wins,
            SUM(CASE WHEN winner NOT IN ('{team1}', '{team2}') THEN 1 ELSE 0 END) as draws,
            MIN(date) as first_match,
            MAX(date) as last_match
        FROM matches
        WHERE (teams[1] = '{team1}' OR teams[2] = '{team1}')
          AND (teams[1] = '{team2}' OR teams[2] = '{team2}')
          {venue_filter}
        """
        
        result = conn.execute(query).fetchone()
        conn.close()
        
        if result and result[0] > 0:
            return {
                'team1': team1,
                'team2': team2,
                'venue': venue if venue else 'All Venues',
                'stats': {
                    'total_matches': result[0],
                    'team1_wins': result[1],
                    'team2_wins': result[2],
                    'draws': result[3],
                    'first_match': str(result[4]),
                    'last_match': str(result[5])
                }
            }
        else:
            return {
                'message': f'No matches found between {team1} and {team2}',
                'stats': {
                    'total_matches': 0,
                    'team1_wins': 0,
                    'team2_wins': 0,
                    'draws': 0
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/teams/compare")
async def compare_custom_xi(request: CustomXIRequest):
    """Compare two custom XIs at a specific venue"""
    try:
        conn = get_db()
        
        # Get venue stats - calculate first innings total per match
        venue_query = """
        WITH first_inn_totals AS (
            SELECT 
                d.match_id,
                SUM(d.runs_total) as first_inn_total
            FROM deliveries d
            WHERE d.inning = 1
            GROUP BY d.match_id
        )
        SELECT 
            AVG(f.first_inn_total) as avg_score,
            COUNT(DISTINCT f.match_id) as matches
        FROM first_inn_totals f
        JOIN matches m ON f.match_id = m.match_id
        WHERE m.venue = ?
        """
        venue_result = conn.execute(venue_query, [request.venue]).fetchone()
        venue_avg = venue_result[0] if venue_result[0] else 165
        
        def get_team_stats(players):
            stats = []
            total_exp = 0
            total_sr = 0
            
            for player in players:
                player_query = """
                SELECT 
                    COUNT(*) as balls_faced,
                    SUM(runs_batter) as runs,
                    AVG(runs_batter) * 100 as strike_rate
                FROM deliveries d
                JOIN matches m ON d.match_id = m.match_id
                WHERE d.batter = ? AND m.venue = ?
                """
                result = conn.execute(player_query, [player, request.venue]).fetchone()
                
                if result and result[0]:
                    stats.append({
                        'player': player,
                        'balls_faced': result[0],
                        'runs': result[1],
                        'strike_rate': round(result[2], 2)
                    })
                    total_exp += result[0]
                    total_sr += result[2]
                else:
                    stats.append({
                        'player': player,
                        'balls_faced': 0,
                        'runs': 0,
                        'strike_rate': 120.0
                    })
                    total_sr += 120.0
            
            return {
                'stats': stats,
                'total_experience': total_exp,
                'avg_strike_rate': round(total_sr / len(players), 2)
            }
        
        team1_stats = get_team_stats(request.team1_players)
        team2_stats = get_team_stats(request.team2_players)
        
        conn.close()
        
        return {
            'venue': request.venue,
            'venue_avg_score': round(venue_avg, 0),
            'team1': team1_stats,
            'team2': team2_stats,
            'advantage': 'Team 1' if team1_stats['total_experience'] > team2_stats['total_experience'] else 'Team 2'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/players/list")
async def get_players(limit: int = 100):
    """Get list of players"""
    try:
        conn = get_db()
        query = f"""
        SELECT DISTINCT batter as player, COUNT(*) as balls_faced
        FROM deliveries
        GROUP BY batter
        ORDER BY balls_faced DESC
        LIMIT {limit}
        """
        players = [row[0] for row in conn.execute(query).fetchall()]
        conn.close()
        
        return {'players': players}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/players/head-to-head")
async def player_head_to_head(batter: str, bowler: str):
    """Get head-to-head stats between batter and bowler"""
    try:
        conn = get_db()
        
        query = """
        SELECT 
            COUNT(*) as balls_faced,
            SUM(runs_batter) as runs_scored,
            SUM(CASE WHEN wicket AND player_out = ? THEN 1 ELSE 0 END) as dismissals,
            SUM(CASE WHEN runs_batter = 4 THEN 1 ELSE 0 END) as fours,
            SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END) as sixes,
            SUM(CASE WHEN runs_total = 0 THEN 1 ELSE 0 END) as dots
        FROM deliveries
        WHERE batter = ? AND bowler = ?
        """
        
        result = conn.execute(query, [batter, batter, bowler]).fetchone()
        conn.close()
        
        if result and result[0] > 0:
            balls = result[0]
            runs = result[1]
            sr = (runs / balls * 100) if balls > 0 else 0
            avg_rpb = (runs / balls) if balls > 0 else 0
            
            return {
                'batter': batter,
                'bowler': bowler,
                'balls_faced': balls,
                'runs_scored': runs,
                'dismissals': result[2],
                'strike_rate': round(sr, 2),
                'avg_runs_per_ball': round(avg_rpb, 2),
                'fours': result[3],
                'sixes': result[4],
                'dots': result[5]
            }
        else:
            return {
                'message': f'No encounters found between {batter} and {bowler}',
                'balls_faced': 0
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# VENUE ENDPOINTS
# ============================================================================

@app.get("/venues/list")
async def get_venues():
    """Get list of all venues"""
    try:
        conn = get_db()
        query = """
        SELECT DISTINCT venue, city, COUNT(*) as matches
        FROM matches
        GROUP BY venue, city
        ORDER BY matches DESC
        """
        venues = [{'venue': row[0], 'city': row[1], 'matches': row[2]} 
                 for row in conn.execute(query).fetchall()]
        conn.close()
        
        return {'venues': venues}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/venues/stats")
async def get_venue_stats():
    """Get comprehensive venue statistics"""
    try:
        conn = get_db()
        
        query = """
        WITH first_inn_totals AS (
            SELECT 
                d.match_id,
                SUM(d.runs_total) as first_inn_total
            FROM deliveries d
            WHERE d.inning = 1
            GROUP BY d.match_id
        )
        SELECT 
            m.venue,
            m.city,
            COUNT(DISTINCT m.match_id) as matches_played,
            AVG(f.first_inn_total) as venue_avg_first_innings,
            AVG(d.runs_total) as venue_avg_rpb,
            AVG(CASE WHEN d.runs_batter >= 4 THEN 1.0 ELSE 0.0 END) * 100 as venue_boundary_rate,
            AVG(CASE WHEN d.wicket THEN 1.0 ELSE 0.0 END) * 100 as venue_wicket_rate
        FROM matches m
        JOIN deliveries d ON m.match_id = d.match_id
        LEFT JOIN first_inn_totals f ON d.match_id = f.match_id
        WHERE d.inning = 1
        GROUP BY m.venue, m.city
        HAVING COUNT(DISTINCT m.match_id) >= 5
        ORDER BY matches_played DESC
        """
        
        results = conn.execute(query).fetchall()
        conn.close()
        
        venue_stats = []
        for row in results:
            venue_stats.append({
                'venue': row[0],
                'city': row[1],
                'matches_played': row[2],
                'venue_avg_first_innings': round(row[3], 1) if row[3] else 0,
                'venue_avg_rpb': round(row[4], 2) if row[4] else 0,
                'venue_boundary_rate': round(row[5], 1) if row[5] else 0,
                'venue_wicket_rate': round(row[6], 2) if row[6] else 0
            })
        
        return venue_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/venues/analysis/{venue}")
async def analyze_venue(venue: str):
    """Detailed analysis of a specific venue"""
    try:
        # URL decode the venue name (FastAPI should do this automatically, but ensure it's handled)
        from urllib.parse import unquote
        venue_decoded = unquote(venue)
        
        conn = get_db()
        
        # Basic stats - use LIKE for partial matching in case of encoding issues
        basic_query = """
        SELECT 
            COUNT(DISTINCT match_id) as total_matches,
            COUNT(DISTINCT winner) as different_winners,
            AVG(CASE WHEN toss_winner = winner THEN 1.0 ELSE 0.0 END) * 100 as toss_win_correlation
        FROM matches
        WHERE venue = ? OR venue LIKE ?
        """
        # Try exact match first, then partial match
        venue_pattern = f"%{venue_decoded}%"
        basic_result = conn.execute(basic_query, [venue_decoded, venue_pattern]).fetchone()
        
        if not basic_result or basic_result[0] == 0:
            # Try to find the exact venue name from database
            find_venue_query = """
            SELECT DISTINCT venue
            FROM matches
            WHERE venue LIKE ? OR venue LIKE ?
            LIMIT 1
            """
            found_venue = conn.execute(find_venue_query, [venue_pattern, f"%{venue_decoded.split(',')[0]}%"]).fetchone()
            if found_venue:
                venue_decoded = found_venue[0]
                basic_result = conn.execute(basic_query, [venue_decoded, venue_decoded]).fetchone()
        
        if not basic_result or basic_result[0] == 0:
            raise HTTPException(status_code=404, detail=f"Venue '{venue_decoded}' not found in database")
        
        # Most successful team
        team_query = """
        SELECT winner, COUNT(*) as wins
        FROM matches
        WHERE (venue = ? OR venue LIKE ?) AND winner IS NOT NULL AND winner != 'No Result'
        GROUP BY winner
        ORDER BY wins DESC
        LIMIT 1
        """
        team_result = conn.execute(team_query, [venue_decoded, venue_pattern]).fetchone()
        
        # Team performance - DuckDB uses list_unpack for arrays
        perf_query = """
        SELECT 
            team,
            COUNT(*) as matches,
            SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) as wins
        FROM (
            SELECT 
                match_id,
                winner,
                UNNEST(teams) as team
            FROM matches
            WHERE venue = ? OR venue LIKE ?
        ) t
        GROUP BY team
        HAVING COUNT(*) >= 3
        ORDER BY wins DESC
        """
        perf_results = conn.execute(perf_query, [venue_decoded, venue_pattern]).fetchall()
        
        conn.close()
        
        team_performance = []
        for row in perf_results:
            matches = row[1]
            wins = row[2]
            team_performance.append({
                'team': row[0],
                'matches': matches,
                'wins': wins,
                'win_pct': round((wins / matches * 100) if matches > 0 else 0, 1)
            })
        
        return {
            'venue': venue_decoded,
            'stats': {
                'total_matches': basic_result[0] if basic_result else 0,
                'different_winners': basic_result[1] if basic_result else 0,
                'toss_win_correlation': round(basic_result[2], 1) if basic_result and basic_result[2] else 0,
                'most_successful_team': team_result[0] if team_result else 'N/A'
            },
            'team_performance': team_performance
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing venue '{venue}': {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing venue: {str(e)}")

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/analysis/win-probability-tracker")
async def win_probability_tracker(match_id: str):
    """Track win probability throughout a match (historical)"""
    try:
        conn = get_db()
        
        # Get match info
        match_query = "SELECT * FROM matches WHERE match_id = ?"
        match_info = conn.execute(match_query, [match_id]).fetchone()
        
        if not match_info:
            raise HTTPException(status_code=404, detail="Match not found")
        
        # Get ball-by-ball data
        query = """
        SELECT 
            over_num,
            ball_num,
            inning,
            runs_so_far,
            wickets_so_far
        FROM (
            SELECT 
                over_num,
                ball_num,
                inning,
                SUM(runs_total) OVER (PARTITION BY match_id, inning ORDER BY over_num, ball_num) as runs_so_far,
                SUM(CASE WHEN wicket THEN 1 ELSE 0 END) OVER (PARTITION BY match_id, inning ORDER BY over_num, ball_num) as wickets_so_far
            FROM deliveries
            WHERE match_id = ?
        )
        WHERE inning = 2
        ORDER BY over_num, ball_num
        """
        
        results = conn.execute(query, [match_id]).fetchall()
        conn.close()
        
        # Calculate win probability at each point (simplified)
        tracker = []
        first_innings_score = 0
        
        for row in results:
            over = row[0]
            ball = row[1]
            runs = row[3]
            wickets = row[4]
            
            if over == 0 and ball == 0:
                # Get first innings score
                first_innings_score = runs
            
            balls_remaining = 120 - (over * 6 + ball)
            runs_needed = first_innings_score - runs
            rrr = (runs_needed / balls_remaining * 6) if balls_remaining > 0 else 0
            
            # Simple win probability model
            if runs >= first_innings_score:
                win_prob = 100
            elif wickets >= 10:
                win_prob = 0
            elif balls_remaining == 0:
                win_prob = 0
            else:
                # Logistic model based on runs needed and resources
                resources = (balls_remaining / 120) * ((10 - wickets) / 10)
                win_prob = 100 / (1 + np.exp(-5 * (resources - runs_needed / first_innings_score)))
            
            tracker.append({
                'over': over,
                'ball': ball,
                'score': f"{runs}/{wickets}",
                'win_probability': round(win_prob, 1),
                'required_run_rate': round(rrr, 2)
            })
        
        return {
            'match_id': match_id,
            'target': first_innings_score,
            'tracker': tracker[::6]  # Every over
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/pressure-moments")
async def analyze_pressure_moments(match_id: str):
    """Find high-pressure moments in a match"""
    try:
        conn = get_db()
        
        query = """
        SELECT 
            over_num,
            ball_num,
            batter,
            bowler,
            runs_total,
            wicket,
            inning
        FROM deliveries
        WHERE match_id = ?
        ORDER BY inning, over_num, ball_num
        """
        
        results = conn.execute(query, [match_id]).fetchall()
        conn.close()
        
        pressure_moments = []
        
        for i, row in enumerate(results):
            # High pressure indicators:
            # 1. Wickets in death overs
            # 2. Big hits in death overs
            # 3. Close finishes
            
            over = row[0]
            ball = row[1]
            runs = row[4]
            wicket = row[5]
            inning = row[6]
            
            pressure_score = 0
            
            if over >= 16:  # Death overs
                pressure_score += 3
                if wicket:
                    pressure_score += 5
                if runs >= 6:
                    pressure_score += 4
            
            if over >= 18:
                pressure_score += 2
            
            if pressure_score >= 7:
                pressure_moments.append({
                    'over': over,
                    'ball': ball,
                    'inning': inning,
                    'batter': row[2],
                    'bowler': row[3],
                    'runs': runs,
                    'wicket': wicket,
                    'pressure_score': pressure_score,
                    'description': f"{'Wicket' if wicket else runs + ' runs'} by {row[2]} off {row[3]}"
                })
        
        return {
            'match_id': match_id,
            'pressure_moments': sorted(pressure_moments, key=lambda x: x['pressure_score'], reverse=True)[:10]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """API health check"""
    try:
        conn = get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        
        return {
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("="*60)
    logger.info("🏏 Cricket ML Analytics API Starting...")
    logger.info("="*60)
    
    # Check database
    try:
        conn = get_db()
        match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        delivery_count = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
        conn.close()
        
        logger.info(f"✅ Database connected: {match_count:,} matches, {delivery_count:,} deliveries")
        logger.info(f"✅ Model Manager initialized")
        if model_manager.models:
            logger.info(f"✅ Loaded {len(model_manager.models)} ML model(s): {list(model_manager.models.keys())}")
        else:
            logger.warning("⚠️  No ML models loaded - using rule-based predictions")
        logger.info("="*60)
        logger.info("API ready at http://localhost:8000")
        logger.info("Docs available at http://localhost:8000/docs")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)