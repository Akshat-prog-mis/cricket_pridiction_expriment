"""
Optimized Advanced Cricket ML Pipeline
======================================
Optimizations:
- DuckDB for base features and aggregations
- Chunked processing for player-specific rolling stats
- Downcast all numeric types
- Temp tables for intermediate results
- In-depth logging and progress tracking
- Pandas for EMA/rolling only where necessary
"""

import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from tqdm import tqdm
import gc

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedMLConfig:
    """Unified configuration combining both approaches"""
    db_path: str = "IPL_cricket.db"
    
    # Temporal dynamics
    season_decay_rate: float = 5.0
    modern_cricket_threshold: int = 2019
    
    # EMA smoothing spans
    ema_spans: Dict[str, int] = None
    
    # Traditional rolling windows
    lookback_balls: Dict[str, int] = None
    
    # Career phase thresholds
    career_phase_balls: Dict[str, int] = None
    
    # Training parameters
    min_balls_threshold: int = 10
    quantiles: List[float] = None
    
    # Processing parameters
    chunk_size: int = 50  # Players per chunk for rolling stats
    
    def __post_init__(self):
        if self.ema_spans is None:
            self.ema_spans = {
                'short': 15,
                'medium': 40,
                'long': 100
            }
        
        if self.lookback_balls is None:
            self.lookback_balls = {
                'batsman_short': 10,
                'batsman_medium': 50,
                'batsman_long': 200,
                'bowler_short': 20,
                'bowler_medium': 50,
                'bowler_long': 100
            }
        
        if self.career_phase_balls is None:
            self.career_phase_balls = {
                'rookie': 200,
                'experienced': 1000,
                'veteran': 1000
            }
        
        if self.quantiles is None:
            self.quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast all numeric columns to smallest possible dtype"""
    logger.info("🔽 Downcasting numeric types...")
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    # Integer columns
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Float columns
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    logger.info(f"   Memory: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df


class UnifiedFeatureEngineer:
    """
    Optimized unified feature engineering using DuckDB and chunked processing
    """
    
    def __init__(self, config: UnifiedMLConfig):
        self.config = config
        self.conn = duckdb.connect(config.db_path, read_only=False)
        
        logger.info("="*80)
        logger.info("🚀 OPTIMIZED CRICKET ML PIPELINE INITIALIZED")
        logger.info("="*80)
        logger.info(f"Database: {config.db_path}")
        logger.info(f"Chunk size: {config.chunk_size} players")
        logger.info("="*80)
    
    def create_base_features(self) -> pd.DataFrame:
        """Extract comprehensive base features with DuckDB"""
        logger.info("\n📊 STEP 1: Creating base features with DuckDB...")
        
        # Create temp table for per-over runs
        logger.info("   Creating per_over_runs temp table...")
        self.conn.execute("""
            CREATE OR REPLACE TEMP TABLE per_over_runs AS
            SELECT 
                match_id,
                inning,
                over_num,
                SUM(runs_total) AS over_runs
            FROM deliveries
            GROUP BY match_id, inning, over_num
        """)
        
        # Main feature query
        logger.info("   Executing main feature extraction query...")
        query = """
        WITH match_context AS (
            SELECT 
                d.*,
                m.venue,
                m.city,
                m.date,
                m.toss_winner,
                m.toss_decision,
                m.winner,
                m.season,
                m.teams,
                m.match_type,
                
                -- Match state
                d.over_num as over,
                d.ball_num as ball_in_over,
                d.inning,
                
                -- Phase classification
                CASE 
                    WHEN d.over_num < 6 THEN 'Powerplay'
                    WHEN d.over_num >= 16 THEN 'Death'
                    ELSE 'Middle'
                END as phase,
                
                CAST(CASE WHEN d.over_num < 6 THEN 1 ELSE 0 END AS TINYINT) as is_powerplay,
                CAST(CASE WHEN d.over_num >= 16 THEN 1 ELSE 0 END AS TINYINT) as is_death_over,
                CAST(CASE WHEN d.over_num >= 6 AND d.over_num < 16 THEN 1 ELSE 0 END AS TINYINT) as is_middle_over,
                CAST(CASE WHEN d.ball_num >= 4 THEN 1 ELSE 0 END AS TINYINT) as late_in_over,
                
                -- Running totals
                SUM(d.runs_total) OVER (
                    PARTITION BY d.match_id, d.inning 
                    ORDER BY d.over_num, d.ball_num
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) as runs_so_far,
                
                SUM(CASE WHEN d.wicket THEN 1 ELSE 0 END) OVER (
                    PARTITION BY d.match_id, d.inning 
                    ORDER BY d.over_num, d.ball_num
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) as wickets_so_far,
                
                ROW_NUMBER() OVER (
                    PARTITION BY d.match_id, d.inning 
                    ORDER BY d.over_num, d.ball_num
                ) as balls_elapsed,
                
                -- Balls/wickets remaining
                CASE 
                    WHEN m.match_type = 'T20' THEN 120
                    WHEN m.match_type = 'ODI' THEN 300
                    ELSE 120
                END - ROW_NUMBER() OVER (
                    PARTITION BY d.match_id, d.inning 
                    ORDER BY d.over_num, d.ball_num
                ) as balls_remaining,
                
                10 - SUM(CASE WHEN d.wicket THEN 1 ELSE 0 END) OVER (
                    PARTITION BY d.match_id, d.inning 
                    ORDER BY d.over_num, d.ball_num
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) as wickets_remaining,
                
                -- Previous over runs
                COALESCE(
                    LAG(por.over_runs, 1) OVER (
                        PARTITION BY d.match_id, d.inning 
                        ORDER BY d.over_num, d.ball_num
                    ), 0
                ) as runs_last_over,
                
                -- Partnership tracking
                SUM(d.runs_total) OVER (
                    PARTITION BY d.match_id, d.inning, d.batter, d.non_striker
                    ORDER BY d.over_num, d.ball_num
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) as partnership_runs,
                
                ROW_NUMBER() OVER (
                    PARTITION BY d.match_id, d.inning, d.batter, d.non_striker
                    ORDER BY d.over_num, d.ball_num
                ) as partnership_balls
                
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            LEFT JOIN per_over_runs por 
                ON d.match_id = por.match_id 
                AND d.inning = por.inning 
                AND d.over_num = por.over_num
            WHERE m.match_type IN ('ODI', 'T20')
        )
        SELECT * FROM match_context
        ORDER BY date, match_id, inning, over_num, ball_num
        """
        
        df = self.conn.execute(query).fetchdf()
        logger.info(f"   ✓ Extracted {len(df):,} rows, {len(df.columns)} columns")
        
        # Date conversion
        if df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'])
        
        # Add runs_batter if missing
        if 'runs_batter' not in df.columns:
            df['runs_batter'] = df['runs_total']
        
        # Calculate derived features
        logger.info("   Computing derived metrics...")
        df['current_run_rate'] = (df['runs_so_far'] / df['balls_elapsed'] * 6).fillna(0)
        df['required_run_rate'] = 0.0
        df['run_rate_pressure'] = 0.0
        df['target'] = 0
        
        # Second innings calculations
        second_inning_mask = df['inning'] == 2
        if second_inning_mask.any():
            first_inn_scores = df[df['inning'] == 1].groupby('match_id')['runs_so_far'].max()
            df.loc[second_inning_mask, 'target'] = df.loc[second_inning_mask, 'match_id'].map(first_inn_scores) + 1
            
            target_vals = df.loc[second_inning_mask, 'target']
            runs_needed = target_vals - df.loc[second_inning_mask, 'runs_so_far']
            balls_left = df.loc[second_inning_mask, 'balls_remaining']
            df.loc[second_inning_mask, 'required_run_rate'] = (runs_needed / balls_left * 6).fillna(0).clip(0, 36)
            df.loc[second_inning_mask, 'run_rate_pressure'] = (
                df.loc[second_inning_mask, 'required_run_rate'] - 
                df.loc[second_inning_mask, 'current_run_rate']
            )
        
        # Downcast
        df = downcast_dataframe(df)
        
        # Save to temp table
        logger.info("   Saving to temp table 'base_features'...")
        self.conn.execute("CREATE OR REPLACE TEMP TABLE base_features AS SELECT * FROM df")
        
        logger.info(f"   ✓ Base features complete: {len(df):,} rows")
        return df
    
    def add_temporal_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal weights using DuckDB"""
        logger.info("\n⏰ STEP 2: Adding temporal weights...")
        
        # Convert season to numeric
        if df['season'].dtype == 'object':
            df['season_numeric'] = df['season'].astype(str).str.extract(r'(\d{4})')[0].astype(int)
        else:
            df['season_numeric'] = df['season'].astype(int)
        
        max_season = df['season_numeric'].max()
        
        df['season_weight'] = np.exp((df['season_numeric'] - max_season) / self.config.season_decay_rate)
        df['is_modern_cricket'] = (df['season_numeric'] >= self.config.modern_cricket_threshold).astype(np.int8)
        
        df['cricket_era'] = pd.cut(
            df['season_numeric'],
            bins=[2007, 2013, 2019, 2025],
            labels=['Early_T20', 'Middle_T20', 'Modern_T20'],
            include_lowest=True
        ).astype(str)
        
        df = downcast_dataframe(df)
        
        logger.info(f"   ✓ Season range: {df['season_numeric'].min()} to {df['season_numeric'].max()}")
        logger.info(f"   ✓ Weight range: {df['season_weight'].min():.3f} to {df['season_weight'].max():.3f}")
        
        return df
    
    def add_rolling_features_chunked(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features using chunked processing for memory efficiency"""
        logger.info("\n📈 STEP 3: Computing rolling features (chunked)...")
        
        df = df.sort_values(['date', 'match_id', 'inning', 'over', 'ball_in_over'])
        
        # Process batsmen in chunks
        logger.info("   Processing batsmen rolling stats...")
        unique_batsmen = df['batter'].unique()
        num_batsmen = len(unique_batsmen)
        
        batsman_features = []
        for i in tqdm(range(0, num_batsmen, self.config.chunk_size), desc="   Batsmen chunks"):
            chunk_batsmen = unique_batsmen[i:i+self.config.chunk_size]
            chunk_df = df[df['batter'].isin(chunk_batsmen)].copy()
            
            for window, label in [
                (self.config.lookback_balls['batsman_short'], 'short'),
                (self.config.lookback_balls['batsman_medium'], 'medium'),
                (self.config.lookback_balls['batsman_long'], 'long')
            ]:
                chunk_df[f'batsman_rpb_{label}'] = chunk_df.groupby('batter')['runs_batter'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                ).fillna(0)
                
                chunk_df[f'batsman_sr_{label}'] = chunk_df[f'batsman_rpb_{label}'] * 100
                
                chunk_df[f'batsman_boundary_pct_{label}'] = chunk_df.groupby('batter').apply(
                    lambda x: ((x['runs_batter'] >= 4).astype(int).rolling(window=window, min_periods=1).mean().shift(1) * 100)
                ).reset_index(level=0, drop=True).fillna(0)
                
                chunk_df[f'batsman_dot_pct_{label}'] = chunk_df.groupby('batter').apply(
                    lambda x: ((x['runs_total'] == 0).astype(int).rolling(window=window, min_periods=1).mean().shift(1) * 100)
                ).reset_index(level=0, drop=True).fillna(30)
            
            batsman_features.append(chunk_df)
            del chunk_df
            gc.collect()
        
        df = pd.concat(batsman_features, ignore_index=True)
        del batsman_features
        gc.collect()
        
        logger.info("   ✓ Batsmen rolling stats complete")
        
        # Process bowlers in chunks
        logger.info("   Processing bowlers rolling stats...")
        unique_bowlers = df['bowler'].unique()
        num_bowlers = len(unique_bowlers)
        
        bowler_features = []
        for i in tqdm(range(0, num_bowlers, self.config.chunk_size), desc="   Bowler chunks"):
            chunk_bowlers = unique_bowlers[i:i+self.config.chunk_size]
            chunk_df = df[df['bowler'].isin(chunk_bowlers)].copy()
            
            for window, label in [
                (self.config.lookback_balls['bowler_short'], 'short'),
                (self.config.lookback_balls['bowler_medium'], 'medium'),
                (self.config.lookback_balls['bowler_long'], 'long')
            ]:
                chunk_df[f'bowler_econ_{label}'] = chunk_df.groupby('bowler')['runs_total'].transform(
                    lambda x: (x.rolling(window=window, min_periods=1).mean().shift(1) * 6)
                ).fillna(7)
                
                chunk_df[f'bowler_wicket_rate_{label}'] = chunk_df.groupby('bowler')['wicket'].transform(
                    lambda x: (x.astype(int).rolling(window=window, min_periods=1).mean().shift(1) * 100)
                ).fillna(2)
                
                chunk_df[f'bowler_dot_pct_{label}'] = chunk_df.groupby('bowler').apply(
                    lambda x: ((x['runs_total'] == 0).astype(int).rolling(window=window, min_periods=1).mean().shift(1) * 100)
                ).reset_index(level=0, drop=True).fillna(40)
            
            bowler_features.append(chunk_df)
            del chunk_df
            gc.collect()
        
        df = pd.concat(bowler_features, ignore_index=True)
        del bowler_features
        gc.collect()
        
        df = df.sort_values(['date', 'match_id', 'inning', 'over', 'ball_in_over'])
        df = downcast_dataframe(df)
        
        logger.info("   ✓ Bowlers rolling stats complete")
        logger.info(f"   ✓ Total rolling features: {sum('rpb' in col or 'econ' in col for col in df.columns)}")
        
        return df
    
    def add_ema_features_chunked(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA features using chunked processing"""
        logger.info("\n📉 STEP 4: Computing EMA features (chunked)...")
        
        df = df.sort_values(['date', 'match_id', 'inning', 'over', 'ball_in_over'])
        
        # Batsman EMA in chunks
        logger.info("   Processing batsmen EMA...")
        unique_batsmen = df['batter'].unique()
        num_batsmen = len(unique_batsmen)
        
        batsman_ema = []
        for i in tqdm(range(0, num_batsmen, self.config.chunk_size), desc="   Batsmen EMA"):
            chunk_batsmen = unique_batsmen[i:i+self.config.chunk_size]
            chunk_df = df[df['batter'].isin(chunk_batsmen)].copy()
            
            for span_name, span_val in self.config.ema_spans.items():
                chunk_df[f'batsman_rpb_ema_{span_name}'] = chunk_df.groupby('batter')['runs_batter'].transform(
                    lambda x: x.ewm(span=span_val, adjust=False).mean().shift(1)
                ).fillna(0.5)
                
                chunk_df[f'batsman_sr_ema_{span_name}'] = chunk_df[f'batsman_rpb_ema_{span_name}'] * 100
                
                chunk_df[f'batsman_boundary_pct_ema_{span_name}'] = chunk_df.groupby('batter').apply(
                    lambda x: ((x['runs_batter'] >= 4).astype(float).ewm(span=span_val, adjust=False).mean().shift(1) * 100)
                ).reset_index(level=0, drop=True).fillna(10)
                
                chunk_df[f'batsman_dot_pct_ema_{span_name}'] = chunk_df.groupby('batter').apply(
                    lambda x: ((x['runs_total'] == 0).astype(float).ewm(span=span_val, adjust=False).mean().shift(1) * 100)
                ).reset_index(level=0, drop=True).fillna(30)
            
            batsman_ema.append(chunk_df)
            del chunk_df
            gc.collect()
        
        df = pd.concat(batsman_ema, ignore_index=True)
        del batsman_ema
        gc.collect()
        
        # Bowler EMA in chunks
        logger.info("   Processing bowlers EMA...")
        unique_bowlers = df['bowler'].unique()
        num_bowlers = len(unique_bowlers)
        
        bowler_ema = []
        for i in tqdm(range(0, num_bowlers, self.config.chunk_size), desc="   Bowler EMA"):
            chunk_bowlers = unique_bowlers[i:i+self.config.chunk_size]
            chunk_df = df[df['bowler'].isin(chunk_bowlers)].copy()
            
            for span_name, span_val in self.config.ema_spans.items():
                chunk_df[f'bowler_econ_ema_{span_name}'] = chunk_df.groupby('bowler')['runs_total'].transform(
                    lambda x: (x.ewm(span=span_val, adjust=False).mean().shift(1) * 6)
                ).fillna(7)
                
                chunk_df[f'bowler_wicket_rate_ema_{span_name}'] = chunk_df.groupby('bowler')['wicket'].transform(
                    lambda x: (x.astype(float).ewm(span=span_val, adjust=False).mean().shift(1) * 100)
                ).fillna(2)
                
                chunk_df[f'bowler_dot_pct_ema_{span_name}'] = chunk_df.groupby('bowler').apply(
                    lambda x: ((x['runs_total'] == 0).astype(float).ewm(span=span_val, adjust=False).mean().shift(1) * 100)
                ).reset_index(level=0, drop=True).fillna(40)
            
            bowler_ema.append(chunk_df)
            del chunk_df
            gc.collect()
        
        df = pd.concat(bowler_ema, ignore_index=True)
        del bowler_ema
        gc.collect()
        
        df = df.sort_values(['date', 'match_id', 'inning', 'over', 'ball_in_over'])
        df = downcast_dataframe(df)
        
        logger.info("   ✓ EMA features complete")
        logger.info(f"   ✓ Total EMA features: {sum('ema' in col for col in df.columns)}")
        
        return df
    
    def add_venue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add venue features using DuckDB aggregations"""
        logger.info("\n🏟️  STEP 5: Adding venue features (DuckDB)...")
        
        # Save current df to temp table
        self.conn.execute("CREATE OR REPLACE TEMP TABLE current_df AS SELECT * FROM df")
        
        # Venue statistics via DuckDB
        logger.info("   Computing venue aggregations...")
        venue_stats_query = """
        SELECT
            m.venue,
            COUNT(DISTINCT d.match_id) as venue_matches,
            AVG(first_innings_runs.runs) as venue_avg_first_innings,
            AVG(d.runs_total) as venue_avg_rpb,
            AVG(CASE WHEN d.runs_batter >= 4 THEN 1.0 ELSE 0.0 END) as venue_boundary_rate,
            AVG(CASE WHEN d.wicket THEN 1.0 ELSE 0.0 END) as venue_wicket_rate
        FROM deliveries d
        JOIN matches m ON d.match_id = m.match_id
        LEFT JOIN (
            SELECT match_id, SUM(runs_total) as runs
            FROM deliveries
            WHERE inning = 1
            GROUP BY match_id
        ) first_innings_runs ON d.match_id = first_innings_runs.match_id
        GROUP BY m.venue
        """
        
        venue_stats = self.conn.execute(venue_stats_query).fetchdf()
        df = df.merge(venue_stats, on='venue', how='left')
        
        # Venue-season stats
        logger.info("   Computing venue-season statistics...")
        venue_season_stats = df.groupby(['venue', 'season_numeric']).agg({
            'runs_total': ['mean', 'std'],
            'wicket': 'mean',
            'runs_batter': lambda x: (x >= 4).mean()
        }).reset_index()
        
        venue_season_stats.columns = [
            'venue', 'season_numeric', 
            'venue_season_avg_rpb', 'venue_season_rpb_volatility',
            'venue_season_wicket_rate', 'venue_season_boundary_rate'
        ]
        
        df = df.merge(venue_season_stats, on=['venue', 'season_numeric'], how='left')
        
        # Venue evolution
        venue_historical = df.groupby('venue').agg({
            'runs_total': 'mean',
            'wicket': 'mean'
        }).reset_index()
        venue_historical.columns = ['venue', 'venue_historical_avg_rpb', 'venue_historical_wicket_rate']
        
        df = df.merge(venue_historical, on='venue', how='left')
        
        df['venue_evolution_runs'] = df['venue_season_avg_rpb'] - df['venue_historical_avg_rpb']
        df['venue_evolution_wickets'] = df['venue_season_wicket_rate'] - df['venue_historical_wicket_rate']
        
        # Fill missing
        venue_cols = [col for col in df.columns if 'venue' in col]
        for col in venue_cols:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].fillna(df[col].mean())
        
        df = downcast_dataframe(df)
        
        logger.info(f"   ✓ Venue features complete: {len(venue_cols)} features")
        
        return df
    
    def add_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup features using DuckDB"""
        logger.info("\n⚔️  STEP 6: Adding matchup features (DuckDB)...")
        
        matchup_query = """
        SELECT
            batter,
            bowler,
            COUNT(*) as h2h_balls,
            AVG(runs_batter) as h2h_rpb,
            AVG(CASE WHEN wicket AND player_out = batter THEN 1.0 ELSE 0.0 END) as h2h_dismissal_rate,
            SUM(runs_batter) * 100.0 / COUNT(*) as h2h_strike_rate
        FROM deliveries
        GROUP BY batter, bowler
        HAVING COUNT(*) >= 10
        """
        
        matchup_stats = self.conn.execute(matchup_query).fetchdf()
        df = df.merge(matchup_stats, on=['batter', 'bowler'], how='left')
        
        df['h2h_balls'] = df['h2h_balls'].fillna(0)
        df['h2h_rpb'] = df['h2h_rpb'].fillna(df.get('batsman_rpb_medium', 0.5))
        df['h2h_dismissal_rate'] = df['h2h_dismissal_rate'].fillna(0.02)
        df['h2h_strike_rate'] = df['h2h_strike_rate'].fillna(df.get('batsman_sr_medium', 100))
        
        df = downcast_dataframe(df)
        
        logger.info(f"   ✓ Matchup features complete")
        
        return df
    
    def add_career_phase_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add career phase features"""
        logger.info("\n🎯 STEP 7: Adding career phase features...")
        
        df['batsman_career_balls'] = df.groupby('batter').cumcount()
        df['bowler_career_balls'] = df.groupby('bowler').cumcount()
        
        df['batsman_career_phase'] = pd.cut(
            df['batsman_career_balls'],
            bins=[0, self.config.career_phase_balls['rookie'], 
                  self.config.career_phase_balls['experienced'], np.inf],
            labels=['Rookie', 'Experienced', 'Veteran']
        ).astype(str)
        
        df['bowler_career_phase'] = pd.cut(
            df['bowler_career_balls'],
            bins=[0, self.config.career_phase_balls['rookie'], 
                  self.config.career_phase_balls['experienced'], np.inf],
            labels=['Rookie', 'Experienced', 'Veteran']
        ).astype(str)
        
        df['batsman_maturity'] = np.tanh(df['batsman_career_balls'] / 500)
        df['bowler_maturity'] = np.tanh(df['bowler_career_balls'] / 300)
        
        df = downcast_dataframe(df)
        
        logger.info("   ✓ Career phase features complete")
        
        return df
    
    def add_form_fatigue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add form and fatigue features"""
        logger.info("\n💪 STEP 8: Adding form and fatigue features...")
        
        df['date'] = pd.to_datetime(df['date'])
        
        df = df.sort_values(['batter', 'date', 'match_id', 'inning', 'over', 'ball_in_over'])
        df['batsman_days_since_last_match'] = df.groupby('batter')['date'].diff().dt.days.fillna(7)
        
        df = df.sort_values(['bowler', 'date', 'match_id', 'inning', 'over', 'ball_in_over'])
        df['bowler_days_since_last_match'] = df.groupby('bowler')['date'].diff().dt.days.fillna(7)
        
        df['batsman_rest_quality'] = np.tanh(df['batsman_days_since_last_match'] / 5)
        df['bowler_rest_quality'] = np.tanh(df['bowler_days_since_last_match'] / 5)
        
        df['batsman_workload_30d'] = np.clip(30 / (df['batsman_days_since_last_match'] + 1), 0, 15)
        df['bowler_workload_30d'] = np.clip(30 / (df['bowler_days_since_last_match'] + 1), 0, 15)
        
        if 'batsman_sr_ema_short' in df.columns:
            df['batsman_form_score'] = (
                df['batsman_sr_ema_short'] * np.log1p(df['batsman_days_since_last_match'])
            )
        
        if 'bowler_econ_ema_short' in df.columns:
            df['bowler_form_score'] = (
                (100 - df['bowler_econ_ema_short'] * 10) * np.log1p(df['bowler_days_since_last_match'])
            )
        
        df = df.sort_values(['date', 'match_id', 'inning', 'over', 'ball_in_over'])
        df = downcast_dataframe(df)
        
        logger.info("   ✓ Form and fatigue features complete")
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        logger.info("\n🚀 STEP 9: Adding momentum and pressure features...")
        
        df['batsman_balls_faced_this_inning'] = df.groupby(['match_id', 'inning', 'batter']).cumcount()
        df['bowler_balls_bowled_this_inning'] = df.groupby(['match_id', 'inning', 'bowler']).cumcount()
        
        df['runs_last_3_overs'] = df.groupby(['match_id', 'inning'])['runs_total'].transform(
            lambda x: x.rolling(window=18, min_periods=1).sum().shift(1).fillna(0)
        )
        
        df['runs_last_5_overs'] = df.groupby(['match_id', 'inning'])['runs_total'].transform(
            lambda x: x.rolling(window=30, min_periods=1).sum().shift(1).fillna(0)
        )
        
        df['wickets_last_3_overs'] = df.groupby(['match_id', 'inning'])['wicket'].transform(
            lambda x: x.astype(int).rolling(window=18, min_periods=1).sum().shift(1).fillna(0)
        )
        
        df['rr_last_3_overs'] = (df['runs_last_3_overs'] / 18 * 6).fillna(0)
        df['rr_last_5_overs'] = (df['runs_last_5_overs'] / 30 * 6).fillna(0)
        df['rr_acceleration'] = df['rr_last_3_overs'] - df['current_run_rate']
        
        df['opponent_current_rr'] = 0.0
        second_inn_mask = df['inning'] == 2
        if second_inn_mask.any():
            first_inn_rr = df[df['inning'] == 1].groupby('match_id')['current_run_rate'].mean()
            df.loc[second_inn_mask, 'opponent_current_rr'] = df.loc[second_inn_mask, 'match_id'].map(first_inn_rr).fillna(0)
        
        df['momentum_differential'] = df['current_run_rate'] - df['opponent_current_rr']
        
        df['partnership_run_rate'] = (df['partnership_runs'] / df['partnership_balls'].clip(lower=1) * 6).fillna(6)
        df['partnership_stability'] = np.tanh(df['partnership_balls'] / 30)
        
        df['bowler_overs_bowled'] = (df['bowler_balls_bowled_this_inning'] // 6).astype(np.int8)
        df['bowler_is_fatigued'] = (df['bowler_overs_bowled'] >= 3).astype(np.int8)
        
        phase_multipliers = {'Powerplay': 1.0, 'Middle': 1.2, 'Death': 1.5}
        df['phase_multiplier'] = df['phase'].map(phase_multipliers)
        df['pressure_index'] = (df['run_rate_pressure'] * df['phase_multiplier']).fillna(0)
        
        df['batsman_is_settled'] = (df['batsman_balls_faced_this_inning'] >= 12).astype(np.int8)
        
        df['boundaries_last_10_balls'] = df.groupby(['match_id', 'inning']).apply(
            lambda x: ((x['runs_batter'] >= 4).astype(int).rolling(window=10, min_periods=1).sum().shift(1))
        ).reset_index(level=[0, 1], drop=True).fillna(0).astype(np.int8)
        
        df['dots_last_10_balls'] = df.groupby(['match_id', 'inning']).apply(
            lambda x: ((x['runs_total'] == 0).astype(int).rolling(window=10, min_periods=1).sum().shift(1))
        ).reset_index(level=[0, 1], drop=True).fillna(5).astype(np.int8)
        
        df = downcast_dataframe(df)
        
        logger.info("   ✓ Momentum and pressure features complete")
        
        return df
    
    def create_all_features(self) -> pd.DataFrame:
        """Complete unified feature engineering pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🏏 UNIFIED CRICKET ML FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # Step-by-step feature creation
        df = self.create_base_features()
        df = self.add_temporal_weights(df)
        df = self.add_rolling_features_chunked(df)
        df = self.add_ema_features_chunked(df)
        df = self.add_venue_features(df)
        df = self.add_matchup_features(df)
        df = self.add_career_phase_features(df)
        df = self.add_form_fatigue_features(df)
        df = self.add_momentum_features(df)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("✅ FEATURE ENGINEERING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Total rows: {len(df):,}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info("="*80)
        
        return df


class TargetCreator:
    """
    Creates multi-task targets for probabilistic prediction
    """
    
    @staticmethod
    def create_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Create all target variables"""
        logger.info("\n🎯 CREATING TARGET VARIABLES...")
        
        if 'runs_batter' not in df.columns:
            df['runs_batter'] = df['runs_total']
        
        # Target 1: runs_class
        df['runs_class'] = df['runs_batter'].apply(lambda x: min(x, 6))
        df.loc[df['wicket'], 'runs_class'] = 7
        
        has_extras = 'wides' in df.columns and 'noballs' in df.columns
        if has_extras:
            df.loc[(df['wides'] > 0) | (df['noballs'] > 0), 'runs_class'] = 8
        else:
            df.loc[(df['runs_total'] > df['runs_batter']) & (~df['wicket']), 'runs_class'] = 8
        
        df['runs_class'] = df['runs_class'].astype(np.int8)
        
        # Target 2: wicket_flag
        df['wicket_flag'] = df['wicket'].astype(np.int8)
        
        # Target 3: runs_next_over
        df['runs_next_over'] = df.groupby(['match_id', 'inning'])['runs_total'].transform(
            lambda x: x[::-1].rolling(window=6, min_periods=1).sum()[::-1].shift(-6).fillna(0)
        ).clip(0, 36)
        
        # Target 4: runs_batsman_next_over
        df['runs_batsman_next_over'] = df.groupby(['match_id', 'inning', 'batter'])['runs_batter'].transform(
            lambda x: x[::-1].rolling(window=6, min_periods=1).sum()[::-1].shift(-6).fillna(0)
        ).clip(0, 36)
        
        # Target 5: boundary_flag
        df['boundary_flag'] = (df['runs_batter'] >= 4).astype(np.int8)
        
        # Remove rows without future data
        initial_rows = len(df)
        df = df.dropna(subset=['runs_next_over', 'runs_batsman_next_over'])
        final_rows = len(df)
        
        df = downcast_dataframe(df)
        
        logger.info(f"   Targets created. Rows: {final_rows:,} (removed {initial_rows - final_rows:,})")
        logger.info(f"   runs_class distribution:\n{df['runs_class'].value_counts(normalize=True).sort_index()}")
        logger.info(f"   Wicket rate: {df['wicket_flag'].mean():.3%}")
        logger.info(f"   Boundary rate: {df['boundary_flag'].mean():.3%}")
        
        return df


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-series aware split"""
    logger.info("\n✂️  TIME-SERIES SPLIT...")
    
    df = df.sort_values('date')
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    logger.info("="*80)
    logger.info(f"Train: {len(train):,} rows | {train['date'].min()} to {train['date'].max()}")
    logger.info(f"Val:   {len(val):,} rows | {val['date'].min()} to {val['date'].max()}")
    logger.info(f"Test:  {len(test):,} rows | {test['date'].min()} to {test['date'].max()}")
    logger.info("="*80)
    
    return train, val, test


def save_to_database(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, 
                     db_path: str, table_prefix: str = "ml_features_unified"):
    """Save all datasets to database using temp tables"""
    logger.info("\n💾 SAVING TO DATABASE...")
    
    conn = duckdb.connect(db_path)
    
    # Save full feature set
    logger.info(f"   Saving {table_prefix}...")
    conn.execute(f"DROP TABLE IF EXISTS {table_prefix}")
    conn.execute(f"CREATE TABLE {table_prefix} AS SELECT * FROM df")
    
    # Save splits
    logger.info(f"   Saving {table_prefix}_train...")
    conn.execute(f"DROP TABLE IF EXISTS {table_prefix}_train")
    conn.execute(f"CREATE TABLE {table_prefix}_train AS SELECT * FROM train")
    
    logger.info(f"   Saving {table_prefix}_val...")
    conn.execute(f"DROP TABLE IF EXISTS {table_prefix}_val")
    conn.execute(f"CREATE TABLE {table_prefix}_val AS SELECT * FROM val")
    
    logger.info(f"   Saving {table_prefix}_test...")
    conn.execute(f"DROP TABLE IF EXISTS {table_prefix}_test")
    conn.execute(f"CREATE TABLE {table_prefix}_test AS SELECT * FROM test")
    
    conn.close()
    
    logger.info(f"   ✓ All datasets saved with prefix '{table_prefix}'")


def print_summary(df_final: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: UnifiedMLConfig):
    """Print comprehensive summary"""
    print("\n" + "="*80)
    print("🏆 UNIFIED FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Database: {config.db_path}")
    print(f"Total features: {len(df_final.columns)}")
    print(f"Total samples: {len(df_final):,}")
    print(f"Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📊 Feature categories:")
    feature_categories = {
        'Base features': len([c for c in df_final.columns if any(x in c for x in ['phase', 'balls_', 'wickets_', 'runs_so_far'])]),
        'Temporal': len([c for c in df_final.columns if 'season' in c or 'era' in c]),
        'Rolling stats': len([c for c in df_final.columns if any(x in c for x in ['_short', '_medium', '_long']) and 'ema' not in c]),
        'EMA features': len([c for c in df_final.columns if 'ema' in c]),
        'Venue': len([c for c in df_final.columns if 'venue' in c]),
        'Matchup': len([c for c in df_final.columns if 'h2h' in c]),
        'Career phase': len([c for c in df_final.columns if 'career' in c or 'maturity' in c]),
        'Form-fatigue': len([c for c in df_final.columns if 'days_since' in c or 'workload' in c or 'form_score' in c]),
        'Momentum': len([c for c in df_final.columns if 'momentum' in c or 'pressure' in c or 'rr_' in c or 'partnership' in c]),
    }
    
    for category, count in feature_categories.items():
        print(f"  • {category}: {count}")
    
    print(f"\n🎯 Targets created:")
    print(f"  • runs_class: Multi-class (0,1,2,3,4,6,wicket,extra)")
    print(f"  • wicket_flag: Binary wicket prediction")
    print(f"  • runs_next_over: Regression (0-36)")
    print(f"  • runs_batsman_next_over: Batsman-specific (0-36)")
    print(f"  • boundary_flag: Binary boundary prediction")
    
    print(f"\n📈 Data splits:")
    print(f"  • Train: {len(train):,} ({len(train)/len(df_final)*100:.1f}%)")
    print(f"  • Val:   {len(val):,} ({len(val)/len(df_final)*100:.1f}%)")
    print(f"  • Test:  {len(test):,} ({len(test)/len(df_final)*100:.1f}%)")
    
    print(f"\n⚡ Performance:")
    print(f"  • Chunk size: {config.chunk_size} players")
    print(f"  • Memory optimized: Downcasted numerics")
    print(f"  • DuckDB accelerated: Base + venue + matchup features")
    
    print(f"\n📊 Season statistics:")
    print(f"  • Season weight range: {df_final['season_weight'].min():.4f} - {df_final['season_weight'].max():.4f}")
    print(f"  • Modern cricket: {df_final['is_modern_cricket'].mean()*100:.1f}%")
    
    print("="*80)
    print("✅ Ready for model training!")
    print("="*80)


# Main execution
if __name__ == "__main__":
    # Configuration
    config = UnifiedMLConfig(
        db_path="ipl_cricket.db",
        chunk_size=50  # Adjust based on your memory
    )
    
    # Create features
    engineer = UnifiedFeatureEngineer(config)
    df_final = engineer.create_all_features()
    
    # Create targets
    df_final = TargetCreator.create_targets(df_final)
    
    # Split data
    train, val, test = time_series_split(df_final, train_ratio=0.7, val_ratio=0.15)
    
    # Save to database
    save_to_database(df_final, train, val, test, config.db_path)
    
    # Print comprehensive summary
    print_summary(df_final, train, val, test, config)
    
    logger.info("\n🎉 Pipeline complete! Check the logs above for detailed stats.")