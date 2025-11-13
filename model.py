# """
# Advanced Cricket ML Pipeline - Complete Pipeline
# ================================================
# This module runs the complete pipeline:
# 1. Train all ML models
# 2. Run Monte Carlo simulations
# 3. Generate predictions
# """

# import pandas as pd
# import numpy as np
# import duckdb
# import lightgbm as lgb
# from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error, mean_squared_error
# from sklearn.isotonic import IsotonicRegression
# from typing import Dict, List, Tuple, Optional
# from dataclasses import dataclass, field
# import pickle
# import json
# import logging
# import time
# import os

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class ModelConfig:
#     """Configuration for models"""
#     # Feature columns (will be populated)
#     feature_cols: List[str] = None
    
#     # Categorical features
#     categorical_features: List[str] = None
    
#     # LightGBM params
#     lgb_params_classification: Dict = None
#     lgb_params_regression: Dict = None
    
#     # Training params
#     early_stopping_rounds: int = 50
#     n_estimators: int = 1000
    
#     def __post_init__(self):
#         if self.categorical_features is None:
#             self.categorical_features = [
#                 'venue', 'city', 'phase', 'batter', 'bowler', 
#                 'non_striker', 'toss_winner', 'toss_decision'
#             ]
        
#         if self.lgb_params_classification is None:
#             self.lgb_params_classification = {
#                 'objective': 'multiclass',
#                 'num_class': 9,  # 0-6 runs, wicket, extra
#                 'metric': 'multi_logloss',
#                 'learning_rate': 0.05,
#                 'num_leaves': 64,
#                 'max_depth': 8,
#                 'min_data_in_leaf': 50,
#                 'feature_fraction': 0.8,
#                 'bagging_fraction': 0.8,
#                 'bagging_freq': 5,
#                 'lambda_l1': 0.5,
#                 'lambda_l2': 0.5,
#                 'verbose': -1,
#                 'device': 'cpu'
#             }
        
#         if self.lgb_params_regression is None:
#             self.lgb_params_regression = {
#                 'objective': 'regression',
#                 'metric': 'mae',
#                 'learning_rate': 0.05,
#                 'num_leaves': 64,
#                 'max_depth': 8,
#                 'min_data_in_leaf': 50,
#                 'feature_fraction': 0.8,
#                 'bagging_fraction': 0.8,
#                 'bagging_freq': 5,
#                 'lambda_l1': 0.5,
#                 'lambda_l2': 0.5,
#                 'verbose': -1
#             }


# class FeatureSelector:
#     """Select relevant features for modeling"""
    
#     @staticmethod
#     def get_base_features() -> List[str]:
#         """Core features for all models"""
#         return [
#             # Match state
#             'over', 'ball_in_over', 'inning', 'balls_elapsed', 'balls_remaining',
#             'runs_so_far', 'wickets_so_far', 'wickets_remaining',
#             'current_run_rate', 'required_run_rate', 'run_rate_pressure',
            
#             # Phase
#             'is_powerplay', 'is_middle_over', 'is_death_over', 'late_in_over',
            
#             # Rolling stats - batsman
#             'batsman_rpb_short', 'batsman_rpb_medium', 'batsman_rpb_long',
#             'batsman_sr_short', 'batsman_sr_medium', 'batsman_sr_long',
#             'batsman_boundary_pct_short', 'batsman_boundary_pct_medium',
#             'batsman_dot_pct_short', 'batsman_dot_pct_medium',
            
#             # Rolling stats - bowler
#             'bowler_econ_short', 'bowler_econ_medium', 'bowler_econ_long',
#             'bowler_wicket_rate_short', 'bowler_wicket_rate_medium',
#             'bowler_dot_pct_short', 'bowler_dot_pct_medium',
            
#             # Venue
#             'venue_avg_first_innings', 'venue_avg_rpb', 
#             'venue_boundary_rate', 'venue_wicket_rate',
            
#             # Matchup
#             'h2h_balls', 'h2h_rpb', 'h2h_dismissal_rate', 'h2h_strike_rate',
            
#             # Momentum
#             'runs_last_3_overs', 'wickets_last_3_overs', 
#             'rr_last_3_overs', 'rr_acceleration',
#             'batsman_balls_faced_this_inning', 'bowler_balls_bowled_this_inning',
#             'partnership_runs'
#         ]
    
#     @staticmethod
#     def get_categorical_features() -> List[str]:
#         """Categorical features for encoding"""
#         return ['venue', 'city', 'phase', 'batter', 'bowler', 'non_striker']


# class MultiTaskModel:
#     """
#     Multi-task model manager for cricket predictions
#     Trains separate models for each task
#     """
    
#     def __init__(self, config: ModelConfig):
#         self.config = config
#         self.models = {}
#         self.calibrators = {}
#         self.feature_importance = {}
        
#     def prepare_data(self, df: pd.DataFrame, target: str, is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
#         """Prepare features and target for training or prediction"""
        
#         # Select features
#         feature_cols = FeatureSelector.get_base_features()
        
#         # Filter columns that exist in dataframe
#         feature_cols = [col for col in feature_cols if col in df.columns]
        
#         X = df[feature_cols].copy()
        
#         # Handle categorical features
#         categorical_cols = [col for col in FeatureSelector.get_categorical_features() if col in X.columns]
        
#         for col in categorical_cols:
#             X[col] = X[col].astype('category')
        
#         # Only extract target if training
#         if is_training and target in df.columns:
#             y = df[target].copy()
#             return X, y
#         else:
#             return X, None
    
#     def train_runs_classifier(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
#         """Train multi-class classifier for runs_class"""
        
#         logger.info("Training runs_class multi-class classifier...")
        
#         X_train, y_train = self.prepare_data(train_df, 'runs_class')
#         X_val, y_val = self.prepare_data(val_df, 'runs_class')
        
#         # Handle class imbalance with weights
#         class_weights = len(y_train) / (9 * np.bincount(y_train.astype(int)))
#         sample_weights = np.array([class_weights[int(y)] for y in y_train])
        
#         # LightGBM
#         train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights, categorical_feature='auto')
#         val_data = lgb.Dataset(X_val, label=y_val, categorical_feature='auto', reference=train_data)
        
#         model = lgb.train(
#             self.config.lgb_params_classification,
#             train_data,
#             valid_sets=[train_data, val_data],
#             num_boost_round=self.config.n_estimators,
#             callbacks=[
#                 lgb.early_stopping(self.config.early_stopping_rounds),
#                 lgb.log_evaluation(50)
#             ]
#         )
        
#         # Evaluate
#         y_pred_proba = model.predict(X_val)
#         y_pred_class = np.argmax(y_pred_proba, axis=1)
        
#         logloss = log_loss(y_val, y_pred_proba)
        
#         # Per-class Brier score
#         brier_scores = {}
#         for cls in range(9):
#             y_binary = (y_val == cls).astype(int)
#             brier_scores[f"class_{cls}"] = brier_score_loss(y_binary, y_pred_proba[:, cls])
        
#         logger.info(f"Runs Classifier - Log Loss: {logloss:.4f}")
#         logger.info(f"Average Brier Score: {np.mean(list(brier_scores.values())):.4f}")
        
#         self.models['runs_class'] = model
#         self.feature_importance['runs_class'] = model.feature_importance(importance_type='gain')
        
#         return {
#             'log_loss': logloss,
#             'brier_scores': brier_scores,
#             'feature_importance': dict(zip(X_train.columns, model.feature_importance()))
#         }
    
#     def train_wicket_classifier(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
#         """Train binary classifier for wicket_flag with calibration"""
        
#         logger.info("Training wicket_flag binary classifier...")
        
#         X_train, y_train = self.prepare_data(train_df, 'wicket_flag')
#         X_val, y_val = self.prepare_data(val_df, 'wicket_flag')
        
#         # Handle severe class imbalance - use scale_pos_weight
#         pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
#         # Create binary classification params (don't copy from multiclass)
#         params = {
#             'objective': 'binary',
#             'metric': 'binary_logloss',
#             'learning_rate': 0.05,
#             'num_leaves': 64,
#             'max_depth': 8,
#             'min_data_in_leaf': 50,
#             'feature_fraction': 0.8,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 5,
#             'lambda_l1': 0.5,
#             'lambda_l2': 0.5,
#             'scale_pos_weight': pos_weight,
#             'verbose': -1
#         }
        
#         train_data = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')
#         val_data = lgb.Dataset(X_val, label=y_val, categorical_feature='auto', reference=train_data)
        
#         model = lgb.train(
#             params,
#             train_data,
#             valid_sets=[train_data, val_data],
#             num_boost_round=self.config.n_estimators,
#             callbacks=[
#                 lgb.early_stopping(self.config.early_stopping_rounds),
#                 lgb.log_evaluation(50)
#             ]
#         )
        
#         # Predictions
#         y_pred_proba = model.predict(X_val)
        
#         logloss = log_loss(y_val, y_pred_proba)
#         brier = brier_score_loss(y_val, y_pred_proba)
        
#         # Calibrate using Isotonic Regression
#         logger.info("Calibrating wicket probabilities...")
#         calibrator = IsotonicRegression(out_of_bounds='clip')
#         calibrator.fit(y_pred_proba, y_val)
#         y_pred_calibrated = calibrator.predict(y_pred_proba)
        
#         brier_calibrated = brier_score_loss(y_val, y_pred_calibrated)
        
#         logger.info(f"Wicket Classifier - Log Loss: {logloss:.4f}, Brier: {brier:.4f}")
#         logger.info(f"After calibration - Brier: {brier_calibrated:.4f}")
        
#         self.models['wicket_flag'] = model
#         self.calibrators['wicket_flag'] = calibrator
#         self.feature_importance['wicket_flag'] = model.feature_importance(importance_type='gain')
        
#         return {
#             'log_loss': logloss,
#             'brier_before': brier,
#             'brier_after': brier_calibrated,
#             'wicket_rate': y_val.mean(),
#             'predicted_rate': y_pred_calibrated.mean()
#         }
    
#     def train_runs_regressor(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
#                             target: str, quantiles: List[float] = [0.05, 0.5, 0.95]) -> Dict:
#         """Train quantile regression for runs_next_over or runs_batsman_next_over"""
        
#         logger.info(f"Training quantile regressor for {target}...")
        
#         X_train, y_train = self.prepare_data(train_df, target)
#         X_val, y_val = self.prepare_data(val_df, target)
        
#         models = {}
#         predictions = {}
        
#         for q in quantiles:
#             logger.info(f"  Training quantile {q}...")
            
#             params = self.config.lgb_params_regression.copy()
#             params.update({
#                 'objective': 'quantile',
#                 'alpha': q,
#                 'metric': 'quantile'
#             })
            
#             train_data = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')
#             val_data = lgb.Dataset(X_val, label=y_val, categorical_feature='auto', reference=train_data)
            
#             model = lgb.train(
#                 params,
#                 train_data,
#                 valid_sets=[val_data],
#                 num_boost_round=self.config.n_estimators,
#                 callbacks=[
#                     lgb.early_stopping(self.config.early_stopping_rounds),
#                     lgb.log_evaluation(100)
#                 ]
#             )
            
#             models[f'q{int(q*100)}'] = model
#             predictions[f'q{int(q*100)}'] = model.predict(X_val)
        
#         # Evaluate median (q50)
#         y_pred_median = predictions['q50']
#         mae = mean_absolute_error(y_val, y_pred_median)
#         rmse = np.sqrt(mean_squared_error(y_val, y_pred_median))
        
#         # Coverage check for 90% interval
#         coverage_90 = np.mean((y_val >= predictions['q5']) & (y_val <= predictions['q95']))
        
#         logger.info(f"{target} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
#         logger.info(f"90% interval coverage: {coverage_90:.2%} (target: 90%)")
        
#         self.models[target] = models
        
#         return {
#             'mae': mae,
#             'rmse': rmse,
#             'coverage_90': coverage_90,
#             'quantile_predictions': predictions
#         }
    
#     def predict(self, df: pd.DataFrame, task: str, return_uncertainty: bool = True) -> Dict:
#         """
#         Make predictions with uncertainty estimates
        
#         Returns:
#             Dictionary with predictions and uncertainty intervals
#         """
        
#         if task == 'runs_class':
#             X, _ = self.prepare_data(df, 'runs_class', is_training=False)
#             proba = self.models['runs_class'].predict(X)
            
#             return {
#                 'probabilities': proba,
#                 'predicted_class': np.argmax(proba, axis=1),
#                 'confidence': np.max(proba, axis=1),
#                 'entropy': -np.sum(proba * np.log(proba + 1e-10), axis=1)  # Uncertainty
#             }
        
#         elif task == 'wicket_flag':
#             X, _ = self.prepare_data(df, 'wicket_flag', is_training=False)
#             proba = self.models['wicket_flag'].predict(X)
            
#             # Apply calibration
#             if 'wicket_flag' in self.calibrators:
#                 proba = self.calibrators['wicket_flag'].predict(proba)
            
#             return {
#                 'probability': proba,
#                 'prediction': (proba > 0.5).astype(int)
#             }
        
#         elif task in ['runs_next_over', 'runs_batsman_next_over']:
#             X, _ = self.prepare_data(df, task, is_training=False)
            
#             quantile_preds = {}
#             for q_name, model in self.models[task].items():
#                 quantile_preds[q_name] = model.predict(X)
            
#             return {
#                 'median': quantile_preds['q50'],
#                 'lower_90': quantile_preds['q5'],
#                 'upper_90': quantile_preds['q95'],
#                 'lower_50': quantile_preds.get('q25', quantile_preds['q50']),
#                 'upper_50': quantile_preds.get('q75', quantile_preds['q50']),
#                 'uncertainty': quantile_preds['q95'] - quantile_preds['q5']  # Width of interval
#             }
        
#         else:
#             raise ValueError(f"Unknown task: {task}")
    
#     def save_models(self, path: str = "models/"):
#         """Save all models and calibrators"""
#         os.makedirs(path, exist_ok=True)
        
#         # Save LightGBM models
#         for name, model in self.models.items():
#             if isinstance(model, dict):  # Quantile models
#                 for q_name, q_model in model.items():
#                     q_model.save_model(f"{path}/{name}_{q_name}.txt")
#             else:
#                 model.save_model(f"{path}/{name}.txt")
        
#         # Save calibrators
#         with open(f"{path}/calibrators.pkl", 'wb') as f:
#             pickle.dump(self.calibrators, f)
        
#         # Save feature importance
#         with open(f"{path}/feature_importance.json", 'w') as f:
#             json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v 
#                       for k, v in self.feature_importance.items()}, f, indent=2)
        
#         logger.info(f"Models saved to {path}")
    
#     def load_models(self, path: str = "models/"):
#         """Load all trained models and calibrators"""
#         import os
#         import pickle
#         import json
        
#         # Load LightGBM models
#         model_files = os.listdir(path)
        
#         # Load runs_class model
#         if 'runs_class.txt' in model_files:
#             self.models['runs_class'] = lgb.Booster(model_file=f"{path}/runs_class.txt")
#             logger.info("Loaded runs_class model")
        
#         # Load wicket_flag model
#         if 'wicket_flag.txt' in model_files:
#             self.models['wicket_flag'] = lgb.Booster(model_file=f"{path}/wicket_flag.txt")
#             logger.info("Loaded wicket_flag model")
        
#         # Load quantile models for runs_next_over
#         runs_over_models = {}
#         for q_file in [f for f in model_files if f.startswith('runs_next_over') and f.endswith('.txt')]:
#             q_name = q_file.replace('runs_next_over_', '').replace('.txt', '')
#             runs_over_models[q_name] = lgb.Booster(model_file=f"{path}/{q_file}")
#         if runs_over_models:
#             self.models['runs_next_over'] = runs_over_models
#             logger.info("Loaded runs_next_over quantile models")
        
#         # Load quantile models for runs_batsman_next_over
#         batsman_runs_models = {}
#         for q_file in [f for f in model_files if f.startswith('runs_batsman_next_over') and f.endswith('.txt')]:
#             q_name = q_file.replace('runs_batsman_next_over_', '').replace('.txt', '')
#             batsman_runs_models[q_name] = lgb.Booster(model_file=f"{path}/{q_file}")
#         if batsman_runs_models:
#             self.models['runs_batsman_next_over'] = batsman_runs_models
#             logger.info("Loaded runs_batsman_next_over quantile models")
        
#         # Load calibrators
#         if os.path.exists(f"{path}/calibrators.pkl"):
#             with open(f"{path}/calibrators.pkl", 'rb') as f:
#                 self.calibrators = pickle.load(f)
#             logger.info("Loaded calibrators")
        
#         # Load feature importance
#         if os.path.exists(f"{path}/feature_importance.json"):
#             with open(f"{path}/feature_importance.json", 'r') as f:
#                 self.feature_importance = json.load(f)
#             logger.info("Loaded feature importance")
        
#         logger.info(f"Models loaded from {path}")


# def train_all_models(train_df: pd.DataFrame, val_df: pd.DataFrame, 
#                      config: ModelConfig = None) -> MultiTaskModel:
#     """Train all multi-task models"""
    
#     if config is None:
#         config = ModelConfig()
    
#     mtm = MultiTaskModel(config)
    
#     # Train each task
#     results = {}
    
#     # 1. Runs classifier
#     results['runs_class'] = mtm.train_runs_classifier(train_df, val_df)
    
#     # 2. Wicket classifier
#     results['wicket_flag'] = mtm.train_wicket_classifier(train_df, val_df)
    
#     # 3. Runs next over regressor
#     results['runs_next_over'] = mtm.train_runs_regressor(
#         train_df, val_df, 'runs_next_over', quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
#     )
    
#     # 4. Batsman runs next over regressor
#     results['runs_batsman_next_over'] = mtm.train_runs_regressor(
#         train_df, val_df, 'runs_batsman_next_over', quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
#     )
    
#     logger.info("\n" + "="*60)
#     logger.info("Training Summary")
#     logger.info("="*60)
#     for task, metrics in results.items():
#         logger.info(f"\n{task}:")
#         for metric, value in metrics.items():
#             if not isinstance(value, dict):
#                 logger.info(f"  {metric}: {value}")
    
#     return mtm, results


# @dataclass
# class MatchState:
#     """Current state of a cricket match"""
#     # Match context
#     match_id: str
#     venue: str
#     city: str
#     inning: int
#     batting_team: str
#     bowling_team: str
    
#     # Current state
#     over: int
#     ball_in_over: int
#     runs_so_far: int
#     wickets_so_far: int
#     balls_remaining: int
#     wickets_remaining: int
    
#     # Players
#     batter: str
#     non_striker: str
#     bowler: str
    
#     # Context features (for model prediction)
#     current_run_rate: float = 0.0
#     required_run_rate: float = 0.0
#     run_rate_pressure: float = 0.0
#     target: Optional[int] = None
    
#     # Phase
#     phase: str = "Powerplay"
#     is_powerplay: int = 0
#     is_middle_over: int = 0
#     is_death_over: int = 0
    
#     # Historical features (would come from feature engineering)
#     features: Dict = field(default_factory=dict)
    
#     def to_dataframe(self) -> pd.DataFrame:
#         """Convert state to DataFrame for model prediction"""
#         data = {
#             'over': self.over,
#             'ball_in_over': self.ball_in_over,
#             'inning': self.inning,
#             'runs_so_far': self.runs_so_far,
#             'wickets_so_far': self.wickets_so_far,
#             'balls_remaining': self.balls_remaining,
#             'wickets_remaining': self.wickets_remaining,
#             'current_run_rate': self.current_run_rate,
#             'required_run_rate': self.required_run_rate,
#             'run_rate_pressure': self.run_rate_pressure,
#             'is_powerplay': self.is_powerplay,
#             'is_middle_over': self.is_middle_over,
#             'is_death_over': self.is_death_over,
#             'late_in_over': 1 if self.ball_in_over >= 4 else 0,
#             'venue': self.venue,
#             'city': self.city,
#             'batter': self.batter,
#             'bowler': self.bowler,
#             'non_striker': self.non_striker,
#             'phase': self.phase,
#             'balls_elapsed': 120 - self.balls_remaining,  # For T20
#             **self.features  # Additional features
#         }
        
#         # Add default values for features that might be missing
#         default_features = {
#             'batsman_rpb_short': 0.5,
#             'batsman_rpb_medium': 0.5,
#             'batsman_rpb_long': 0.5,
#             'batsman_sr_short': 100.0,
#             'batsman_sr_medium': 100.0,
#             'batsman_sr_long': 100.0,
#             'batsman_boundary_pct_short': 10.0,
#             'batsman_boundary_pct_medium': 10.0,
#             'batsman_dot_pct_short': 30.0,
#             'batsman_dot_pct_medium': 30.0,
#             'bowler_econ_short': 7.0,
#             'bowler_econ_medium': 7.0,
#             'bowler_econ_long': 7.0,
#             'bowler_wicket_rate_short': 2.0,
#             'bowler_wicket_rate_medium': 2.0,
#             'bowler_dot_pct_short': 40.0,
#             'bowler_dot_pct_medium': 40.0,
#             'venue_avg_first_innings': 160.0,
#             'venue_avg_rpb': 0.6,
#             'venue_boundary_rate': 15.0,
#             'venue_wicket_rate': 0.05,
#             'h2h_balls': 0,
#             'h2h_rpb': 0.5,
#             'h2h_dismissal_rate': 0.02,
#             'h2h_strike_rate': 100.0,
#             'runs_last_3_overs': 0,
#             'wickets_last_3_overs': 0,
#             'rr_last_3_overs': 6.0,
#             'rr_acceleration': 0.0,
#             'batsman_balls_faced_this_inning': 0,
#             'bowler_balls_bowled_this_inning': 0,
#             'partnership_runs': 0
#         }
        
#         # Add missing features with defaults
#         for key, default_value in default_features.items():
#             if key not in data:
#                 data[key] = default_value
        
#         return pd.DataFrame([data])
    
#     def update_after_ball(self, runs: int, wicket: bool):
#         """Update state after a ball is bowled"""
#         self.runs_so_far += runs
        
#         if wicket:
#             self.wickets_so_far += 1
#             self.wickets_remaining -= 1
        
#         # Update ball count
#         self.ball_in_over += 1
#         if self.ball_in_over >= 6:
#             self.over += 1
#             self.ball_in_over = 0
        
#         self.balls_remaining -= 1
        
#         # Update run rates
#         balls_elapsed = 120 - self.balls_remaining  # For T20
#         self.current_run_rate = (self.runs_so_far / balls_elapsed * 6) if balls_elapsed > 0 else 0
        
#         if self.inning == 2 and self.target:
#             runs_needed = self.target - self.runs_so_far
#             self.required_run_rate = (runs_needed / self.balls_remaining * 6) if self.balls_remaining > 0 else 0
#             self.run_rate_pressure = self.required_run_rate - self.current_run_rate
        
#         # Update phase
#         if self.over < 6:
#             self.phase = "Powerplay"
#             self.is_powerplay = 1
#             self.is_middle_over = 0
#             self.is_death_over = 0
#         elif self.over >= 16:
#             self.phase = "Death"
#             self.is_powerplay = 0
#             self.is_middle_over = 0
#             self.is_death_over = 1
#         else:
#             self.phase = "Middle"
#             self.is_powerplay = 0
#             self.is_middle_over = 1
#             self.is_death_over = 0
    
#     def is_innings_over(self) -> bool:
#         """Check if innings is complete"""
#         return self.wickets_so_far >= 10 or self.balls_remaining <= 0 or (
#             self.inning == 2 and self.target and self.runs_so_far >= self.target
#         )


# class MonteCarloSimulator:
#     """
#     Monte Carlo simulation engine for cricket match prediction
#     """
    
#     def __init__(self, model, n_simulations: int = 5000, random_seed: int = 42):
#         """
#         Args:
#             model: MultiTaskModel with trained predictors
#             n_simulations: Number of simulation runs
#             random_seed: For reproducibility
#         """
#         self.model = model
#         self.n_simulations = n_simulations
#         self.random_seed = random_seed
#         np.random.seed(random_seed)
        
#     def simulate_ball(self, state: MatchState) -> Tuple[int, bool]:
#         """
#         Simulate a single ball outcome
        
#         Returns:
#             (runs_scored, wicket_flag)
#         """
#         # Get state as DataFrame
#         state_df = state.to_dataframe()
        
#         # Predict runs distribution
#         runs_pred = self.model.predict(state_df, 'runs_class')
#         runs_proba = runs_pred['probabilities'][0]
        
#         # Predict wicket probability
#         wicket_pred = self.model.predict(state_df, 'wicket_flag')
#         wicket_proba = wicket_pred['probability'][0]
        
#         # Sample runs from categorical distribution
#         # Classes: 0,1,2,3,4,5,6,wicket,extra
#         runs_class = np.random.choice(9, p=runs_proba)
        
#         if runs_class == 7:  # Wicket class
#             # Sample runs (could be 0 or few)
#             runs = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
#             wicket = True
#         elif runs_class == 8:  # Extra
#             runs = np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.07, 0.03])
#             wicket = False
#         else:
#             runs = min(runs_class, 6)
#             # Check wicket separately
#             wicket = np.random.random() < wicket_proba
        
#         return runs, wicket
    
#     def simulate_innings(self, initial_state: MatchState) -> Dict:
#         """
#         Simulate a complete innings from current state
        
#         Returns:
#             Dictionary with final score and ball-by-ball details
#         """
#         state = MatchState(**initial_state.__dict__)  # Copy state
        
#         balls_simulated = []
        
#         while not state.is_innings_over():
#             runs, wicket = self.simulate_ball(state)
            
#             balls_simulated.append({
#                 'over': state.over,
#                 'ball': state.ball_in_over,
#                 'runs': runs,
#                 'wicket': wicket,
#                 'score': state.runs_so_far + runs,
#                 'wickets': state.wickets_so_far + (1 if wicket else 0)
#             })
            
#             state.update_after_ball(runs, wicket)
        
#         return {
#             'final_score': state.runs_so_far,
#             'wickets_lost': state.wickets_so_far,
#             'balls_used': 120 - state.balls_remaining,
#             'balls_simulated': balls_simulated
#         }
    
#     def run_simulations(self, initial_state: MatchState, 
#                        n_sims: Optional[int] = None) -> Dict:
#         """
#         Run multiple simulations and aggregate results
        
#         Returns:
#             Dictionary with score distribution, win probability, etc.
#         """
#         if n_sims is None:
#             n_sims = self.n_simulations
        
#         logger.info(f"Running {n_sims} simulations...")
#         start_time = time.time()
        
#         final_scores = []
#         wickets_lost = []
        
#         for i in range(n_sims):
#             if (i + 1) % 1000 == 0:
#                 logger.info(f"  Completed {i+1}/{n_sims} simulations")
            
#             result = self.simulate_innings(initial_state)
#             final_scores.append(result['final_score'])
#             wickets_lost.append(result['wickets_lost'])
        
#         final_scores = np.array(final_scores)
#         wickets_lost = np.array(wickets_lost)
        
#         elapsed = time.time() - start_time
#         logger.info(f"Simulations complete in {elapsed:.2f}s ({n_sims/elapsed:.0f} sims/sec)")
        
#         # Calculate statistics
#         results = {
#             'n_simulations': n_sims,
#             'score_distribution': {
#                 'mean': float(np.mean(final_scores)),
#                 'median': float(np.median(final_scores)),
#                 'std': float(np.std(final_scores)),
#                 'min': int(np.min(final_scores)),
#                 'max': int(np.max(final_scores)),
#                 'q05': float(np.percentile(final_scores, 5)),
#                 'q25': float(np.percentile(final_scores, 25)),
#                 'q75': float(np.percentile(final_scores, 75)),
#                 'q95': float(np.percentile(final_scores, 95))
#             },
#             'wickets_distribution': {
#                 'mean': float(np.mean(wickets_lost)),
#                 'median': float(np.median(wickets_lost))
#             },
#             'raw_scores': final_scores.tolist()  # For histograms
#         }
        
#         # Win probability (for 2nd innings)
#         if initial_state.inning == 2 and initial_state.target:
#             wins = np.sum(final_scores >= initial_state.target)
#             results['win_probability'] = float(wins / n_sims)
#             results['target'] = initial_state.target
            
#             # Probability of winning by over
#             results['win_by_over_probability'] = float(
#                 np.sum((final_scores >= initial_state.target) & (wickets_lost < 10)) / n_sims
#             )
        
#         return results
    
#     def predict_next_N_balls(self, state: MatchState, N: int = 6) -> Dict:
#         """
#         Predict outcomes for next N balls with probabilities
        
#         Useful for real-time UI updates
#         """
#         ball_predictions = []
        
#         for ball_num in range(N):
#             state_df = state.to_dataframe()
            
#             # Get predictions
#             runs_pred = self.model.predict(state_df, 'runs_class')
#             wicket_pred = self.model.predict(state_df, 'wicket_flag')
            
#             ball_predictions.append({
#                 'ball_number': ball_num + 1,
#                 'over': state.over,
#                 'ball_in_over': state.ball_in_over,
#                 'runs_probabilities': {
#                     str(i): float(runs_pred['probabilities'][0][i]) 
#                     for i in range(9)
#                 },
#                 'wicket_probability': float(wicket_pred['probability'][0]),
#                 'expected_runs': float(np.sum(
#                     np.arange(9) * runs_pred['probabilities'][0]
#                 ))
#             })
            
#             # Update state for next ball (use expected values)
#             expected_runs = int(np.argmax(runs_pred['probabilities'][0]))
#             expected_wicket = wicket_pred['probability'][0] > 0.5
#             state.update_after_ball(expected_runs, expected_wicket)
            
#             if state.is_innings_over():
#                 break
        
#         return {
#             'predictions': ball_predictions,
#             'over_expected_runs': sum(p['expected_runs'] for p in ball_predictions)
#         }


# def run_complete_pipeline(db_path: str = "ipl_cricket.db", n_simulations: int = 1000):
#     """
#     Run the complete pipeline: train models and run Monte Carlo simulations
#     """
#     logger.info("Starting complete cricket ML pipeline...")
    
#     # Load data
#     logger.info("Loading data...")
#     conn = duckdb.connect(db_path)
#     train_df = conn.execute("SELECT * FROM train_set").fetchdf()
#     val_df = conn.execute("SELECT * FROM val_set").fetchdf()
#     test_df = conn.execute("SELECT * FROM test_set").fetchdf()
#     conn.close()
    
#     # Train models
#     logger.info("Training models...")
#     mtm, results = train_all_models(train_df, val_df)
    
#     # Save models
#     mtm.save_models("models/")
    
#     # Example prediction from test data
#     sample = test_df.head(100)
    
#     runs_pred = mtm.predict(sample, 'runs_class')
#     logger.info("Runs Class Predictions (first 5):")
#     logger.info(f"Predicted: {runs_pred['predicted_class'][:5]}")
#     logger.info(f"Confidence: {runs_pred['confidence'][:5]}")
    
#     wicket_pred = mtm.predict(sample, 'wicket_flag')
#     logger.info("Wicket Probabilities (first 5):")
#     logger.info(wicket_pred['probability'][:5])
    
#     runs_over_pred = mtm.predict(sample, 'runs_next_over')
#     logger.info("Runs Next Over (first 5):")
#     logger.info(f"Median: {runs_over_pred['median'][:5]}")
#     logger.info(f"90% Interval: {list(zip(runs_over_pred['lower_90'][:5], runs_over_pred['upper_90'][:5]))}")
    
#     # Run Monte Carlo simulation
#     logger.info("Setting up Monte Carlo simulation...")
    
#     # Create initial match state
#     state = MatchState(
#         match_id="match_001",
#         venue="Eden Gardens",
#         city="Kolkata",
#         inning=2,
#         batting_team="Team A",
#         bowling_team="Team B",
#         over=10,
#         ball_in_over=3,
#         runs_so_far=85,
#         wickets_so_far=3,
#         balls_remaining=57,
#         wickets_remaining=7,
#         batter="Player A",
#         non_striker="Player B",
#         bowler="Bowler X",
#         target=165,
#         features={
#             'batsman_sr_medium': 125.0,
#             'bowler_econ_medium': 8.5,
#             'venue_avg_first_innings': 160.0,
#             # ... other features
#         }
#     )
    
#     # Create simulator
#     simulator = MonteCarloSimulator(mtm, n_simulations=n_simulations)
    
#     # Run simulations
#     logger.info("Running Monte Carlo simulations...")
#     results = simulator.run_simulations(state)
    
#     logger.info("\n" + "="*60)
#     logger.info("MONTE CARLO SIMULATION RESULTS")
#     logger.info("="*60)
#     logger.info(f"Expected Score: {results['score_distribution']['median']:.0f}")
#     logger.info(f"90% Range: {results['score_distribution']['q05']:.0f} - {results['score_distribution']['q95']:.0f}")
#     if 'win_probability' in results:
#         logger.info(f"Win Probability: {results['win_probability']:.1%}")
    
#     # Next balls prediction
#     next_balls = simulator.predict_next_N_balls(state, N=6)
#     logger.info("\n" + "="*60)
#     logger.info("NEXT OVER PREDICTION")
#     logger.info("="*60)
#     logger.info(f"Expected Runs: {next_balls['over_expected_runs']:.1f}")
#     for ball in next_balls['predictions'][:3]:
#         logger.info(f"Ball {ball['ball_number']}: {ball['expected_runs']:.2f} runs "
#               f"(Wicket: {ball['wicket_probability']:.1%})")
    
#     logger.info("\nPipeline completed successfully!")
    
#     return mtm, results


# if __name__ == "__main__":
#     # Run the complete pipeline
#     model, sim_results = run_complete_pipeline(n_simulations=1000)  # Reduced for faster execution


"""
Enhanced Cricket ML Model Training
===================================
Uses temporal features with sample weighting and era-aware modeling
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedFeatureSelector:
    """Select enhanced features for modeling"""
    
    @staticmethod
    def get_enhanced_base_features() -> List[str]:
        """Core enhanced features"""
        return [
            # Original match state
            'over', 'ball_in_over', 'inning', 'balls_elapsed', 'balls_remaining',
            'runs_so_far', 'wickets_so_far', 'wickets_remaining',
            'current_run_rate', 'required_run_rate', 'run_rate_pressure',
            
            # Phase
            'is_powerplay', 'is_middle_over', 'is_death_over', 'late_in_over',
            
            # TEMPORAL FEATURES (NEW)
            'is_modern_cricket',  # Post-2019 flag
            # Note: season_weight used in training, not as feature
            
            # VENUE EVOLUTION (NEW)
            'venue_season_avg_rpb', 'venue_season_rpb_volatility',
            'venue_season_wicket_rate', 'venue_season_boundary_rate',
            'venue_evolution_runs', 'venue_evolution_wickets',
            
            # CAREER PHASE (NEW)
            'batsman_career_balls', 'bowler_career_balls',
            'batsman_maturity', 'bowler_maturity',
            
            # FORM & FATIGUE (NEW)
            'batsman_days_since_last_match', 'bowler_days_since_last_match',
            'batsman_rest_quality', 'bowler_rest_quality',
            'batsman_workload_30d', 'bowler_workload_30d',
            'batsman_form_score', 'bowler_form_score',
            
            # EMA ROLLING STATS (REPLACES SIMPLE ROLLING)
            'batsman_rpb_ema_short', 'batsman_rpb_ema_medium', 'batsman_rpb_ema_long',
            'batsman_sr_ema_short', 'batsman_sr_ema_medium', 'batsman_sr_ema_long',
            'batsman_boundary_pct_ema_short', 'batsman_boundary_pct_ema_medium',
            'batsman_dot_pct_ema_short', 'batsman_dot_pct_ema_medium',
            
            'bowler_econ_ema_short', 'bowler_econ_ema_medium', 'bowler_econ_ema_long',
            'bowler_wicket_rate_ema_short', 'bowler_wicket_rate_ema_medium',
            'bowler_dot_pct_ema_short', 'bowler_dot_pct_ema_medium',
            
            # GENIUS-LEVEL FEATURES (NEW)
            'momentum_differential',
            'partnership_run_rate', 'partnership_stability',
            'bowler_overs_bowled', 'bowler_is_fatigued',
            'pressure_index',
            'runs_last_3_overs', 'runs_last_5_overs', 'wickets_last_3_overs',
            'rr_last_3_overs', 'rr_last_5_overs', 'rr_acceleration',
            'batsman_is_settled',
            'boundaries_last_10_balls', 'dots_last_10_balls'
        ]
    
    @staticmethod
    def get_categorical_features() -> List[str]:
        """Categorical features + career phases"""
        return [
            'venue', 'city', 'phase', 'batter', 'bowler', 'non_striker',
            'cricket_era', 'batsman_career_phase', 'bowler_career_phase'
        ]


class EnhancedMultiTaskModel:
    """
    Enhanced model with temporal awareness
    """
    
    def __init__(self, config=None):
        if config is None:
            config = self._get_default_config()
        self.config = config
        self.models = {}
        self.calibrators = {}
        
    def _get_default_config(self):
        """Default LightGBM config"""
        return {
            'classification': {
                'objective': 'multiclass',
                'num_class': 9,
                'metric': 'multi_logloss',
                'learning_rate': 0.05,
                'num_leaves': 64,
                'max_depth': 8,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                'verbose': -1
            },
            'binary': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': 0.05,
                'num_leaves': 64,
                'max_depth': 8,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                'verbose': -1
            },
            'regression': {
                'objective': 'regression',
                'metric': 'mae',
                'learning_rate': 0.05,
                'num_leaves': 64,
                'max_depth': 8,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                'verbose': -1
            }
        }
    
    def prepare_data_enhanced(self, df: pd.DataFrame, target: str, 
                            is_training: bool = True) -> Tuple:
        """Prepare features with temporal weighting"""
        
        feature_cols = EnhancedFeatureSelector.get_enhanced_base_features()
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        
        # Handle categorical features
        categorical_cols = [col for col in EnhancedFeatureSelector.get_categorical_features() 
                          if col in X.columns]
        
        for col in categorical_cols:
            X[col] = X[col].astype('category')
        
        # Extract sample weights from season_weight
        sample_weights = df['season_weight'].values if 'season_weight' in df.columns else None
        
        if is_training and target in df.columns:
            y = df[target].copy()
            return X, y, sample_weights
        else:
            return X, None, sample_weights
    
    def train_enhanced_classifier(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                 target: str, task_type: str = 'multiclass') -> Dict:
        """
        Train classifier with temporal sample weighting
        
        Args:
            target: 'runs_class' or 'wicket_flag'
            task_type: 'multiclass' or 'binary'
        """
        logger.info(f"Training enhanced {target} classifier with temporal weighting...")
        
        X_train, y_train, train_weights = self.prepare_data_enhanced(train_df, target, True)
        X_val, y_val, val_weights = self.prepare_data_enhanced(val_df, target, True)
        
        # Apply class weights for imbalanced data
        if task_type == 'multiclass':
            class_weights = len(y_train) / (9 * np.bincount(y_train.astype(int)))
            class_sample_weights = np.array([class_weights[int(y)] for y in y_train])
            # Combine class weights with temporal weights
            combined_weights = class_sample_weights * train_weights if train_weights is not None else class_sample_weights
            
            params = self.config['classification'].copy()
        elif task_type == 'binary':
            pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            params = self.config['binary'].copy()
            params['scale_pos_weight'] = pos_weight
            combined_weights = train_weights if train_weights is not None else np.ones(len(y_train))
        
        # Create datasets with weights
        train_data = lgb.Dataset(X_train, label=y_train, weight=combined_weights,
                                categorical_feature='auto')
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature='auto',
                              reference=train_data)
        
        # Train
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(50)
            ]
        )
        
        # Evaluate
        if task_type == 'multiclass':
            y_pred_proba = model.predict(X_val)
            logloss = log_loss(y_val, y_pred_proba)
            logger.info(f"{target} - Log Loss: {logloss:.4f}")
            metrics = {'log_loss': logloss}
        else:
            y_pred_proba = model.predict(X_val)
            logloss = log_loss(y_val, y_pred_proba)
            brier = brier_score_loss(y_val, y_pred_proba)
            logger.info(f"{target} - Log Loss: {logloss:.4f}, Brier: {brier:.4f}")
            metrics = {'log_loss': logloss, 'brier': brier}
        
        self.models[target] = model
        
        return metrics
    
    def train_enhanced_regressor(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                               target: str, quantiles: List[float] = [0.05, 0.5, 0.95]) -> Dict:
        """Train quantile regressor with temporal weighting"""
        
        logger.info(f"Training enhanced quantile regressor for {target}...")
        
        X_train, y_train, train_weights = self.prepare_data_enhanced(train_df, target, True)
        X_val, y_val, val_weights = self.prepare_data_enhanced(val_df, target, True)
        
        models = {}
        predictions = {}
        
        for q in quantiles:
            logger.info(f"  Training quantile {q} with temporal weighting...")
            
            params = self.config['regression'].copy()
            params.update({
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile'
            })
            
            train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights,
                                    categorical_feature='auto')
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature='auto',
                                  reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(100)
                ]
            )
            
            models[f'q{int(q*100)}'] = model
            predictions[f'q{int(q*100)}'] = model.predict(X_val)
        
        # Evaluate
        y_pred_median = predictions['q50']
        mae = mean_absolute_error(y_val, y_pred_median)
        coverage_90 = np.mean((y_val >= predictions['q5']) & (y_val <= predictions['q95']))
        
        logger.info(f"{target} - MAE: {mae:.2f}, 90% Coverage: {coverage_90:.2%}")
        
        self.models[target] = models
        
        return {'mae': mae, 'coverage_90': coverage_90}
    
    def predict(self, df: pd.DataFrame, task: str) -> Dict:
        """Make predictions"""
        
        if task == 'runs_class':
            X, _, _ = self.prepare_data_enhanced(df, 'runs_class', is_training=False)
            proba = self.models['runs_class'].predict(X)
            
            return {
                'probabilities': proba,
                'predicted_class': np.argmax(proba, axis=1),
                'confidence': np.max(proba, axis=1),
                'entropy': -np.sum(proba * np.log(proba + 1e-10), axis=1)
            }
        
        elif task == 'wicket_flag':
            X, _, _ = self.prepare_data_enhanced(df, 'wicket_flag', is_training=False)
            proba = self.models['wicket_flag'].predict(X)
            
            return {
                'probabilities': proba,
                'probability': proba,
                'prediction': (proba > 0.5).astype(int)
            }
        
        elif task in ['runs_next_over', 'runs_batsman_next_over']:
            X, _, _ = self.prepare_data_enhanced(df, task, is_training=False)
            
            quantile_preds = {}
            for q_name, model in self.models[task].items():
                quantile_preds[q_name] = model.predict(X)
            
            return {
                'median': quantile_preds['q50'],
                'lower_90': quantile_preds['q5'],
                'upper_90': quantile_preds['q95'],
                'uncertainty': quantile_preds['q95'] - quantile_preds['q5']
            }
        
        else:
            raise ValueError(f"Unknown task: {task}")


def train_era_specific_models(train_df: pd.DataFrame, val_df: pd.DataFrame,
                              modern_only: bool = False) -> Tuple:
    """
    Train separate models for different cricket eras
    
    Strategy:
    - If modern_only=True: Train only on post-2019 data
    - If modern_only=False: Train on all data with temporal weighting
    """
    
    if modern_only:
        logger.info("Training MODERN ERA models (2019+)")
        train_modern = train_df[train_df['is_modern_cricket'] == 1]
        val_modern = val_df[val_df['is_modern_cricket'] == 1]
        
        logger.info(f"Modern data: Train={len(train_modern):,}, Val={len(val_modern):,}")
        
        if len(train_modern) < 1000:
            logger.warning("Insufficient modern data, falling back to weighted approach")
            modern_only = False
    
    if modern_only:
        train_data = train_modern
        val_data = val_modern
        model_suffix = "_modern"
    else:
        logger.info("Training on ALL ERAS with temporal weighting")
        train_data = train_df
        val_data = val_df
        model_suffix = "_weighted"
    
    # Train models
    mtm = EnhancedMultiTaskModel()
    
    results = {}
    
    # 1. Runs classifier
    results['runs_class'] = mtm.train_enhanced_classifier(
        train_data, val_data, 'runs_class', 'multiclass'
    )
    
    # 2. Wicket classifier
    results['wicket_flag'] = mtm.train_enhanced_classifier(
        train_data, val_data, 'wicket_flag', 'binary'
    )
    
    # 3. Runs regressors
    results['runs_next_over'] = mtm.train_enhanced_regressor(
        train_data, val_data, 'runs_next_over'
    )
    
    results['runs_batsman_next_over'] = mtm.train_enhanced_regressor(
        train_data, val_data, 'runs_batsman_next_over'
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Summary{model_suffix}")
    logger.info(f"{'='*60}")
    for task, metrics in results.items():
        logger.info(f"{task}: {metrics}")
    
    return mtm, results


# ============================================================================
# MONTE CARLO SIMULATION COMPONENTS
# ============================================================================

from dataclasses import dataclass, field
import time

@dataclass
class MatchState:
    """Current state of a cricket match"""
    # Match context
    match_id: str
    venue: str
    city: str
    inning: int
    batting_team: str
    bowling_team: str
    
    # Current state
    over: int
    ball_in_over: int
    runs_so_far: int
    wickets_so_far: int
    balls_remaining: int
    wickets_remaining: int
    
    # Players
    batter: str
    non_striker: str
    bowler: str
    
    # Context features
    current_run_rate: float = 0.0
    required_run_rate: float = 0.0
    run_rate_pressure: float = 0.0
    target: Optional[int] = None
    
    # Phase
    phase: str = "Powerplay"
    is_powerplay: int = 0
    is_middle_over: int = 0
    is_death_over: int = 0
    
    # Enhanced features (from feature engineering)
    features: Dict = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert state to DataFrame for model prediction"""
        data = {
            'over': self.over,
            'ball_in_over': self.ball_in_over,
            'inning': self.inning,
            'runs_so_far': self.runs_so_far,
            'wickets_so_far': self.wickets_so_far,
            'balls_remaining': self.balls_remaining,
            'wickets_remaining': self.wickets_remaining,
            'current_run_rate': self.current_run_rate,
            'required_run_rate': self.required_run_rate,
            'run_rate_pressure': self.run_rate_pressure,
            'is_powerplay': self.is_powerplay,
            'is_middle_over': self.is_middle_over,
            'is_death_over': self.is_death_over,
            'late_in_over': 1 if self.ball_in_over >= 4 else 0,
            'venue': self.venue,
            'city': self.city,
            'batter': self.batter,
            'bowler': self.bowler,
            'non_striker': self.non_striker,
            'phase': self.phase,
            'balls_elapsed': 120 - self.balls_remaining,
            **self.features  # Additional enhanced features
        }
        
        # Add default values for enhanced features
        default_features = {
            'is_modern_cricket': 1,
            'venue_season_avg_rpb': 0.6,
            'venue_season_rpb_volatility': 0.15,
            'venue_season_wicket_rate': 0.05,
            'venue_season_boundary_rate': 0.15,
            'venue_evolution_runs': 0.0,
            'venue_evolution_wickets': 0.0,
            'batsman_career_balls': 500,
            'bowler_career_balls': 300,
            'batsman_maturity': 0.6,
            'bowler_maturity': 0.5,
            'batsman_days_since_last_match': 5,
            'bowler_days_since_last_match': 5,
            'batsman_rest_quality': 0.6,
            'bowler_rest_quality': 0.6,
            'batsman_workload_30d': 5,
            'bowler_workload_30d': 5,
            'batsman_form_score': 120,
            'bowler_form_score': 70,
            'batsman_rpb_ema_short': 0.5,
            'batsman_rpb_ema_medium': 0.5,
            'batsman_rpb_ema_long': 0.5,
            'batsman_sr_ema_short': 125,
            'batsman_sr_ema_medium': 120,
            'batsman_sr_ema_long': 115,
            'batsman_boundary_pct_ema_short': 15,
            'batsman_boundary_pct_ema_medium': 12,
            'batsman_dot_pct_ema_short': 30,
            'batsman_dot_pct_ema_medium': 32,
            'bowler_econ_ema_short': 8,
            'bowler_econ_ema_medium': 7.5,
            'bowler_econ_ema_long': 7.2,
            'bowler_wicket_rate_ema_short': 3,
            'bowler_wicket_rate_ema_medium': 2.5,
            'bowler_dot_pct_ema_short': 40,
            'bowler_dot_pct_ema_medium': 38,
            'momentum_differential': 0,
            'partnership_run_rate': 6,
            'partnership_stability': 0.3,
            'bowler_overs_bowled': 0,
            'bowler_is_fatigued': 0,
            'pressure_index': 0,
            'runs_last_3_overs': 0,
            'runs_last_5_overs': 0,
            'wickets_last_3_overs': 0,
            'rr_last_3_overs': 6,
            'rr_last_5_overs': 6,
            'rr_acceleration': 0,
            'batsman_is_settled': 0,
            'boundaries_last_10_balls': 0,
            'dots_last_10_balls': 5,
            'batsman_career_phase': 'Experienced',
            'bowler_career_phase': 'Experienced',
            'cricket_era': 'Modern_T20'
        }
        
        for key, default_value in default_features.items():
            if key not in data:
                data[key] = default_value
        
        return pd.DataFrame([data])
    
    def update_after_ball(self, runs: int, wicket: bool):
        """Update state after a ball is bowled"""
        self.runs_so_far += runs
        
        if wicket:
            self.wickets_so_far += 1
            self.wickets_remaining -= 1
        
        # Update ball count
        self.ball_in_over += 1
        if self.ball_in_over >= 6:
            self.over += 1
            self.ball_in_over = 0
        
        self.balls_remaining -= 1
        
        # Update run rates
        balls_elapsed = 120 - self.balls_remaining
        self.current_run_rate = (self.runs_so_far / balls_elapsed * 6) if balls_elapsed > 0 else 0
        
        if self.inning == 2 and self.target:
            runs_needed = self.target - self.runs_so_far
            self.required_run_rate = (runs_needed / self.balls_remaining * 6) if self.balls_remaining > 0 else 0
            self.run_rate_pressure = self.required_run_rate - self.current_run_rate
        
        # Update phase
        if self.over < 6:
            self.phase = "Powerplay"
            self.is_powerplay = 1
            self.is_middle_over = 0
            self.is_death_over = 0
        elif self.over >= 16:
            self.phase = "Death"
            self.is_powerplay = 0
            self.is_middle_over = 0
            self.is_death_over = 1
        else:
            self.phase = "Middle"
            self.is_powerplay = 0
            self.is_middle_over = 1
            self.is_death_over = 0
    
    def is_innings_over(self) -> bool:
        """Check if innings is complete"""
        return self.wickets_so_far >= 10 or self.balls_remaining <= 0 or (
            self.inning == 2 and self.target and self.runs_so_far >= self.target
        )


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for cricket match prediction
    """
    
    def __init__(self, model: EnhancedMultiTaskModel, n_simulations: int = 5000, random_seed: int = 42):
        """
        Args:
            model: EnhancedMultiTaskModel with trained predictors
            n_simulations: Number of simulation runs
            random_seed: For reproducibility
        """
        self.model = model
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def simulate_ball(self, state: MatchState) -> Tuple[int, bool]:
        """
        Simulate a single ball outcome
        
        Returns:
            (runs_scored, wicket_flag)
        """
        # Get state as DataFrame
        state_df = state.to_dataframe()
        
        # Predict runs distribution
        runs_pred = self.model.predict(state_df, 'runs_class')
        runs_proba = runs_pred['probabilities'][0]
        
        # Predict wicket probability
        wicket_pred = self.model.predict(state_df, 'wicket_flag')
        wicket_proba = wicket_pred['probability'][0]
        
        # Sample runs from categorical distribution
        # Classes: 0,1,2,3,4,5,6,wicket,extra
        runs_class = np.random.choice(9, p=runs_proba)
        
        if runs_class == 7:  # Wicket class
            runs = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
            wicket = True
        elif runs_class == 8:  # Extra
            runs = np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.07, 0.03])
            wicket = False
        else:
            runs = min(runs_class, 6)
            wicket = np.random.random() < wicket_proba
        
        return runs, wicket
    
    def simulate_innings(self, initial_state: MatchState) -> Dict:
        """
        Simulate a complete innings from current state
        
        Returns:
            Dictionary with final score and ball-by-ball details
        """
        state = MatchState(**{k: v for k, v in initial_state.__dict__.items()})  # Deep copy
        
        balls_simulated = []
        
        while not state.is_innings_over():
            runs, wicket = self.simulate_ball(state)
            
            balls_simulated.append({
                'over': state.over,
                'ball': state.ball_in_over,
                'runs': runs,
                'wicket': wicket,
                'score': state.runs_so_far + runs,
                'wickets': state.wickets_so_far + (1 if wicket else 0)
            })
            
            state.update_after_ball(runs, wicket)
        
        return {
            'final_score': state.runs_so_far,
            'wickets_lost': state.wickets_so_far,
            'balls_used': 120 - state.balls_remaining,
            'balls_simulated': balls_simulated
        }
    
    def run_simulations(self, initial_state: MatchState, 
                       n_sims: Optional[int] = None) -> Dict:
        """
        Run multiple simulations and aggregate results
        
        Returns:
            Dictionary with score distribution, win probability, etc.
        """
        if n_sims is None:
            n_sims = self.n_simulations
        
        logger.info(f"Running {n_sims} Monte Carlo simulations...")
        start_time = time.time()
        
        final_scores = []
        wickets_lost = []
        
        for i in range(n_sims):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Completed {i+1}/{n_sims} simulations")
            
            result = self.simulate_innings(initial_state)
            final_scores.append(result['final_score'])
            wickets_lost.append(result['wickets_lost'])
        
        final_scores = np.array(final_scores)
        wickets_lost = np.array(wickets_lost)
        
        elapsed = time.time() - start_time
        logger.info(f"Simulations complete in {elapsed:.2f}s ({n_sims/elapsed:.0f} sims/sec)")
        
        # Calculate statistics
        results = {
            'n_simulations': n_sims,
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
            },
            'raw_scores': final_scores.tolist()
        }
        
        # Win probability (for 2nd innings)
        if initial_state.inning == 2 and initial_state.target:
            wins = np.sum(final_scores >= initial_state.target)
            results['win_probability'] = float(wins / n_sims)
            results['target'] = initial_state.target
            results['win_by_over_probability'] = float(
                np.sum((final_scores >= initial_state.target) & (wickets_lost < 10)) / n_sims
            )
        
        return results
    
    def predict_next_N_balls(self, state: MatchState, N: int = 6) -> Dict:
        """
        Predict outcomes for next N balls with probabilities
        """
        ball_predictions = []
        current_state = MatchState(**{k: v for k, v in state.__dict__.items()})
        
        for ball_num in range(N):
            state_df = current_state.to_dataframe()
            
            runs_pred = self.model.predict(state_df, 'runs_class')
            wicket_pred = self.model.predict(state_df, 'wicket_flag')
            
            ball_predictions.append({
                'ball_number': ball_num + 1,
                'over': current_state.over,
                'ball_in_over': current_state.ball_in_over,
                'runs_probabilities': {
                    str(i): float(runs_pred['probabilities'][0][i]) 
                    for i in range(9)
                },
                'wicket_probability': float(wicket_pred['probability'][0]),
                'expected_runs': float(np.sum(
                    np.arange(9) * runs_pred['probabilities'][0]
                ))
            })
            
            expected_runs = int(np.argmax(runs_pred['probabilities'][0]))
            expected_wicket = wicket_pred['probability'][0] > 0.5
            current_state.update_after_ball(expected_runs, expected_wicket)
            
            if current_state.is_innings_over():
                break
        
        return {
            'predictions': ball_predictions,
            'over_expected_runs': sum(p['expected_runs'] for p in ball_predictions)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import duckdb
    
    # Load enhanced features
    logger.info("="*60)
    logger.info("ENHANCED CRICKET ML PIPELINE WITH MONTE CARLO")
    logger.info("="*60)
    
    conn = duckdb.connect("IPL_cricket.db")
    
    logger.info("\nLoading datasets...")
    train = conn.execute("SELECT * FROM ml_features_unified_train").fetchdf()
    val = conn.execute("SELECT * FROM ml_features_unified_val").fetchdf()
    test = conn.execute("SELECT * FROM ml_features_unified_test").fetchdf()
    conn.close()
    
    logger.info(f"Train: {len(train):,} rows")
    logger.info(f"Val: {len(val):,} rows")
    logger.info(f"Test: {len(test):,} rows")
    
    # Check modern cricket distribution
    logger.info(f"\nModern cricket (2019+):")
    logger.info(f"  Train: {train['is_modern_cricket'].sum():,} ({train['is_modern_cricket'].mean():.1%})")
    logger.info(f"  Val: {val['is_modern_cricket'].sum():,} ({val['is_modern_cricket'].mean():.1%})")
    
    # Train weighted model (RECOMMENDED)
    logger.info("\n" + "="*60)
    logger.info("Training model with temporal weighting")
    logger.info("="*60)
    model, results = train_era_specific_models(train, val, modern_only=False)
    
    # Example Monte Carlo simulation
    logger.info("\n" + "="*60)
    logger.info("MONTE CARLO SIMULATION DEMO")
    logger.info("="*60)
    
    # Create example match state
    state = MatchState(
        match_id="demo_match",
        venue="Wankhede Stadium",
        city="Mumbai",
        inning=2,
        batting_team="Team A",
        bowling_team="Team B",
        over=12,
        ball_in_over=3,
        runs_so_far=95,
        wickets_so_far=3,
        balls_remaining=45,
        wickets_remaining=7,
        batter="Virat Kohli",
        non_striker="AB de Villiers",
        bowler="Jasprit Bumrah",
        target=165,
        features={
            'batsman_sr_ema_medium': 135.0,
            'bowler_econ_ema_medium': 8.2,
            'is_modern_cricket': 1
        }
    )
    
    # Run simulations
    simulator = MonteCarloSimulator(model, n_simulations=2000)
    sim_results = simulator.run_simulations(state)
    
    logger.info("\n" + "="*60)
    logger.info("SIMULATION RESULTS")
    logger.info("="*60)
    logger.info(f"Expected Score: {sim_results['score_distribution']['median']:.0f}")
    logger.info(f"90% Range: {sim_results['score_distribution']['q05']:.0f} - {sim_results['score_distribution']['q95']:.0f}")
    logger.info(f"Win Probability: {sim_results.get('win_probability', 0):.1%}")
    
    # Next over prediction
    next_balls = simulator.predict_next_N_balls(state, N=6)
    logger.info("\n" + "="*60)
    logger.info("NEXT OVER PREDICTION")
    logger.info("="*60)
    logger.info(f"Expected Runs: {next_balls['over_expected_runs']:.1f}")
    for ball in next_balls['predictions'][:3]:
        logger.info(f"  Ball {ball['ball_number']}: {ball['expected_runs']:.2f} runs "
                   f"(Wicket: {ball['wicket_probability']:.1%})")
    
    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info("\nKey improvements applied:")
    logger.info("  ✓ Temporal weighting (recent data prioritized)")
    logger.info("  ✓ Venue evolution tracking")
    logger.info("  ✓ Player career phase modeling")
    logger.info("  ✓ Form vs fatigue dynamics")
    logger.info("  ✓ EMA noise reduction")
    logger.info("  ✓ Genius-level contextual features")
    logger.info("  ✓ Monte Carlo simulation with uncertainty quantification")
    logger.info("\nModel ready for production use!")