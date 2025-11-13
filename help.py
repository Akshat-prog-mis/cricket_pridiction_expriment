"""
Advanced Cricket ML Pipeline - Part 4: Explainability & Evaluation
===================================================================
This module implements:
1. SHAP-based explanations for predictions
2. Comprehensive evaluation metrics (calibration, Brier, ECE)
3. Human-readable explanations
4. Visualization utilities
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    log_loss, brier_score_loss, accuracy_score, 
    precision_recall_fscore_support, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for models"""
    # Feature columns (will be populated)
    feature_cols: List[str] = None
    
    # Categorical features
    categorical_features: List[str] = None
    
    # LightGBM params
    lgb_params_classification: Dict = None
    lgb_params_regression: Dict = None
    
    # Training params
    early_stopping_rounds: int = 50
    n_estimators: int = 1000
    
    def __post_init__(self):
        if self.categorical_features is None:
            self.categorical_features = [
                'venue', 'city', 'phase', 'batter', 'bowler', 
                'non_striker', 'toss_winner', 'toss_decision'
            ]
        
        if self.lgb_params_classification is None:
            self.lgb_params_classification = {
                'objective': 'multiclass',
                'num_class': 9,  # 0-6 runs, wicket, extra
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
                'verbose': -1,
                'device': 'cpu'
            }
        
        if self.lgb_params_regression is None:
            self.lgb_params_regression = {
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


class FeatureSelector:
    """Select relevant features for modeling"""
    
    @staticmethod
    def get_base_features() -> List[str]:
        """Core features for all models"""
        return [
            # Match state
            'over', 'ball_in_over', 'inning', 'balls_elapsed', 'balls_remaining',
            'runs_so_far', 'wickets_so_far', 'wickets_remaining',
            'current_run_rate', 'required_run_rate', 'run_rate_pressure',
            
            # Phase
            'is_powerplay', 'is_middle_over', 'is_death_over', 'late_in_over',
            
            # Rolling stats - batsman
            'batsman_rpb_short', 'batsman_rpb_medium', 'batsman_rpb_long',
            'batsman_sr_short', 'batsman_sr_medium', 'batsman_sr_long',
            'batsman_boundary_pct_short', 'batsman_boundary_pct_medium',
            'batsman_dot_pct_short', 'batsman_dot_pct_medium',
            
            # Rolling stats - bowler
            'bowler_econ_short', 'bowler_econ_medium', 'bowler_econ_long',
            'bowler_wicket_rate_short', 'bowler_wicket_rate_medium',
            'bowler_dot_pct_short', 'bowler_dot_pct_medium',
            
            # Venue
            'venue_avg_first_innings', 'venue_avg_rpb', 
            'venue_boundary_rate', 'venue_wicket_rate',
            
            # Matchup
            'h2h_balls', 'h2h_rpb', 'h2h_dismissal_rate', 'h2h_strike_rate',
            
            # Momentum
            'runs_last_3_overs', 'wickets_last_3_overs', 
            'rr_last_3_overs', 'rr_acceleration',
            'batsman_balls_faced_this_inning', 'bowler_balls_bowled_this_inning',
            'partnership_runs'
        ]
    
    @staticmethod
    def get_categorical_features() -> List[str]:
        """Categorical features for encoding"""
        return ['venue', 'city', 'phase', 'batter', 'bowler', 'non_striker']


class MultiTaskModel:
    """
    Multi-task model manager for cricket predictions
    Trains separate models for each task
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.calibrators = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, target: str, is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features and target for training or prediction"""
        
        # Select features
        feature_cols = FeatureSelector.get_base_features()
        
        # Filter columns that exist in dataframe
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        
        # Handle categorical features
        categorical_cols = [col for col in FeatureSelector.get_categorical_features() if col in X.columns]
        
        for col in categorical_cols:
            X[col] = X[col].astype('category')
        
        # Only extract target if training
        if is_training and target in df.columns:
            y = df[target].copy()
            return X, y
        else:
            return X, None
    
    def predict(self, df: pd.DataFrame, task: str, return_uncertainty: bool = True) -> Dict:
        """
        Make predictions with uncertainty estimates
        
        Returns:
            Dictionary with predictions and uncertainty intervals
        """
        
        if task == 'runs_class':
            X, _ = self.prepare_data(df, 'runs_class', is_training=False)
            proba = self.models['runs_class'].predict(X)
            
            return {
                'probabilities': proba,
                'predicted_class': np.argmax(proba, axis=1),
                'confidence': np.max(proba, axis=1),
                'entropy': -np.sum(proba * np.log(proba + 1e-10), axis=1)  # Uncertainty
            }
        
        elif task == 'wicket_flag':
            X, _ = self.prepare_data(df, 'wicket_flag', is_training=False)
            proba = self.models['wicket_flag'].predict(X)
            
            # Apply calibration
            if 'wicket_flag' in self.calibrators:
                proba = self.calibrators['wicket_flag'].predict(proba)
            
            return {
                'probability': proba,
                'prediction': (proba > 0.5).astype(int)
            }
        
        elif task in ['runs_next_over', 'runs_batsman_next_over']:
            X, _ = self.prepare_data(df, task, is_training=False)
            
            quantile_preds = {}
            for q_name, model in self.models[task].items():
                quantile_preds[q_name] = model.predict(X)
            
            return {
                'median': quantile_preds['q50'],
                'lower_90': quantile_preds['q5'],
                'upper_90': quantile_preds['q95'],
                'lower_50': quantile_preds.get('q25', quantile_preds['q50']),
                'upper_50': quantile_preds.get('q75', quantile_preds['q50']),
                'uncertainty': quantile_preds['q95'] - quantile_preds['q5']  # Width of interval
            }
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def load_models(self, path: str = "models/"):
        """Load all trained models and calibrators"""
        import lightgbm as lgb
        
        # Load LightGBM models
        model_files = os.listdir(path)
        
        # Load runs_class model
        if 'runs_class.txt' in model_files:
            self.models['runs_class'] = lgb.Booster(model_file=f"{path}/runs_class.txt")
            logger.info("Loaded runs_class model")
        
        # Load wicket_flag model
        if 'wicket_flag.txt' in model_files:
            self.models['wicket_flag'] = lgb.Booster(model_file=f"{path}/wicket_flag.txt")
            logger.info("Loaded wicket_flag model")
        
        # Load quantile models for runs_next_over
        runs_over_models = {}
        for q_file in [f for f in model_files if f.startswith('runs_next_over') and f.endswith('.txt')]:
            q_name = q_file.replace('runs_next_over_', '').replace('.txt', '')
            runs_over_models[q_name] = lgb.Booster(model_file=f"{path}/{q_file}")
        if runs_over_models:
            self.models['runs_next_over'] = runs_over_models
            logger.info("Loaded runs_next_over quantile models")
        
        # Load quantile models for runs_batsman_next_over
        batsman_runs_models = {}
        for q_file in [f for f in model_files if f.startswith('runs_batsman_next_over') and f.endswith('.txt')]:
            q_name = q_file.replace('runs_batsman_next_over_', '').replace('.txt', '')
            batsman_runs_models[q_name] = lgb.Booster(model_file=f"{path}/{q_file}")
        if batsman_runs_models:
            self.models['runs_batsman_next_over'] = batsman_runs_models
            logger.info("Loaded runs_batsman_next_over quantile models")
        
        # Load calibrators
        if os.path.exists(f"{path}/calibrators.pkl"):
            with open(f"{path}/calibrators.pkl", 'rb') as f:
                self.calibrators = pickle.load(f)
            logger.info("Loaded calibrators")
        
        # Load feature importance
        if os.path.exists(f"{path}/feature_importance.json"):
            with open(f"{path}/feature_importance.json", 'r') as f:
                self.feature_importance = json.load(f)
            logger.info("Loaded feature importance")
        
        logger.info(f"Models loaded from {path}")


class ModelEvaluator:
    """
    Comprehensive model evaluation with calibration analysis
    """
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE)
        
        ECE measures how well predicted probabilities match actual frequencies
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred_proba: np.ndarray,
                            task_name: str = "Classification") -> Dict:
        
        if y_pred_proba.ndim == 1:  # Binary
            """
        Comprehensive classification evaluation
        """
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'log_loss': log_loss(y_true, y_pred_proba),
                'brier_score': brier_score_loss(y_true, y_pred_proba),
                'ece': ModelEvaluator.expected_calibration_error(y_true, y_pred_proba)
            }
            
            # Precision, recall, F1
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            metrics.update({
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            })
            
        else:  # Multi-class
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Handle the case where test set doesn't contain all classes
            # Specify all 9 classes explicitly for log_loss
            all_classes = np.arange(9)  # Classes 0-8
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'log_loss': log_loss(y_true, y_pred_proba, labels=all_classes),
                'macro_f1': precision_recall_fscore_support(
                    y_true, y_pred, average='macro', zero_division=0, labels=all_classes
                )[2]
            }
            
            # Per-class Brier scores
            brier_scores = {}
            for cls in range(y_pred_proba.shape[1]):
                y_binary = (y_true == cls).astype(int)
                brier_scores[f'class_{cls}'] = brier_score_loss(y_binary, y_pred_proba[:, cls])
            
            metrics['brier_scores'] = brier_scores
            metrics['avg_brier_score'] = np.mean(list(brier_scores.values()))
        
        logger.info(f"\n{task_name} Evaluation:")
        for k, v in metrics.items():
            if not isinstance(v, dict):
                logger.info(f"  {k}: {v:.4f}")
        
        return metrics
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray,
                          task_name: str = "Regression") -> Dict:
        """
        Regression evaluation metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # +epsilon to avoid div by 0
        }
        
        logger.info(f"\n{task_name} Evaluation:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        return metrics
    
    @staticmethod
    def evaluate_quantile_coverage(y_true: np.ndarray, 
                                  y_pred_lower: np.ndarray,
                                  y_pred_upper: np.ndarray,
                                  target_coverage: float = 0.9) -> Dict:
        """
        Evaluate quantile regression coverage
        """
        in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        actual_coverage = np.mean(in_interval)
        
        # Interval width
        avg_width = np.mean(y_pred_upper - y_pred_lower)
        
        metrics = {
            'target_coverage': target_coverage,
            'actual_coverage': actual_coverage,
            'coverage_gap': abs(actual_coverage - target_coverage),
            'avg_interval_width': avg_width
        }
        
        logger.info(f"\nQuantile Regression Coverage:")
        logger.info(f"  Target: {target_coverage:.1%}")
        logger.info(f"  Actual: {actual_coverage:.1%}")
        logger.info(f"  Gap: {metrics['coverage_gap']:.3f}")
        logger.info(f"  Avg Width: {avg_width:.2f}")
        
        return metrics
    
    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                              n_bins: int = 10, title: str = "Calibration Curve"):
        """
        Plot reliability diagram
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


class SHAPExplainer:
    """
    SHAP-based model explanations
    """
    
    def __init__(self, model, X_background: pd.DataFrame, model_type: str = "tree"):
        """
        Args:
            model: Trained model
            X_background: Background dataset for SHAP (sample of training data)
            model_type: 'tree' for tree models, 'kernel' for kernel explainer
        """
        import shap
        self.model = model
        self.model_type = model_type
        
        if model_type == "tree":
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                # Fall back to kernel explainer if tree explainer fails
                logger.warning("Tree explainer failed, using kernel explainer instead")
                self.explainer = shap.KernelExplainer(
                    model.predict, 
                    shap.sample(X_background, 100)
                )
                self.model_type = "kernel"
        else:  # kernel
            self.explainer = shap.KernelExplainer(
                model.predict, 
                shap.sample(X_background, 100)
            )
        
        logger.info(f"SHAP explainer initialized with {self.model_type} backend")
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict:
        """
        Explain a single prediction
        
        Returns:
            Dictionary with SHAP values and feature contributions
        """
        import shap
        shap_values = self.explainer.shap_values(X.iloc[[index]], nsamples=100)
        
        # Handle multi-class (returns list of arrays)
        if isinstance(shap_values, list):
            # Use predicted class
            pred_class = np.argmax(self.model.predict(X.iloc[[index]])[0])
            shap_values = shap_values[pred_class]
        elif shap_values.ndim > 1 and shap_values.shape[0] > 1:  # Multi-class kernel
            pred_class = np.argmax(self.model.predict(X.iloc[[index]])[0])
            shap_values = shap_values[pred_class]
        
        # Get top features
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'shap_value': shap_values[0] if shap_values.ndim > 1 else shap_values,
            'feature_value': X.iloc[index].values
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return {
            'shap_values': shap_values[0] if shap_values.ndim > 1 else shap_values,
            'feature_importance': feature_importance,
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else np.mean(self.model.predict(X))
        }
    
    def create_human_explanation(self, X: pd.DataFrame, index: int,
                                prediction: Dict, top_n: int = 5) -> str:
        """
        Generate human-readable explanation
        
        Args:
            X: Input features
            index: Row index
            prediction: Model prediction dictionary
            top_n: Number of top features to include
        
        Returns:
            Natural language explanation
        """
        explanation = self.explain_prediction(X, index)
        top_features = explanation['feature_importance'].head(top_n)
        
        # Extract prediction details
        if 'probabilities' in prediction:  # Classification
            pred_class = prediction['predicted_class'][index]
            confidence = prediction['confidence'][index]
            
            # Map class to outcome
            outcome_map = {
                0: "no runs (dot ball)",
                1: "1 run",
                2: "2 runs",
                3: "3 runs",
                4: "a boundary (4 runs)",
                6: "a six",
                7: "a wicket",
                8: "extras"
            }
            outcome = outcome_map.get(pred_class, f"class {pred_class}")
            
            explanation_text = f"**Prediction: {outcome}** (confidence: {confidence:.1%})\n\n"
            explanation_text += "**Key factors influencing this prediction:**\n\n"
            
        elif 'probability' in prediction:  # Binary
            prob = prediction['probability'][index]
            explanation_text = f"**Wicket Probability: {prob:.1%}**\n\n"
            explanation_text += "**Key factors:**\n\n"
        
        else:  # Regression
            pred_val = prediction.get('median', prediction.get('prediction'))[index]
            uncertainty = prediction.get('uncertainty', [0])[index]
            explanation_text = f"**Predicted Runs: {pred_val:.1f}** (± {uncertainty:.1f})\n\n"
            explanation_text += "**Key factors:**\n\n"
        
        # Add top features with interpretations
        for idx, row in top_features.iterrows():
            feature = row['feature']
            shap_val = row['shap_value']
            feat_val = row['feature_value']
            
            # Interpret feature
            interpretation = self._interpret_feature(feature, feat_val, shap_val)
            explanation_text += f"{idx+1}. {interpretation}\n"
        
        return explanation_text
    
    def _interpret_feature(self, feature: str, value: float, shap_value: float) -> str:
        """
        Create human-readable interpretation of a feature's contribution
        """
        impact = "increases" if shap_value > 0 else "decreases"
        
        interpretations = {
            'batsman_sr_medium': f"Batsman's strike rate ({value:.1f}) {impact} the likelihood",
            'batsman_rpb_short': f"Batsman's recent form ({value:.2f} rpb) {impact} the prediction",
            'bowler_econ_medium': f"Bowler's economy rate ({value:.1f}) {impact} the outcome",
            'bowler_wicket_rate_medium': f"Bowler's wicket-taking ability ({value:.1%}) {impact} risk",
            'is_powerplay': f"{'Powerplay' if value else 'Non-powerplay'} phase {impact} scoring",
            'is_death_over': f"{'Death' if value else 'Non-death'} overs {impact} boundaries",
            'current_run_rate': f"Current run rate ({value:.2f}) {impact} pressure",
            'required_run_rate': f"Required rate ({value:.2f}) {impact} aggressive play",
            'wickets_remaining': f"{int(value)} wickets in hand {impact} approach",
            'balls_remaining': f"{int(value)} balls left {impact} urgency",
            'venue_boundary_rate': f"Venue's boundary tendency ({value:.1%}) {impact} big shots",
            'h2h_rpb': f"Head-to-head record ({value:.2f} rpb) {impact} matchup",
            'rr_acceleration': f"Recent momentum ({value:+.2f}) {impact} continuation"
        }
        
        return interpretations.get(
            feature,
            f"**{feature}** = {value:.2f} {impact} prediction (SHAP: {shap_value:+.3f})"
        )
    
    def plot_feature_importance(self, X: pd.DataFrame, max_display: int = 20):
        """
        Create SHAP summary plot
        """
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            # Plot for each class
            for cls_idx, cls_shap in enumerate(shap_values):
                plt.figure(figsize=(10, 8))
                shap.summary_plot(cls_shap, X, max_display=max_display, show=False)
                plt.title(f"SHAP Feature Importance - Class {cls_idx}")
                plt.tight_layout()
        else:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
        
        return plt.gcf()


class BacktestEngine:
    """
    Backtest win probability predictions against historical outcomes
    """
    
    @staticmethod
    def backtest_win_probability(predictions_df: pd.DataFrame,
                                 actual_outcomes_df: pd.DataFrame) -> Dict:
        """
        Backtest win probability predictions
        
        Args:
            predictions_df: DataFrame with columns ['match_id', 'win_prob_team1']
            actual_outcomes_df: DataFrame with columns ['match_id', 'winner']
        
        Returns:
            Backtest metrics including Brier score
        """
        merged = predictions_df.merge(actual_outcomes_df, on='match_id')
        
        # Convert winner to binary (1 if team1 won, 0 otherwise)
        merged['actual'] = (merged['winner'] == merged['team1']).astype(int)
        
        # Brier score
        brier = brier_score_loss(merged['actual'], merged['win_prob_team1'])
        
        # Log loss
        logloss = log_loss(merged['actual'], merged['win_prob_team1'])
        
        # Calibration
        ece = ModelEvaluator.expected_calibration_error(
            merged['actual'].values,
            merged['win_prob_team1'].values
        )
        
        # Accuracy at different confidence thresholds
        high_conf_mask = (merged['win_prob_team1'] > 0.7) | (merged['win_prob_team1'] < 0.3)
        high_conf_acc = accuracy_score(
            merged[high_conf_mask]['actual'],
            (merged[high_conf_mask]['win_prob_team1'] > 0.5).astype(int)
        )
        
        metrics = {
            'brier_score': brier,
            'log_loss': logloss,
            'ece': ece,
            'overall_accuracy': accuracy_score(
                merged['actual'],
                (merged['win_prob_team1'] > 0.5).astype(int)
            ),
            'high_confidence_accuracy': high_conf_acc,
            'n_matches': len(merged)
        }
        
        logger.info("\nWin Probability Backtest:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        return metrics


def generate_prediction_report(X: pd.DataFrame, 
                              predictions: Dict,
                              explainer: Optional[SHAPExplainer] = None,
                              index: int = 0) -> str:
    """
    Generate comprehensive prediction report for a single ball
    
    This is the "genius feel" output combining all insights
    """
    report = "="*70 + "\n"
    report += "CRICKET ML PREDICTION REPORT\n"
    report += "="*70 + "\n\n"
    
    # Current match state
    row = X.iloc[index]
    report += "**CURRENT STATE**\n"
    report += f"Over {row['over']}.{row['ball_in_over']} | "
    report += f"Score: {row['runs_so_far']}/{row['wickets_so_far']} | "
    report += f"CRR: {row['current_run_rate']:.2f}\n"
    
    if row['inning'] == 2:
        report += f"Target: {row.get('target', 'N/A')} | "
        report += f"RRR: {row['required_run_rate']:.2f} | "
        report += f"Pressure: {row['run_rate_pressure']:+.2f}\n"
    
    report += f"\n{row['batter']} facing {row['bowler']} at {row['venue']}\n"
    report += f"Phase: {row['phase']} | Balls Remaining: {row['balls_remaining']}\n\n"
    
    # Predictions
    report += "**PREDICTIONS**\n"
    
    if 'probabilities' in predictions:  # Runs class
        probs = predictions['probabilities'][index]
        report += "Next ball outcome probabilities:\n"
        for i, prob in enumerate(probs):
            if i < 7:
                report += f"  {i} runs: {prob:.1%}\n"
            elif i == 7:
                report += f"  Wicket: {prob:.1%}\n"
            else:
                report += f"  Extra: {prob:.1%}\n"
        
        # Highlight most likely
        top_outcome = np.argmax(probs)
        report += f"\nMost likely: {top_outcome} ({'wicket' if top_outcome == 7 else f'{top_outcome} runs'})\n"
        report += f"Confidence: {predictions['confidence'][index]:.1%}\n"
        report += f"Prediction uncertainty: {predictions['entropy'][index]:.3f}\n\n"
    
    if 'probability' in predictions:  # Wicket
        prob = predictions['probability'][index]
        report += f"Wicket probability: {prob:.1%}\n\n"
    
    # Explanation
    if explainer:
        report += "**KEY FACTORS**\n"
        explanation = explainer.create_human_explanation(X, index, predictions)
        report += explanation + "\n"
    
    report += "="*70 + "\n"
    
    return report


# Example usage
if __name__ == "__main__":
    import duckdb
    from dataclasses import dataclass
    
    # Load test data
    conn = duckdb.connect("ipl_cricket.db")
    test_df = conn.execute("SELECT * FROM test_set LIMIT 1000").fetchdf()
    train_sample = conn.execute("SELECT * FROM train_set LIMIT 5000").fetchdf()
    conn.close()
    
    # Load models (assuming trained)
    config = ModelConfig()
    mtm = MultiTaskModel(config)
    mtm.load_models("models/")
    
    # Verify models are loaded
    if 'runs_class' not in mtm.models:
        raise ValueError("runs_class model not found. Please train and save models first.")
    if 'wicket_flag' not in mtm.models:
        raise ValueError("wicket_flag model not found. Please train and save models first.")
    
    # Evaluate runs classifier
    X_test, y_test = mtm.prepare_data(test_df, 'runs_class')
    predictions = mtm.predict(test_df, 'runs_class')
    
    metrics = ModelEvaluator.evaluate_classification(
        y_test.values,
        predictions['probabilities'],
        "Runs Class"
    )
    
    # Evaluate wicket classifier
    X_test_wicket, y_test_wicket = mtm.prepare_data(test_df, 'wicket_flag')
    wicket_preds = mtm.predict(test_df, 'wicket_flag')
    
    wicket_metrics = ModelEvaluator.evaluate_classification(
        y_test_wicket.values,
        wicket_preds['probability'],
        "Wicket Probability"
    )
    
    # Plot calibration
    fig = ModelEvaluator.plot_calibration_curve(
        y_test_wicket.values,
        wicket_preds['probability'],
        title="Wicket Probability Calibration"
    )
    fig.savefig("wicket_calibration.png")
    
    # SHAP explanation - use KernelExplainer instead of TreeExplainer to avoid JSON issues
    try:
        explainer = SHAPExplainer(mtm.models['runs_class'], X_test.head(100), model_type="kernel")
        
        # Generate report for a sample prediction
        report = generate_prediction_report(
            test_df,
            predictions,
            explainer,
            index=42  # Sample index
        )
        
        print(report)
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        logger.info("Generating prediction report without SHAP explanations...")
        
        # Generate report without SHAP explanations
        report = generate_prediction_report(
            test_df,
            predictions,
            explainer=None,
            index=42  # Sample index
        )
        
        print(report)