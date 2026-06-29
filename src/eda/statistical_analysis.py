"""
Statistical analysis module for EDA pipeline.

Computes statistical test scores including Mutual Information and Chi-Square
for feature ranking and importance assessment.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler


class StatisticalAnalysis:
    """
    Performs statistical analysis for feature ranking and selection.
    
    This class computes mutual information scores and chi-square scores
    for numerical and categorical features.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the StatisticalAnalysis analyzer.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_mutual_information(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary",
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Compute mutual information scores between features and target.
        
        Args:
            df: DataFrame containing features and target.
            target_column: Name of the target column.
            random_state: Random state for reproducibility.
            
        Returns:
            Dictionary mapping feature names to MI scores.
        """
        self.logger.info("Computing Mutual Information scores")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Compute mutual information
        mi_scores = mutual_info_classif(X, y, random_state=random_state)
        
        mi_dict = {
            feature: float(score)
            for feature, score in zip(X.columns, mi_scores)
        }
        
        # Sort by score
        mi_dict = dict(sorted(mi_dict.items(), key=lambda x: x[1], reverse=True))
        
        self.logger.info(f"Mutual Information scores computed for {len(mi_dict)} features")
        return mi_dict
    
    def compute_chi_square(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary"
    ) -> Dict[str, float]:
        """
        Compute Chi-Square scores for categorical features.
        
        Args:
            df: DataFrame containing features and target.
            target_column: Name of the target column.
            
        Returns:
            Dictionary mapping categorical feature names to Chi-Square scores.
        """
        self.logger.info("Computing Chi-Square scores")
        
        y = df[target_column]
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        chi2_dict = {}
        
        for feature in categorical_features:
            try:
                X = df[[feature]].fillna(0)
                # Ensure non-negative values for chi2
                X = X - X.min() + 1
                
                scores, _ = chi2(X, y)
                chi2_dict[feature] = float(scores[0])
            except Exception as e:
                self.logger.warning(f"Could not compute Chi-Square for {feature}: {str(e)}")
        
        # Sort by score
        chi2_dict = dict(sorted(chi2_dict.items(), key=lambda x: x[1], reverse=True))
        
        self.logger.info(f"Chi-Square scores computed for {len(chi2_dict)} features")
        return chi2_dict
    
    def rank_features(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary",
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Rank features using multiple statistical methods.
        
        Args:
            df: DataFrame containing features and target.
            target_column: Name of the target column.
            random_state: Random state for reproducibility.
            
        Returns:
            Dictionary containing feature rankings.
        """
        self.logger.info("Computing feature rankings")
        
        # Compute Mutual Information
        mi_scores = self.compute_mutual_information(df, target_column, random_state)
        
        # Compute Chi-Square
        chi2_scores = self.compute_chi_square(df, target_column)
        
        # Create combined ranking
        all_features = set(mi_scores.keys()) | set(chi2_scores.keys())
        
        combined_ranking = {}
        for feature in all_features:
            combined_ranking[feature] = {
                "mutual_information": mi_scores.get(feature, 0.0),
                "chi_square": chi2_scores.get(feature, 0.0),
            }
        
        # Sort by mutual information
        combined_ranking = dict(
            sorted(
                combined_ranking.items(),
                key=lambda x: x[1]["mutual_information"],
                reverse=True
            )
        )
        
        ranking = {
            "mutual_information_scores": mi_scores,
            "chi_square_scores": chi2_scores,
            "combined_ranking": combined_ranking,
            "top_10_features": list(combined_ranking.keys())[:10],
        }
        
        self.logger.info("Feature ranking completed")
        return ranking
    
    def save_ranking(
        self,
        ranking: Dict[str, Any],
        output_path: str
    ) -> Path:
        """
        Save feature ranking to JSON file.
        
        Args:
            ranking: Feature ranking dictionary.
            output_path: Path to save the JSON file.
            
        Returns:
            Path object pointing to the saved file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_file.write_text(json.dumps(ranking, indent=2), encoding='utf-8')
        self.logger.info(f"Feature ranking saved to {output_file}")
        
        return output_file
