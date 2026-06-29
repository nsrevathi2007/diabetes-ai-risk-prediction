"""Optuna-based hyperparameter optimization for selected models."""

from __future__ import annotations

from typing import Any

import optuna


class OptunaOptimizer:
    """Wrap Optuna study creation and execution for baseline model tuning."""

    def __init__(self, model_name: str, n_trials: int = 30, random_state: int = 42) -> None:
        """Initialize the optimizer for a given model family."""
        self.model_name = model_name
        self.n_trials = n_trials
        self.random_state = random_state

    def run_study(self, objective: Any) -> dict[str, Any]:
        """Run an Optuna study using the supplied objective function."""
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return {
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                }
                for trial in study.trials
            ],
        }
