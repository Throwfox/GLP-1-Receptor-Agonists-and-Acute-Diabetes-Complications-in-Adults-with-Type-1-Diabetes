"""Minimal propensity score matching example with optional balancing strategies.

This script focuses on the core propensity score matching (PSM) workflow:
estimating propensity scores with logistic regression and pairing treated
individuals with controls inside a caliper on the logit scale.  To mirror the
production script, it exposes the same two balancing options: a `psmpy` style
chunked re-fit and random over-sampling (`ros`).  All preprocessing (feature
engineering, missing value handling, etc.) is assumed to be complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Core utilities
# -----------------------------------------------------------------------------


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable logit transform."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def estimate_propensity_scores(
    df: pd.DataFrame,
    treatment_col: str,
    feature_cols: Iterable[str],
    method: str = "plain",
) -> Tuple[np.ndarray, LogisticRegression]:
    """Fit logistic models under different balancing schemes and return scores."""
    if method not in {"plain", "psmpy", "ros"}:
        raise ValueError(f"Unsupported balancing method: {method}")

    X = df.loc[:, feature_cols]
    y = df[treatment_col].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    treated_mask = y == 1
    treated = scaled_df.loc[treated_mask]
    controls = scaled_df.loc[~treated_mask]

    if method == "plain":
        model = LogisticRegression(max_iter=500)
        model.fit(scaled_df, y)
        scores = model.predict_proba(scaled_df)[:, 1]
        return scores, model

    if treated.empty or controls.empty:
        raise ValueError("Both treated and control observations are required for balancing methods.")

    if method == "psmpy":
        chunk_size = len(treated)
        user_scores = np.zeros(len(treated))
        control_scores: List[float] = []
        iterations = 0

        for start in range(0, len(controls), chunk_size):
            chunk = controls.iloc[start : start + chunk_size]
            if chunk.empty:
                continue

            multiplier = max(1, chunk_size // len(chunk))
            oversampled_chunk = pd.concat([chunk] * multiplier, ignore_index=True)
            training_features = pd.concat([oversampled_chunk, treated.reset_index(drop=True)], ignore_index=True)
            training_labels = [0] * len(oversampled_chunk) + [1] * len(treated)

            model = LogisticRegression(max_iter=500, random_state=start)
            model.fit(training_features, training_labels)

            user_scores += model.predict_proba(treated)[:, 1]
            control_scores.extend(model.predict_proba(chunk)[:, 1])
            iterations += 1

        user_scores /= max(iterations, 1)
        scores = np.empty(len(df))
        scores[treated_mask.to_numpy()] = user_scores
        scores[~treated_mask.to_numpy()] = np.array(control_scores)
        return scores, model

    # method == "ros"
    from imblearn.over_sampling import RandomOverSampler

    sampler = RandomOverSampler(sampling_strategy="minority", random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(scaled_df, y)

    model = LogisticRegression(max_iter=500)
    model.fit(X_resampled, y_resampled)
    scores = model.predict_proba(scaled_df)[:, 1]
    return scores, model


def greedy_match_with_caliper(
    df: pd.DataFrame,
    treatment_col: str,
    score_col: str,
    caliper_width: float = 0.2,
) -> pd.DataFrame:
    """Pair treated observations with the nearest control inside a caliper.

    The caliper is expressed as a proportion of the standard deviation of the
    treated group's logit propensity scores, following common practice.
    """
    treated = df[df[treatment_col] == 1].copy()
    controls = df[df[treatment_col] == 0].copy()

    treated["score_logit"] = _logit(treated[score_col].to_numpy())
    controls["score_logit"] = _logit(controls[score_col].to_numpy())

    sd_logit = treated["score_logit"].std()
    caliper = caliper_width * sd_logit if sd_logit > 0 else 1e-6

    matches: List[dict] = []
    available_controls = controls.copy()

    nbrs = NearestNeighbors(n_neighbors=1)

    for _, row in treated.iterrows():
        if available_controls.empty:
            break

        nbrs.fit(available_controls["score_logit"].to_numpy().reshape(-1, 1))
        distances, indices = nbrs.kneighbors([[row["score_logit"]]])

        distance = distances[0][0]
        if distance > caliper:
            continue

        control_index = available_controls.index[indices[0][0]]
        control_row = available_controls.loc[control_index]

        matches.append(
            {
                "treated_id": row["id"],
                "control_id": control_row["id"],
                "treated_score": row[score_col],
                "control_score": control_row[score_col],
                "logit_distance": float(distance),
            }
        )

        available_controls = available_controls.drop(index=control_index)

    return pd.DataFrame(matches)


def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    feature_cols: Iterable[str],
    *,
    caliper_width: float = 0.2,
    balancing_method: str = "plain",
) -> pd.DataFrame:
    """High-level helper that runs PSM end-to-end on prepared data."""
    scores, _ = estimate_propensity_scores(df, treatment_col, feature_cols, method=balancing_method)
    df = df.copy()
    df["propensity_score"] = scores
    return greedy_match_with_caliper(df, treatment_col, "propensity_score", caliper_width)


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """Simple container controlling the size of the synthetic demo."""

    n_treated: int = 80
    n_control: int = 160
    random_state: int = 42


def make_synthetic_dataset(cfg: SyntheticConfig) -> pd.DataFrame:
    """Generate fake, already-cleaned covariates to illustrate the workflow."""
    rng = np.random.default_rng(cfg.random_state)

    treated = pd.DataFrame(
        {
            "id": np.arange(cfg.n_treated),
            "treatment": 1,
            "age": rng.normal(58, 8, cfg.n_treated),
            "bmi": rng.normal(32, 4, cfg.n_treated),
            "a1c": rng.normal(8.0, 0.8, cfg.n_treated),
        }
    )

    controls = pd.DataFrame(
        {
            "id": np.arange(cfg.n_treated, cfg.n_treated + cfg.n_control),
            "treatment": 0,
            "age": rng.normal(55, 10, cfg.n_control),
            "bmi": rng.normal(30, 5, cfg.n_control),
            "a1c": rng.normal(7.2, 1.0, cfg.n_control),
        }
    )

    return pd.concat([treated, controls], ignore_index=True)


def main() -> None:
    features = ["age", "bmi", "a1c"]
    data = make_synthetic_dataset(SyntheticConfig())

    for method in ("plain", "psmpy", "ros"):
        try:
            matched_pairs = propensity_score_matching(
                data,
                treatment_col="treatment",
                feature_cols=features,
                caliper_width=0.2,
                balancing_method=method,
            )
            print(f"\nMatched pairs preview ({method}):")
            print(matched_pairs.head())
        except Exception as exc:
            print(f"\nMethod {method} failed: {exc}")


if __name__ == "__main__":
    main()
