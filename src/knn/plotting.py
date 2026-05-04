"""
Debug plots for ``knn.predict_curve_knn``.

The two plotting functions are invoked only when the root logger is at
DEBUG level. They write PNGs to ``debug/`` for inspection:

    * ``plot_current_and_neighbors`` - the current event's partial
      trajectory overlaid with the k nearest neighbours' partials.
    * ``plot_neighbors_full_and_prediction`` - neighbours' full curves,
      the prediction, and (when available) the real final trajectory.
"""

import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NeighbourId = Tuple[float, float]


def is_debug_logging() -> bool:
    return logging.getLevelName(logging.getLogger().getEffectiveLevel()) == "DEBUG"


def _silence_matplotlib_logs() -> None:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def plot_current_and_neighbors(
    current_scores: np.ndarray,
    neighbor_ids: np.ndarray,
    candidate_trajectories: List[np.ndarray],
    candidate_ids: List[NeighbourId],
    current_step: int,
    current_event_id: float,
    current_idol_id: int,
    border: float,
    output_dir: str = "debug",
) -> None:
    """Overlay the current event's partial trajectory with its k nearest neighbours."""
    _silence_matplotlib_logs()
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.plot(
        np.arange(len(current_scores)), current_scores,
        color="red", linewidth=3, marker="o", markersize=4,
        label=f"Current Event {int(current_event_id)} (Idol {current_idol_id})",
        zorder=10,
    )

    palette = plt.get_cmap("tab10")(np.linspace(0, 1, len(neighbor_ids)))
    for i, (neighbor_event_id, neighbor_idol_id) in enumerate(neighbor_ids):
        neighbor_trajectory = None
        for j, (candidate_event_id, candidate_idol_id) in enumerate(candidate_ids):
            if candidate_event_id == neighbor_event_id and candidate_idol_id == neighbor_idol_id:
                neighbor_trajectory = candidate_trajectories[j]
                break
        if neighbor_trajectory is None:
            continue
        neighbor_partial = neighbor_trajectory[:current_step]
        plt.plot(
            np.arange(len(neighbor_partial)), neighbor_partial,
            color=palette[i], linewidth=2, marker="s", markersize=3,
            label=f"Neighbor Event {int(neighbor_event_id)} (Idol {neighbor_idol_id})",
            alpha=0.7,
        )

    plt.axvline(
        x=current_step - 1, color="black", linestyle="--", alpha=0.5,
        label=f"Current Step {current_step}",
    )
    plt.title(
        f"Current Event vs Neighbors - Step {current_step}\n"
        f"Event {int(current_event_id)}, Idol {current_idol_id}, Border {int(border)}",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    filename = (
        f"current_vs_neighbors_e{int(current_event_id)}_i{current_idol_id}"
        f"_s{current_step}_b{int(border)}.png"
    )
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Current vs neighbors plot saved to {output_path}")


def plot_neighbors_full_and_prediction(
    current_partial_data: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_full_list: List[np.ndarray],
    prediction: np.ndarray,
    current_step: int,
    current_event_id: float,
    current_idol_id: int,
    border: float,
    prediction_full_df: pd.DataFrame,
    output_dir: str = "debug",
) -> None:
    """Overlay neighbours' full curves, the prediction, and the real final trajectory."""
    _silence_matplotlib_logs()
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))
    plt.plot(
        np.arange(len(current_partial_data)), current_partial_data,
        color="red", linewidth=3, marker="o", markersize=4,
        label=f"Current Event {int(current_event_id)} (Known)", zorder=10,
    )

    palette = plt.get_cmap("tab10")(np.linspace(0, 1, len(neighbor_ids)))
    for i, (neighbor_event_id, neighbor_idol_id) in enumerate(neighbor_ids):
        neighbor_trajectory = neighbor_full_list[i]
        plt.plot(
            np.arange(len(neighbor_trajectory)), neighbor_trajectory,
            color=palette[i], linewidth=2, marker="s", markersize=2,
            label=f"Neighbor Event {int(neighbor_event_id)} (Idol {neighbor_idol_id})",
            alpha=0.7,
        )

    if len(prediction) > 0:
        plt.plot(
            np.arange(len(prediction)), prediction,
            color="blue", linewidth=3, marker="^", markersize=3,
            label=f"Prediction for Event {int(current_event_id)}",
            linestyle="--", alpha=0.9, zorder=8,
        )

    real_df = prediction_full_df[
        (prediction_full_df["event_id"] == current_event_id)
        & (prediction_full_df["idol_id"] == current_idol_id)
    ]
    if len(real_df) > 0:
        real_trajectory = real_df["score"].to_numpy()
        real_final_value = real_trajectory[-1]
        final_step = len(real_trajectory) - 1
        plt.plot(
            final_step, real_final_value,
            color="green", marker="o", markersize=10,
            label=f"Real Final Value: {real_final_value:.0f}", zorder=12,
        )
        plt.plot(
            np.arange(len(real_trajectory)), real_trajectory,
            color="green", linewidth=2, alpha=0.5,
            label=f"Real Full Trajectory (Event {int(current_event_id)})",
            linestyle="-", zorder=5,
        )

    plt.axvline(
        x=current_step - 1, color="black", linestyle="--", alpha=0.5,
        label=f"Current Step {current_step}",
    )
    if len(prediction) > current_step:
        plt.axvspan(
            current_step - 1, len(prediction) - 1,
            alpha=0.1, color="gray", label="Future Steps",
        )

    plt.title(
        f"Full Trajectories, Prediction & Real Final Value - Step {current_step}\n"
        f"Event {int(current_event_id)}, Idol {current_idol_id}, Border {int(border)}",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    filename = (
        f"full_prediction_e{int(current_event_id)}_i{current_idol_id}"
        f"_s{current_step}_b{int(border)}.png"
    )
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Full prediction plot saved to {output_path}")
