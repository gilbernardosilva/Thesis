from datetime import timedelta
from typing import Dict

import movingpandas as mpd
import pandas as pd


def calculate_avg_speed(traj: mpd.Trajectory) -> float:
    """
    Para dados RAW: calcula a velocidade média (em km/h) usando MovingPandas.
    """
    traj.add_speed(overwrite=True)
    avg_speed = traj.df.speed.mean() * 3.6
    return avg_speed


def calculate_duration(start_time, end_time) -> timedelta:
    duration = end_time - start_time
    return duration


def calculate_probes_per_second(df: pd.DataFrame) -> float:
    total_probes = len(df)
    time_duration = (df.index[-1] - df.index[0]).total_seconds()
    return total_probes / time_duration if time_duration > 0 else 0


def calculate_avg_speed_by_address(traj: mpd.Trajectory) -> Dict[str, float]:
    """Calcula a velocidade média por 'address' dentro de uma trajetória, usando calculate_avg_speed."""

    speeds_by_address: Dict[str, float] = {}

    for address in traj.df["address"].unique():
        traj_subset = traj.df[traj.df["address"] == address].copy()

        if len(traj_subset) > 1:

            temp_traj = mpd.Trajectory(traj_subset, traj.id)
            avg_speed = calculate_avg_speed(temp_traj)
            speeds_by_address[address] = avg_speed
        else:
            speeds_by_address[address] = 0.0

    return speeds_by_address
