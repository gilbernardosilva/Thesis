import logging
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import traci

from app.fcd.config.schemas import SUMO_ACCIDENTS_SCHEMA
from app.fcd.config.variables import (
    SUMO_ACCIDENT_PROB,
    SUMO_ACCIDENTS_FILE,
    SUMO_CONFIG_FILE,
    SUMO_END_TIME,
    SUMO_FCD_FILE,
    SUMO_HOME,
    SUMO_MAX_ACCIDENT_DURATION,
    SUMO_MIN_ACCIDENT_DURATION,
    SUMO_NET_FILE,
    SUMO_ROUTES_FILE,
    SUMO_STARTING_HOUR,
    SUMO_TRIPS_FILE,
    TS_END,
    TS_START,
)

SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo")
RANDOM_TRIPS_SCRIPT = os.path.join(SUMO_HOME, "tools", "randomTrips.py")

sys.path.append(os.path.join(SUMO_HOME, "tools"))

TS_START = pd.to_datetime(TS_START)
TS_END = pd.to_datetime(TS_END)


def map_to_valhalla_edge(latitude: float, longitude: float) -> int:
    url = "http://localhost:8002/locate"
    payload = {
        "locations": [{"lat": latitude, "lon": longitude}],
        "costing": "auto",
        "filters": {"attributes": ["edge.id"], "action": "include"},
        "radius": 10,
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        if (
            result
            and isinstance(result, list)
            and len(result) > 0
            and "edges" in result[0]
            and len(result[0]["edges"]) > 0
        ):
            edge = result[0]["edges"][0]
            way_id = edge.get("way_id", -1)
            logging.debug(f"Mapped ({latitude}, {longitude}) to edge_id: {way_id}")
            return int(way_id)
        else:
            logging.warning(f"No edge found for coordinates ({latitude}, {longitude})")
            return -1
    except requests.exceptions.RequestException as e:
        logging.error(
            f"Failed to map coordinates ({latitude}, {longitude}) to edge: {e}"
        )
        return -1


def update_sumo_config():
    tree = ET.parse(SUMO_CONFIG_FILE)
    root = tree.getroot()
    input_elem = root.find("input")
    net_file_elem = input_elem.find("net-file")
    route_files_elem = input_elem.find("route-files")
    net_file_elem.set("value", os.path.abspath(SUMO_NET_FILE))
    route_files_elem.set("value", os.path.abspath(SUMO_TRIPS_FILE))
    tree.write(SUMO_CONFIG_FILE)


def generate_random_trips(period=0.5, fringe_factor=5, end_time=3600, max_retries=10):
    seed = int.from_bytes(os.urandom(4), "big")
    for attempt in range(max_retries):
        logging.info(
            f"Attempt {attempt + 1}/{max_retries} with period={period}, "
            f"fringe_factor={fringe_factor}, end_time={end_time}, seed={seed}"
        )
        cmd = [
            "python",
            RANDOM_TRIPS_SCRIPT,
            "-n",
            SUMO_NET_FILE,
            "-o",
            SUMO_TRIPS_FILE,
            "-e",
            str(end_time),
            "-p",
            str(period),
            "--fringe-factor",
            str(fringe_factor),
            "--seed",
            str(seed),
            "--route-file",
            SUMO_ROUTES_FILE,
            "--validate",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logging.info("randomTrips.py executed successfully.")
            logging.info(f"Output:\n{result.stdout}")
            return
        except subprocess.CalledProcessError as e:
            logging.error(f"randomTrips.py failed with error: {e}")
            logging.error(f"Output:\n{e.stdout}")
            logging.error(f"Error:\n{e.stderr}")
            time.sleep(1)
            seed = int.from_bytes(os.urandom(4), "big")
            continue
    raise RuntimeError("Failed to generate random trips after maximum retries.")


def run_simulation(sim_start):
    accidents = []
    os.makedirs(os.path.dirname(SUMO_FCD_FILE), exist_ok=True)
    traci.start(
        [
            SUMO_BINARY,
            "-c",
            SUMO_CONFIG_FILE,
            "--fcd-output",
            SUMO_FCD_FILE,
            "--fcd-output.geo",
        ]
    )
    try:
        step = 0
        while step < SUMO_END_TIME:
            traci.simulationStep()
            if random.random() < SUMO_ACCIDENT_PROB:
                vehicles = traci.vehicle.getIDList()
                if vehicles:
                    vehicle_id = random.choice(vehicles)
                    try:
                        time_remaining = SUMO_END_TIME - step
                        max_duration = min(time_remaining, SUMO_MAX_ACCIDENT_DURATION)
                        duration = random.randint(
                            min(SUMO_MIN_ACCIDENT_DURATION, max_duration), max_duration
                        )
                        traci.vehicle.setSpeed(vehicle_id, 0)
                        sumo_edge_id = traci.vehicle.getRoadID(vehicle_id)
                        traci.vehicle.setStop(
                            vehicle_id,
                            edgeID=sumo_edge_id,
                            pos=traci.vehicle.getLanePosition(vehicle_id),
                            duration=duration,
                        )
                        accident_time = sim_start + pd.Timedelta(seconds=step)
                        accident_end_time = accident_time + pd.Timedelta(
                            seconds=duration
                        )
                        lon, lat = traci.vehicle.getPosition(vehicle_id)
                        lon, lat = traci.simulation.convertGeo(lon, lat)
                        edge_id = map_to_valhalla_edge(lat, lon)
                        accidents.append(
                            {
                                "vehicle_id": vehicle_id,
                                "start_time": accident_time,
                                "end_time": accident_end_time,
                                "duration": duration,
                                "latitude": lat,
                                "longitude": lon,
                                "edge_id": edge_id,
                                "sumo_edge_id": sumo_edge_id,
                            }
                        )
                    except traci.exceptions.TraCIException:
                        pass
            step += 1
    finally:
        traci.close()
    if accidents:
        accidents_df = pd.DataFrame(accidents)
        print("First few rows of Accidents DataFrame before writing to Parquet:")
        print(accidents_df.head())
        accidents_df.to_parquet(
            SUMO_ACCIDENTS_FILE,
            engine="pyarrow",
            schema=SUMO_ACCIDENTS_SCHEMA,
            index=False,
        )


def calculate_sim_start():
    ts_start_dt = pd.to_datetime(TS_START)
    ts_end_dt = pd.to_datetime(TS_END)
    days_diff = (ts_end_dt - ts_start_dt).days
    random_day = random.randint(0, days_diff)
    sim_start = ts_start_dt + pd.Timedelta(days=random_day)
    sim_start = sim_start + pd.Timedelta(hours=SUMO_STARTING_HOUR)
    return sim_start


def simulate_parse_accidents(
    sim_start, period=1, end_time=SUMO_END_TIME, fringe_factor=5
):
    generate_random_trips(
        period=period, fringe_factor=fringe_factor, end_time=end_time, max_retries=10
    )
    update_sumo_config()
    run_simulation(sim_start)
