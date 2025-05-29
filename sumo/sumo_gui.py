import logging
import os
from datetime import timedelta

import pandas as pd
import traci

from app.fcd.config.variables import (
    SUMO_ACCIDENTS_FILE,
    SUMO_ACCIDENTS_PREDICTION_FILE,
    SUMO_CONFIG_FILE,
    SUMO_HOME,
    SUMO_NET_FILE,
    SUMO_ROUTES_FILE,
    SUMO_TRIPS_FILE,
    TS_START,
)

SUMO_GUI_BINARY = os.path.join(SUMO_HOME, "bin", "sumo-gui")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sumo/output/visualization.log", mode="w"),
        logging.StreamHandler(),
    ],
)


def update_sumo_config():
    import xml.etree.ElementTree as ET

    tree = ET.parse(SUMO_CONFIG_FILE)
    root = tree.getroot()
    input_elem = root.find("input")
    net_file_elem = input_elem.find("net-file")
    route_files_elem = input_elem.find("route-files")
    net_file_elem.set("value", os.path.abspath(SUMO_NET_FILE))
    route_files_elem.set("value", os.path.abspath(SUMO_TRIPS_FILE))
    tree.write(SUMO_CONFIG_FILE)


def visualize_accidents():
    # Load accidents and predictions
    accidents_df = pd.read_parquet(SUMO_ACCIDENTS_FILE)
    predictions_df = pd.read_csv(SUMO_ACCIDENTS_PREDICTION_FILE)

    # Filter accidents with predicted_congestion different from "unknown"
    filtered_predictions = predictions_df[
        predictions_df["predicted_congestion"] != "unknown"
    ]

    if filtered_predictions.empty:
        logging.info(
            "No accidents with valid predictions (not 'unknown') to visualize."
        )
        return

    # Iterate over filtered accidents
    for _, prediction in filtered_predictions.iterrows():
        idx = prediction["accident_index"]
        if idx >= len(accidents_df):
            logging.warning(
                f"Accident index {idx} not found in accidents data. Skipping."
            )
            continue

        accident = accidents_df.iloc[idx]
        vehicle_id = accident["vehicle_id"]
        start_time = pd.to_datetime(accident["start_time"])
        end_time = pd.to_datetime(accident["end_time"])
        duration = int(accident["duration"])
        sumo_edge_id = accident["sumo_edge_id"]
        predicted_congestion = prediction["predicted_congestion"]

        logging.info(
            f"Visualizing accident {idx} with predicted congestion: {predicted_congestion}"
        )

        sim_start = pd.to_datetime(TS_START)
        start_step = int((start_time - sim_start).total_seconds())
        end_step = int((end_time - sim_start).total_seconds())
        visualization_start_step = max(0, start_step - 60)

        update_sumo_config()

        traci.start(
            [
                SUMO_GUI_BINARY,
                "-c",
                SUMO_CONFIG_FILE,
                "--start",
                "--delay",
                "100",
                "--demo",
            ]
        )

        try:

            step = 0
            while step <= end_step:
                if step >= visualization_start_step:
                    traci.simulationStep()
                else:
                    traci.simulationStep()
                    step += 1
                    continue

                if step == start_step:
                    vehicles = traci.vehicle.getIDList()
                    if vehicle_id in vehicles:
                        traci.vehicle.setStop(
                            vehicle_id,
                            edgeID=sumo_edge_id,
                            pos=traci.vehicle.getLanePosition(vehicle_id),
                            duration=duration,
                        )
                        logging.info(
                            f"Accident recreated for vehicle {vehicle_id} on edge {sumo_edge_id} at step {step}"
                        )

                traci.edge.setColor(sumo_edge_id, (0, 255, 0, 255))

                for veh_id in traci.vehicle.getIDList():
                    speed = traci.vehicle.getSpeed(veh_id) * 3.6
                    if speed < 1:
                        traci.vehicle.setColor(veh_id, (255, 0, 0, 255))
                    else:
                        traci.vehicle.setColor(veh_id, (0, 255, 0, 255))

                step += 1

        except traci.exceptions.TraCIException as e:
            logging.info(f"SUMO-GUI closed manually: {e}. Moving to next accident.")
        finally:
            try:
                traci.close()
            except traci.exceptions.TraCIException:
                pass
            logging.info(
                f"Finished visualizing accident {idx}. Moving to next accident."
            )


def main():
    visualize_accidents()


if __name__ == "__main__":
    main()
