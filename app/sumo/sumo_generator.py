import os
import shutil
import subprocess
import logging
import argparse
import time
import json
from functools import partial
from multiprocessing import Pool
import xml.etree.ElementTree as ET
from datetime import datetime


def setup_logging(base_path, date_folder):
    output_dir = os.path.join(base_path, "output", date_folder, "SUMO")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "sumo_generator.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def clear_output_folder(folder_path, logger):
    folder_path = os.path.abspath(folder_path)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    logger.info(f"Cleared and recreated: {folder_path}")


def generate_random_trips(
    name,
    network_file,
    output_prefix,
    logger,
    begin=0,
    end=3600,
    period=1,
    fringe_factor=1,
    seed=42,
):
    output_file = f"{output_prefix}_{name}.trips.xml"
    sumo_home = os.environ.get("SUMO_HOME", r"C:\Users\Gil\Downloads\sumo-1.22.0")
    sumo_tools = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.exists(sumo_tools):
        logger.error(f"randomTrips.py not found at {sumo_tools}. Check SUMO_HOME.")
        raise FileNotFoundError(f"randomTrips.py not found at {sumo_tools}")

    network_file = os.path.abspath(network_file)
    if not os.path.exists(network_file):
        raise FileNotFoundError(f"Network file not found: {network_file}")

    command = [
        "python",
        sumo_tools,
        "-n",
        network_file,
        "-o",
        output_file,
        "--begin",
        str(begin),
        "--end",
        str(end),
        "--period",
        str(period),
        "--route-file",
        f"{output_prefix}_{name}.rou.xml",
        "--validate",
        "--fringe-factor",
        str(fringe_factor),
        "--random",
        "--seed",
        str(seed),
        "--vehicle-class",
        "passenger",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Trips generated: {output_file}")
        logger.debug(f"randomTrips stdout: {result.stdout}")
        logger.debug(f"randomTrips stderr: {result.stderr}")
        trips_size = os.path.getsize(output_file)
        if trips_size < 500:
            logger.error(f"Trips file {output_file} too small: {trips_size} bytes")
            return None, None
        return os.path.abspath(output_file), os.path.abspath(
            f"{output_prefix}_{name}.rou.xml"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating trips for {name}: {e.stderr}")
        return None, None


def generate_fcd(
    route_file,
    config_file,
    output_prefix,
    network_file,
    logger,
):
    if not route_file or not os.path.exists(route_file):
        logger.error(f"Route file not provided or does not exist: {route_file}")
        return None
    config_file = os.path.abspath(config_file)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    scenario = os.path.basename(route_file).split("_")[1].split(".")[0]
    output_file = f"{output_prefix}_{scenario}.xml"
    sumo_home = os.environ.get("SUMO_HOME", r"C:\Users\Gil\Downloads\sumo-1.22.0")
    sumo_exe = os.path.join(sumo_home, "bin", "sumo.exe")
    if not os.path.exists(sumo_exe):
        logger.error(f"sumo.exe not found at {sumo_exe}. Check SUMO_HOME.")
        raise FileNotFoundError(f"sumo.exe not found at {sumo_exe}")

    additional_file = f"{output_prefix}_{scenario}_additional.xml"
    with open(additional_file, "w") as f:
        f.write("<additional></additional>")

    command = [
        sumo_exe,
        "-c",
        config_file,
        "--route-files",
        route_file,
        "--fcd-output",
        output_file,
        "--fcd-output.geo",
        "--random",
        "--verbose",
        "--additional-files",
        additional_file,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        logger.info(f"FCD generated: {output_file}")
        logger.debug(f"SUMO stdout: {result.stdout}")
        logger.debug(f"SUMO stderr: {result.stderr}")
        if result.returncode != 0:
            logger.warning(
                f"SUMO returned non-zero exit code {result.returncode}: {result.stderr}"
            )
        timeout = 10
        elapsed = 0
        while (
            not os.path.exists(output_file)
            or os.path.getsize(output_file) == 0
            and elapsed < timeout
        ):
            time.sleep(1)
            elapsed += 1
        if not os.path.exists(output_file):
            logger.error(f"FCD file not created: {output_file}")
            return None
        fcd_size = os.path.getsize(output_file)
        if fcd_size < 5000:
            logger.error(f"FCD file {output_file} too small: {fcd_size} bytes")
            return None
        return os.path.abspath(output_file)
    except Exception as e:
        logger.error(f"Unexpected error generating FCD for {scenario}: {str(e)}")
        return None


def generate_scenario(
    scenario_params,
    network_file,
    trips_prefix,
    config_file,
    fcd_prefix,
    base_path,
    date_folder,
):
    logger = setup_logging(base_path, date_folder)
    scenario, params = scenario_params
    attempt = 0

    while True:
        attempt += 1
        logger.info(
            f"Generating {scenario} (Attempt {attempt}): "
            f"period={params['period']}, begin={params['begin']}, end={params['end']}, fringe_factor={params['fringe_factor']}"
        )

        trips_file, route_file = generate_random_trips(
            scenario,
            network_file,
            trips_prefix,
            logger,
            begin=params["begin"],
            end=params["end"],
            period=params["period"],
            fringe_factor=params["fringe_factor"],
        )
        if not route_file:
            logger.warning(f"Failed to generate route file for {scenario}. Retrying...")
            time.sleep(2)
            continue

        fcd_file = generate_fcd(
            route_file,
            config_file,
            fcd_prefix,
            network_file,
            logger,
        )
        if not fcd_file:
            logger.warning(f"Failed to generate FCD file for {scenario}. Retrying...")
            time.sleep(2)
            continue

        logger.info(f"Successfully generated {scenario} after {attempt} attempts.")
        return fcd_file


def generate_sumo_files(args, logger, date_folder):
    base_path = os.path.abspath(args.base_path)
    input_dir = os.path.join(base_path, "input", "sumo_config")
    network_file = os.path.join(input_dir, "osm.net.xml")
    config_file = os.path.join(input_dir, "osm.sumocfg")
    # Ajusta os caminhos para dentro de SUMO
    trips_prefix = os.path.join(
        base_path, "output", date_folder, "SUMO", "trips", "trips"
    )
    fcd_prefix = os.path.join(base_path, "output", date_folder, "SUMO", "fcd", "fcd")
    params_file = os.path.join(
        base_path, "output", date_folder, "SUMO", "scenario_params.json"
    )

    default_sumo_home = r"C:\Users\Gil\Downloads\sumo-1.22.0"
    if "SUMO_HOME" not in os.environ:
        logger.warning(f"SUMO_HOME not set. Setting to default: {default_sumo_home}")
        os.environ["SUMO_HOME"] = default_sumo_home
    else:
        logger.info(f"SUMO_HOME already set: {os.environ['SUMO_HOME']}")

    logger.info(
        f"Starting SUMO file generation with base_path: {base_path} and date_folder: {date_folder}"
    )

    if not os.path.exists(network_file):
        logger.error(f"Network file not found: {network_file}")
        return []
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        return []

    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    clear_output_folder(os.path.dirname(trips_prefix), logger)  # Limpa a pasta trips
    clear_output_folder(os.path.dirname(fcd_prefix), logger)  # Limpa a pasta fcd

    scenario_settings = {
        "scenario1": {"period": 1.0, "begin": 0, "end": 5000, "fringe_factor": 1.0},
        "scenario2": {"period": 0.8, "begin": 0, "end": 3600, "fringe_factor": 1.0},
        "scenario3": {"period": 1.2, "begin": 0, "end": 7000, "fringe_factor": 1.0},
        "scenario4": {"period": 1.0, "begin": 0, "end": 5000, "fringe_factor": 1.0},
        "scenario5": {"period": 0.8, "begin": 0, "end": 3600, "fringe_factor": 1.0},
        "scenario6": {"period": 1.2, "begin": 0, "end": 7000, "fringe_factor": 1.0},
    }

    with open(params_file, "w") as f:
        json.dump(scenario_settings, f, indent=4)
    logger.info(f"Scenario parameters saved to {params_file}")

    num_cores = 7
    logger.info(f"Using {num_cores} cores for parallel processing")
    with Pool(processes=num_cores) as pool:
        fcd_files = pool.map(
            partial(
                generate_scenario,
                network_file=network_file,
                trips_prefix=trips_prefix,
                config_file=config_file,
                fcd_prefix=fcd_prefix,
                base_path=base_path,
                date_folder=date_folder,
            ),
            scenario_settings.items(),
        )
    fcd_files = [f for f in fcd_files if f]

    valid_fcd_files = []
    for fcd_file in fcd_files:
        if not os.path.exists(fcd_file):
            logger.error(f"FCD file not generated: {fcd_file}")
        elif os.path.getsize(fcd_file) == 0:
            logger.error(f"FCD file is empty: {fcd_file}")
        else:
            logger.info(
                f"FCD file verified: {fcd_file}, size: {os.path.getsize(fcd_file)} bytes"
            )
            valid_fcd_files.append(fcd_file)

    if len(valid_fcd_files) != 6:
        logger.error(
            f"Expected 6 valid FCD files, got {len(valid_fcd_files)}. Process failed."
        )
        return []
    logger.info(
        f"Successfully generated {len(valid_fcd_files)} FCD files: {valid_fcd_files}"
    )
    return valid_fcd_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO scenario file generator")
    parser.add_argument("--base-path", default=os.getcwd(), help="Base directory path")
    args = parser.parse_args()
    date_folder = datetime.now().strftime("%Y-%m-%d")
    logger = setup_logging(args.base_path, date_folder)
    fcd_files = generate_sumo_files(args, logger, date_folder)
