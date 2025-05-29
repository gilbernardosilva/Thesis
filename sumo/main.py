import logging

from app.fcd.config.utils import export_n_first_rows_to_csv, print_n_first_rows
from app.fcd.config.variables import (
    SUMO_ACCIDENTS_FILE,
    SUMO_END_TIME,
    SUMO_FCD_PARSED_FILE,
    SUMO_MATCHED_FILE,
)
from sumo.sumo_map_matching import map_match_fcd
from sumo.sumo_parser import process_fcd
from sumo.sumo_simulate import calculate_sim_start, simulate_parse_accidents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "sumo/output/simulation.log",
        ),
        logging.StreamHandler(),
    ],
)


def main():
    """Orchestrate the SUMO pipeline: simulation, parsing, and map matching."""
    try:
        logging.info("Starting SUMO pipeline...")

        logging.info("Running simulation and generating FCD file...")
        sim_start = calculate_sim_start()
        simulate_parse_accidents(
            sim_start,
            period=1,
            end_time=SUMO_END_TIME,
            fringe_factor=5,
        )

        export_n_first_rows_to_csv(
            SUMO_ACCIDENTS_FILE, "sumo/output/sumo_accidents.csv", 1000
        )
        logging.info("Parsing FCD file...")
        process_fcd(sim_start=sim_start)
        export_n_first_rows_to_csv(
            SUMO_FCD_PARSED_FILE, "sumo/output/sumo_fcd_parsed.csv", 1000
        )

        logging.info("Applying map matching to parsed FCD data...")
        map_match_fcd()

        export_n_first_rows_to_csv(
            SUMO_MATCHED_FILE, "sumo/output/sumo_matched.csv", 1000
        )

        logging.info("SUMO pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Error in pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()
