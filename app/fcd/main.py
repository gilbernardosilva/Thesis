import logging
import os

import app.fcd.config.logger
from app.fcd.config.utils import export_n_first_rows_to_csv, print_n_first_rows
from app.fcd.config.variables import (
    CLEAN_DATA_FILE,
    FEATURES_DATA_FILE,
    FILTERED_DATA_FILE,
    MAPPED_DATA_FILE,
)
from app.fcd.data_processing import load_and_clean_data
from app.fcd.feature_engineering import feature_engineering
from app.fcd.map_matching import map_matching
from app.fcd.model import main as model_main
from app.fcd.regional_spatial_filter import load_filter_region


def main():
    """Main pipeline to process FCD data and predict congestion."""

    logging.info("Pipeline started.")

    try:

        load_and_clean_data()
        print_n_first_rows(CLEAN_DATA_FILE)

        load_filter_region()
        print_n_first_rows(FILTERED_DATA_FILE)

        map_matching()
        print_n_first_rows(MAPPED_DATA_FILE)

        feature_engineering()
        print_n_first_rows(FEATURES_DATA_FILE)
        export_n_first_rows_to_csv(
            FEATURES_DATA_FILE, "output/data/features_data.csv", 1000
        )

        model_main()
    except Exception as e:
        logging.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
        raise

    logging.info("Pipeline finished.")


if __name__ == "__main__":
    main()
