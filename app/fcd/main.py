import logging
import os

import app.fcd.config.logger
from app.fcd.config.utils import print_n_first_rows
from app.fcd.config.variables import (
    CLEAN_DATA_FILE,
    FEATURES_DATA_FILE,
    FILTERED_DATA_FILE,
    MAPPED_DATA_FILE,
)
from app.fcd.data_parser import load_and_clean_data
from app.fcd.feature_engineering import feature_engineering
from app.fcd.filter_region import load_filter_region
from app.fcd.segment_mapping import map_matching


def main():
    """Main pipeline to process FCD data and predict congestion."""

    logging.info("Pipeline started.")

    try:

        # load_and_clean_data()

        # print_n_first_rows(CLEAN_DATA_FILE)

        # load_filter_region()
        # print_n_first_rows(FILTERED_DATA_FILE)

        # map_matching()

        # print_n_first_rows(MAPPED_DATA_FILE)

        # feature_engineering()
        print_n_first_rows(FEATURES_DATA_FILE)

    except Exception as e:
        logging.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
        raise

    logging.info("Pipeline finished.")


if __name__ == "__main__":
    main()
