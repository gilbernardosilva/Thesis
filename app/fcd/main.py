import logging

import app.fcd.config.logger
from app.fcd.config.utils import print_n_random_rows
from app.fcd.data_parser import load_and_clean_data
from app.fcd.feature_engineering import feature_engineer
from app.fcd.filter_region import load_filter_region
from app.fcd.segment_mapping import map_matching


def main():
    """Main pipeline to process FCD data and predict congestion."""

    logging.info("Pipeline started.")

    try:
        # load_and_clean_data()

        # print_n_random_rows(
        #     "/Users/gilsilva/Work/thesis/output/dataframes/cleaned_df.parquet"
        # )

        # load_filter_region()
        # print_n_random_rows(
        #     "/Users/gilsilva/Work/thesis/output/dataframes/filtered_df.parquet"
        # )

        map_matching()
        print_n_random_rows(
            "/Users/gilsilva/Work/thesis/output/dataframes/mapped_df.parquet"
        )

        # feature_engineer()
        # print_first_20_rows(
        #     "/Users/gilsilva/Work/thesis/output/dataframes/enriched_df.parquet"
        # )

    except Exception as e:
        logging.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        raise

    logging.info("Pipeline finished.")
    print("Pipeline finished")


if __name__ == "__main__":
    main()
