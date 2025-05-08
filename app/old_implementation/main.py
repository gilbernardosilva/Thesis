from app.db.db_utils import delete_table
from app.db.init_db import create_tables
from app.db.session import SessionLocal
from app.services.moving_pandas.mp import (
    test_matched,
    test_matched_address,
    test_matched_address_sumo,
    test_raw,
)
from app.test.sumo import process_sumo_file


def main() -> None:
    with SessionLocal() as session:
        # batch_size = 3

        # identifier = "88b518ff-7111-4ccd-bea4-6c1ea2e3cab9"
        # identifier2 = "e423e6cd-e317-4390-b190-411658019aeb"
        # identifier3 = "veh0"
        # test_raw(session, identifier2)
        # test_matched_address(session, identifier2, batch_size)
        # test_matched(session, identifier2, batch_size)
        # Criando array com veh0, veh1, veh2, veh3
        # veh_identifiers = ["veh0", "veh1", "veh2", "veh3"]

        # # Fazendo chamadas ao test_matched_address_sumo para cada identificador no array
        # for veh_id in veh_identifiers:
        #     test_matched_address_sumo(veh_id, batch_size)
        delete_table(session, "probes_test")
        delete_table(session, "ruas")
        create_tables()
        # Chamada da função para processar um arquivo CSV do SUMO
        process_sumo_file("/Users/gilsilva/Work/thesis/input/sumo/fcd.csv")


if __name__ == "__main__":
    main()
