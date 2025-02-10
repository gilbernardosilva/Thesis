from sqlalchemy.orm import Session

from app.core.db.db_utils import (
    delete_table,
    get_table_names,
    import_probes_from_file,
    print_probes,
)
from app.core.db.init_db import create_tables
from app.core.db.session import SessionLocal


def test(session: Session) -> None:
    delete_table(session, "probes")
    create_tables()
    table_names = get_table_names(session)
    print("Tabelas no banco de dados:", table_names)
    import_probes_from_file(session, "probes/2018-11-25")
    print_probes(session)


def main() -> None:
    with SessionLocal() as session:
        #test(session)
        


if __name__ == "__main__":
    main()
