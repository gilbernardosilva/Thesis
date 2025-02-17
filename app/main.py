from app.db.session import SessionLocal
from app.services.moving_pandas import test_matched, test_matched_street, test_raw


def main() -> None:
    with SessionLocal() as session:
        identifier = "88b518ff-7111-4ccd-bea4-6c1ea2e3cab9"
        batch_size = 5
        # test_raw(session, identifier)
        test_matched_street(session, identifier, batch_size)


if __name__ == "__main__":
    main()
