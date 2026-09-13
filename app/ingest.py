"""Compatibility entry point; ingestion code lives in app.ingestion."""

from app.ingestion.ingest import *  # noqa: F403

if __name__ == "__main__":
    main()
