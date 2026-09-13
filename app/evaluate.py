"""Compatibility entry point; CLI commands live in app.cli."""

from app.cli.evaluate import *  # noqa: F403

if __name__ == "__main__":
    main()
