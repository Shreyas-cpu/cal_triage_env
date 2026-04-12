"""Compatibility shim that exposes the root FastAPI app."""

from app import app


def main() -> None:
    from app import main as root_main

    root_main()


if __name__ == "__main__":
    main()
