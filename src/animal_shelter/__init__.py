"""Animal Shelter package for outcome prediction."""

import logging


def setup_logger(*, level: int = logging.INFO) -> None:
    """Set up the logger for the application.

    Args:
    ----
        level (int): The logging level to set. Defaults to
            `logging.INFO`.

    """
    logger = logging.getLogger("animal_shelter")

    # Avoid adding duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def set_log_level(level: int | str) -> None:
    """Set the logging level for the animal_shelter logger.

    Args:
    ----
        level (int | str): The logging level to set. Can be an integer
            (e.g., logging.DEBUG) or a string (e.g., "DEBUG").

    Raises:
    ------
        ValueError: If the level string is not a valid logging level.

    """
    logger = logging.getLogger("animal_shelter")

    # Convert string to logging level if necessary
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        level = numeric_level

    # Set level for logger and all handlers
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# Initialize logger when module is imported (default to INFO level)
setup_logger(level=logging.INFO)


def main() -> None:
    """Entry point for the animal-shelter package."""
    print("Hello from animal-shelter!")
