# custom_slog_upd.py

from contextlib import contextmanager

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default global log level (index). "INFO" by default; change if you like.
GLOB_LOG_LEVEL = LOG_LEVELS.index("INFO")


def _resolve_level(level):
    """
    Accepts either a string level name or an integer index.
    Returns an integer index into LOG_LEVELS, clamped to valid range.
    """
    if isinstance(level, str):
        lvl = level.strip().upper()
        if lvl not in LOG_LEVELS:
            raise ValueError(f"Unknown log level '{level}'. Valid: {', '.join(LOG_LEVELS)}")
        return LOG_LEVELS.index(lvl)
    # allow ints directly (0..len-1)
    if isinstance(level, int):
        return max(0, min(level, len(LOG_LEVELS) - 1))
    raise TypeError("log level must be a string name or an int index")


def set_glob_loglevel(level):
    """
    Set the module-global log level (affects print_cust decisions).
    Example: set_glob_loglevel('WARNING') or set_glob_loglevel(2)
    """
    global GLOB_LOG_LEVEL
    GLOB_LOG_LEVEL = _resolve_level(level)
    print(f"set_glob_loglevel, GLOB_LOG_LEVEL: {GLOB_LOG_LEVEL}")


def get_glob_loglevel():
    """Return the current global log level name (string)."""
    return LOG_LEVELS[GLOB_LOG_LEVEL]


def print_cust(*args, level="DEBUG"):
    """
    Simple, custom printing function that takes a log level.

    Parameters:
        *args: values passed to print()
        level (str|int): the message level (string like 'INFO' or int index)

    Prints only if message level >= global level.
    """
    int_level = _resolve_level(level)
    if int_level >= GLOB_LOG_LEVEL:
        print(*args)


@contextmanager
def temporary_log_level(level):
    """
    Context manager to temporarily override the global log level.

    Example:
        with temporary_log_level("DEBUG"):
            print_cust("noisy detail", level="DEBUG")
    """
    global GLOB_LOG_LEVEL
    old = GLOB_LOG_LEVEL
    try:
        GLOB_LOG_LEVEL = _resolve_level(level)
        yield
    finally:
        GLOB_LOG_LEVEL = old
