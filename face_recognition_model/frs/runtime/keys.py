QUIT_KEYS = {ord("q"), ord("Q")}
SNAPSHOT_KEYS = {ord("s"), ord("S")}
RELOAD_KEYS = {ord("r"), ord("R")}
CAPTURE_KEY = ord(" ")


def is_quit_key(key: int) -> bool:
    return key in QUIT_KEYS


def is_snapshot_key(key: int) -> bool:
    return key in SNAPSHOT_KEYS


def is_reload_key(key: int) -> bool:
    return key in RELOAD_KEYS
