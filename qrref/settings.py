from qrref.custom_types import *
from dataclasses import dataclass
from contextlib import contextmanager


__all__ = ["SETTINGS", "temporary_settings"]


@dataclass
class Settings:
    byte_encoding: Literal["latin-1", "utf-8"] = "latin-1"


SETTINGS = Settings()


@contextmanager
def temporary_settings(**kwargs):
    # Copy the initial settings
    settings_before = dict(SETTINGS.__dict__)
    try:
        for key, value in kwargs.items():
            setattr(SETTINGS, key, value)
        yield SETTINGS
    finally:
        for key, value in settings_before.items():
            setattr(SETTINGS, key, value)
