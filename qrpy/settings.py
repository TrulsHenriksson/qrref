from qrpy.custom_types import *
from dataclasses import dataclass


__all__ = ["SETTINGS"]


@dataclass
class Settings:
    byte_encoding: Literal["latin-1", "utf-8"] = "latin-1"


SETTINGS = Settings()
