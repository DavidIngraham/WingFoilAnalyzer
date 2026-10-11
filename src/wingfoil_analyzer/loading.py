"""Load FIT activities and normalize their geographic samples."""

import io
import pathlib
import zipfile
from os import PathLike
from typing import TypeAlias

import fitdecode
import numpy as np

from .types import FloatArray, RecordList, Session

InputPath: TypeAlias = str | PathLike[str]


def load(path: InputPath) -> tuple[RecordList, FloatArray, FloatArray, FloatArray, Session]:
    """Load a bare FIT file or a ZIP archive containing a FIT file."""
    raw = pathlib.Path(path).read_bytes()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            fit_name = next(
                name for name in archive.namelist() if name.lower().endswith(".fit")
            )
            raw = archive.read(fit_name)

    rows = []
    session = {}
    with fitdecode.FitReader(io.BytesIO(raw)) as fit:
        for message in fit:
            if not isinstance(message, fitdecode.FitDataMessage):
                continue
            data = {field.name: field.value for field in message.fields}
            if message.name == "record":
                rows.append(data)
            elif message.name == "session":
                session = data

    timestamps = np.array(
        [record["timestamp"].timestamp() for record in rows], dtype=float
    )
    latitude = (
        np.array([record.get("position_lat") for record in rows], dtype=float)
        * np.pi
        / 2**31
    )
    longitude = (
        np.array([record.get("position_long") for record in rows], dtype=float)
        * np.pi
        / 2**31
    )
    first = np.flatnonzero(np.isfinite(latitude) & np.isfinite(longitude))[0]
    positions = np.column_stack(
        (
            (longitude - longitude[first]) * 6_371_000 * np.cos(np.nanmean(latitude)),
            (latitude - latitude[first]) * 6_371_000,
        )
    )
    speed = np.array(
        [record.get("enhanced_speed", record.get("speed", np.nan)) for record in rows],
        dtype=float,
    )
    return rows, timestamps, positions, speed, session
