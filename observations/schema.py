"""One frame per row; pitch-normalized coordinates and player speed in m/s."""

import numpy as np

SLOTS = tuple(str(i) for i in range(1, 47))
COLUMNS = ["frame", "ball_x", "ball_y"] + [
    "{}_{}".format(slot, field) for slot in SLOTS for field in ("x", "y", "s")]


def validate(observations):
    """Check column order and contiguous zero-based frames without altering the data."""
    if list(observations.columns) != COLUMNS:
        raise ValueError("Expected exactly the 141 shared observation columns in order")
    if not np.array_equal(observations["frame"].to_numpy(), np.arange(len(observations))):
        raise ValueError("Frames must be ordered, unique and contiguous from 0")
