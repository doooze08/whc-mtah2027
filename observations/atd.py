"""Load tactical ATD through kloppy, without opening labels or other match assets.

    python -m observations.atd [--data-dir /path/to/data]
"""

import argparse
import json

import numpy as np

from games import GAME, require, source_frames
from observations.schema import COLUMNS, SLOTS, validate


def from_kloppy(dataset, expected_frames):
    """Adapt an untransformed Metrica TrackingDataset to (observations, metadata)."""
    from kloppy.domain import Ground

    pitch = dataset.metadata.pitch_dimensions
    slot_teams, rename = {}, {}
    labels = {Ground.HOME: "H", Ground.AWAY: "A"}
    for team in dataset.metadata.teams:
        label = labels[team.ground]
        for player in team.players:
            # In this export ShirtNumber is the track slot, never a player identity.
            slot = str(player.jersey_no)
            slot_teams[slot] = label
            for field in ("x", "y", "s"):
                rename["{}_{}".format(player.player_id, field)] = slot + "_" + field
    # Select only public observation fields; discard timestamps, periods, ball_z, etc.
    raw = dataset.to_df(engine="pandas")
    if len(raw) != expected_frames or not np.array_equal(
            raw["frame_id"].to_numpy(), np.arange(1, expected_frames + 1)):
        raise ValueError("ATD must cover frame_id 1..{} exactly in order".format(expected_frames))
    raw = raw.rename(columns=rename)
    observations = raw[COLUMNS[1:]].astype(float)
    observations.insert(0, "frame", raw["frame_id"].to_numpy(dtype=np.int64) - 1)
    metadata = {"slot_teams": slot_teams, "pitch_length": pitch.pitch_length,
                "pitch_width": pitch.pitch_width, "fps": dataset.metadata.frame_rate}
    validate(observations)
    return observations, metadata


def load_tactical(data_dir=None):
    """Load the complete tactical export using official path and frame-count helpers."""
    from kloppy import metrica

    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir, source="tactical")
    dataset = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)
    return from_kloppy(dataset, expected_frames=source_frames("tactical", GAME))


def report(observations, metadata):
    """Report bounds, including out-of-pitch values, instead of silently clipping them."""
    coordinates = observations[[c for c in COLUMNS if c.endswith(("_x", "_y"))]]
    speeds = observations[[slot + "_s" for slot in SLOTS]]

    def bounds(table):
        values = table.to_numpy()
        finite = values[np.isfinite(values)]
        return [float(finite.min()), float(finite.max())] if finite.size else [None, None]

    return {"rows": len(observations), "columns": list(observations.columns),
            "column_count": len(observations.columns),
            "frame_range": [int(observations.frame.min()), int(observations.frame.max())],
            "coordinate_range": bounds(coordinates), "speed_range_m_s": bounds(speeds),
            "coordinate_values_outside_pitch": int(((coordinates < 0) | (coordinates > 1)).sum().sum()),
            "ball_missing_frames": int(observations[["ball_x", "ball_y"]].isna().any(axis=1).sum()),
            "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    print(json.dumps(report(*load_tactical(args.data_dir)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
