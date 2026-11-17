"""Smoke test on a hand-made 60-frame clip; no match data is needed."""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from algorithms.dead_ball_aware.predict import predict  # noqa: E402
from observations.schema import COLUMNS, validate  # noqa: E402

FRAMES = 60
LENGTH, WIDTH = 105.0, 68.0


def make_clip():
    """Home slot 1 holds the ball, it rolls to away slot 25, which then holds it."""
    obs = pd.DataFrame(np.nan, index=range(FRAMES), columns=COLUMNS)
    obs['frame'] = range(FRAMES)
    obs[['1_x', '1_y']] = [40 / LENGTH, 34 / WIDTH]
    obs[['25_x', '25_y']] = [48 / LENGTH, 34 / WIDTH]
    obs['ball_y'] = 34 / WIDTH
    obs.loc[:19, 'ball_x'] = 40 / LENGTH
    obs.loc[20:29, 'ball_x'] = np.linspace(40, 48, 10) / LENGTH
    obs.loc[30:, 'ball_x'] = 48 / LENGTH
    meta = {'fps': 25, 'pitch_length': LENGTH, 'pitch_width': WIDTH,
            'slot_teams': {str(i): 'H' if i <= 23 else 'A' for i in range(1, 47)}}
    return obs, meta


class SmokeTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, 'config', 'whc.yaml')) as handle:
            self.params = yaml.safe_load(handle)
        self.obs, self.meta = make_clip()
        validate(self.obs)

    def check(self, labels):
        self.assertEqual(len(labels), FRAMES)
        self.assertTrue(set(labels) <= {'H', 'A', 'D'}, set(labels))

    def test_with_and_without_tags(self):
        self.check(list(predict(self.obs, self.meta, self.params, tags=[])))
        self.check(list(predict(self.obs, self.meta, self.params)))

    def test_with_a_set_piece_tag(self):
        tag = {'start_frame': 10, 'end_frame': 40, 'code': 'SET PIECES',
               'type': 'Free Kick', 'team': 'H'}
        self.check(list(predict(self.obs, self.meta, self.params, tags=[tag])))

    def test_ball_out_of_pitch_is_dead(self):
        self.obs.loc[:, 'ball_x'] = 110 / LENGTH
        labels = list(predict(self.obs, self.meta, self.params, tags=[]))
        self.check(labels)
        self.assertEqual(labels, ['D'] * FRAMES)


if __name__ == '__main__':
    unittest.main()
