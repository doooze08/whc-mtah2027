# WHC — More than a Hack 2027, Phase 1

**Team WHC** · submission `WHC.csv` · 199,319 rows of `frame,label` for Barça–Athletic.

The bar for this phase is simple to state and nasty to hit: give every frame of the tactical video
an `H`, `A` or `D`, and make the dead-ball stretches honest. Most of the difficulty is not in
telling the two teams apart. It's in knowing when the ball is actually out of play, when a
throw-in becomes live, and when a long ball gap is really a stoppage rather than tracking losing
the ball for a moment.

We answer those questions with an offline, rule-based state machine: one pass over the tracking
data, a handful of physical thresholds, no training, no randomness. It reads only the ATD
positions and the Smart Tagging `SET PIECES` intervals, and it emits the same file every time.

## Why rules instead of a model

We would have loved a model. There is no labelled data with the official H/A/D definition anywhere
outside this game, and the labels for this game are off-limits for training. That leaves two
options: fit something on proxy labels and hope the definition transfers, or write down what
possession and a dead ball mean and check those statements directly.

We took the second option, and it turned out to be the nicer engineering problem. Every threshold
in the submission is a physical or definitional choice we can defend in one sentence — 1.3 m of
control distance, 3 m/s of relative ball speed, 0.32 s of sustained control. Nothing was tuned
against the public five minutes or against the provided game. When a rule misfires we can point at
the frame, the number, and the observation that broke it, instead of retraining and hoping.

The whole algorithm is 365 lines of NumPy in `algorithms/dead_ball_aware/predict.py`. It has no
learned parameters, no GPU, no model artifacts and no hidden state between runs.

## What we read, and what we ignored

| Asset | Used | For what |
|---|---|---|
| Tactical ATD (`BAR-ATH_tactical_atd.txt` + `.xml`) | **Yes** | Ball and player plane positions, slot teams, pitch size, 25 fps. Speeds come from position differences. We ignore ball height, the `_s` speed column, player identity and referee signals. |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`) | **Yes, `SET PIECES` only** | A set-piece tag is treated as proof that a stoppage and a restart happened. Coordinates still decide where they begin and end. Phase, transition and goal tags are unused. |
| Official match boundaries and frame alignment | Yes | Through the provided `games.py` and loaders. |
| Tactical / broadcast video | No | We considered whistle detection from player halts and referee gestures. Cost and time won; no video-derived signal made it into the submission. |
| Broadcast tracking, homography | No | Once the tactical ATD was the single timeline, we didn't need a second source. |
| Public 5-minute ground truth | Pipeline check only | Format, frame alignment, evaluation flow, and understanding what the official `D` actually covers. Never used to pick, tune or stop an experiment. |

**Training data: none.** There is no model to train.

External checks use other games and synthetic clips. No labels of any kind were used on the
provided match.

| Dataset | Source and licence | Labels | Amount | Role |
|---|---|---|---|---|
| Our simulator | Written by the team | Comes from the scripted scenario, never from a candidate algorithm | 18 scenario subtypes (throw-ins, header duels, bouncing passes, loose and contested balls, stoppages and restarts), with home/away mirrors and several ball-dropout conditions | Regression only. It encodes our own assumptions, so it can't tell us anything about real-match accuracy. |
| Metrica sample data, Games 1–3 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data); no explicit licence, attribution requested | Frame labels **derived automatically by us** from Metrica's events with the recipe near the end of this document. Vendor event data, not hand-labelling. | 3 matches, ~95 min each, 25 fps | Main external check. Games 1–2 for design, Game 3 held out. |
| SkillCorner open data, match 1925299 | [SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT; data by SkillCorner and PySport | Outside every vendor `phases_of_play` interval = `D`, inside = the phase team | 1 match, 10 fps resampled to 25 fps | Secondary check, run once. |

These weak labels are approximations, not the official definition. We use them to ask coarse
questions — "does this rule flip whole dead-ball stretches to live?" — and not to claim an
accuracy number. The development repository with all experiment records is private and can be
shared with the organisers on request.

## How the state machine works

Two questions per frame: **is the ball in play**, and **if so, who has it**.

### Control means sustained control

A slot becomes a control candidate when the ball is within **1.3 m** of it *and* the ball's speed
relative to that slot stays at or below **3 m/s** for **0.32 s** (8 intervals at 25 fps). Being
nearest to the ball is not enough, and neither is a touch.

- If both teams satisfy this simultaneously, it's a contest. No new control is established.
- Possession changes only when an opponent confirms control. The switch is backdated to the start
  of that control, with a small correction to the frame where the incoming ball's velocity first
  changes.
- Loose balls, flights, deflections and contests keep the previous team. That follows the
  challenge rule: a touch without control doesn't change possession.
- The submission never emits `X`. Frames outside the graded halves go through the same state
  machine and simply aren't scored.

### Dead ball starts from three things

- **Ball out.** The ball centre is beyond a line by more than the ball radius (0.11 m). We use
  plane positions only, so a high ball projected outside the touchline counts as out.
- **A long ball gap.** The ball is missing for more than **5 s**. The whole gap becomes `D`,
  backdated to its start. The provider drops the ball during most stoppages; short gaps keep the
  previous state.
- **A tagged stoppage.** In trusted-tag mode, a `SET PIECES` tag with coordinate support starts
  `D` at an earlier exit or long gap, or at the start of a slow-ball period of at least 0.5 s before
  the restart. A tag without coordinate support doesn't create `D` on its own.

Inside `D`, picking up, carrying, placing or walking with the ball does **not** restart play.

### Getting out of `D` is the interesting part

1. **Observed release.** The ball goes from slow (≤ 3 m/s) to a release speed of at least 4 m/s,
   rises by at least 1.5 m/s within 0.12 s, and separates from a taker within reach. The taker's
   team gets possession. For a throw-in, the restart is the first frame the ball is no longer
   wholly out.
2. **Pass confirmation.** Two different players of the same team confirm control in turn and the
   ball visibly travels between them. Play is backdated to when the ball left the first player.
3. **Moving reappearance after a long gap.** If a gap ends with the ball already moving (first
   speed sample above 3 m/s), no release was observed, and the ball then stays visible and inside
   the pitch with no further gap until a player confirms control within 5 s, we treat the restart
   as having happened while the ball was hidden. `D` ends at the reappearance frame — or, for a
   throw-in, the frame the ball enters the pitch. The team comes from the matched set-piece tag,
   otherwise from the first team to confirm control within 5 s.

   This rule is read directly off the official `D`: a `D` that contains a ball gap ends soon after
   the ball reappears, and on our match the provider usually brings the ball back only after it has
   been played.

### Tags say *that* something happened; coordinates say *when* and *by whom*

- Tags are trusted to report a stoppage. Coordinates decide its start, its end and who restarts.
  Each stoppage takes the first coordinate restart candidate that has exit, gap or slow-ball
  support.
- **Team.** When both teams are within control distance at the restart, the tag's team is used. When
  coordinates clearly show only one team, coordinates win, and the conflict is counted.
- A tag is overruled only when the ball stays visible and fast with both teams contesting
  throughout. On a diagnostic run over the full match, the trusted-tag configuration used 45 of 59
  `SET PIECES` tags. 14 were unmatched, none overruled.

## The code, if you want to poke at it

| Path | What's in it |
|---|---|
| `run.py` | Loads ATD + Smart Tagging, calls the algorithm, writes `frame,label` |
| `config/whc.yaml` | Every submitted parameter, one line each, named after the quantity it measures |
| `algorithms/dead_ball_aware/predict.py` | The state machine, 365 lines of NumPy |
| `observations/atd.py`, `schema.py` | ATD loader and observation schema |
| `evaluation/` | Format validator and possession metrics from the starter kit |
| `tests/` | A smoke test on a hand-made clip, no match data needed |

Things we deliberately got right:

- **`infer()` never touches a file or an answer.** It takes observations, metadata, config and
  tags, and returns labels plus observation-only diagnostics — restarts, confirmed passes, visible
  stoppages, long gaps, moving reappearances. Most of the numbers quoted in this document come
  straight out of that dict.
- **All thresholds live in YAML.** No magic constants inside the loop. A reviewer can change a
  number, rerun, and see exactly what moved.
- **One vectorised pass.** Distances and relative speeds are computed as arrays, and the per-slot
  loops only walk run-lengths. A full match is a handful of large array operations, not a Python
  loop over 130k frames.
- **The diagnostics run in the same pass as the predictions.** We never had a separate probe script
  drifting away from the submitted code.

## How we actually got here

1. **Simulation first.** Scripted scenarios with known answers, built to check that each rule does
   what it claims. Regression evidence only — the simulator shares our blind spots.
2. **External weak labels.** Metrica and SkillCorner were converted to the same input format as our
   match, with automatically derived frame labels. We designed on Metrica Games 1–2 and looked at
   Game 3 once, after the definitions were frozen.
3. **Answer-free diagnostics on the provided match.** Count how many segments a rule would flip,
   and what they look like: ball state, collective player speed, nearby tag types. Go/no-go
   thresholds were fixed before each run. No answers were read.

### Dead ends we're not sorry about

| Idea | Why it died |
|---|---|
| Plain sustained-control state machine, no dead-ball layer | Every long stoppage came out `H`/`A`. Largest error source in the first version, and the reason `dead_ball_aware` exists. |
| Collective player speed as a veto (a gap is `D` only if players slow down), and as restart confirmation | Median player speed separates dead from live frames well per-frame (AUC ~0.8–0.9 on external data). But it gave no stable gain on held-out data and broke quick restarts. |
| Ignoring very short exits as tracking error, and absorbing possessions under 1 s | On Metrica one real exit that bounced back cost 33 s. 11 of 12 absorbed short possessions were real. |
| Overruling `D` in long gaps that look live (players moving, ball reappearing far away) | At most 9 gaps (67 s) on our match look like that, and 4 of them precede a set piece. Too small a gain for the risk. |
| A learned classifier | No training labels with the official definition. |

## Known limits

- **Whistle stoppages with the ball visible and no tag** — fouls, offsides not restarted from an
  exit or a gap — stay labelled live. Coordinates carry almost no evidence for them, and Smart
  Tagging only covers some.
- **`D` can start late.** When the whistle goes while the ball is still moving, `D` begins only
  when the ball leaves the pitch or disappears.
- **Live play with the ball hidden for more than 5 s** becomes `D`. On our match that's a few gaps
  at most.
- **Switch timing is backdated** to the start of sustained control, which may be earlier than the
  official moment. We didn't tune a shift, because the only labels using the official definition
  are the public five minutes.
- **The restart team after a hidden restart** comes from the tag or the first control. If that
  first control is an interception, the flight ends up with the wrong team.
- **A pass to the taker during a dead ball** can end `D` a few seconds early.
- **Slots aren't stable identities.** Slot switches and plane projection of high balls will
  occasionally confuse the control logic.

With more time we'd add video-based stoppage evidence for the visible-whistle case (collective
halt, referee signals) and a principled study of the official switch moment on external data.

## Reproducing our checks

- **Metrica.** A stoppage event (ball out, shot out, foul received, card, challenge foul) starts
  `D`. The span up to the next restart — a set piece, or any event whose subtype contains
  `GOAL KICK` — is `D`, unless a live event (pass, carry, a shot that isn't out) happens in
  between. A goal is `D` until the next kick-off. The team between events is the team of the
  latest pass, carry, recovery, shot or set piece.
- **SkillCorner.** Outside every `phases_of_play` interval = `D`; inside exactly one phase = that
  phase's team; overlapping phases = `X`. Weak labels carry known errors: missed stoppages,
  event-time offsets, and goal kicks recorded without a `GOAL KICK` subtype.

## Reproduce the submission

Tested with Python 3.9.6 and 3.11.16 and the exact packages in `requirements.txt`. The repository
contains no match data. Put the three official tactical files under `data/tactical/`:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

```bash
git clone https://github.com/doooze08/whc-mtah2027 && cd whc-mtah2027
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# copy the three official files into data/tactical/ (see above)
.venv/bin/python run.py --data-dir data --out WHC.csv
.venv/bin/python evaluation/validate_submission.py WHC.csv
```

No randomness, no network, only local files. Expected SHA-256 of `WHC.csv`:
`dc215b076e0134a62cec7271673c34a5d7247442549347f913bd5724bd0801ef`.

| Input file | SHA-256 |
|---|---|
| `data/tactical/BAR-ATH_tactical_atd.txt` | `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431` |
| `data/tactical/BAR-ATH_tactical_atd.xml` | `327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a` |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | `0594d35e3df01b9c5002372d8615e2e8ef5379ac8f4e84c5ef7a53f4304c9019` |
