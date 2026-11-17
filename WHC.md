# WHC — More than a Hack 2027, Phase 1: frame-level possession for BAR–ATH

Team: **WHC**. Submission file: `WHC.csv` (199,319 rows, `frame,label`).

## 1. Summary

We label every frame of the tactical video with an **offline, rule-based state machine** that reads
only the automatic tracking (ATD) and the automatic Smart Tagging `SET PIECES` intervals. There is
no trained model and no learned parameter: every threshold is a physical or definitional choice
written in the experiment config, and none was fitted to the provided game or to the public
five-minute answer.

The algorithm answers two questions per frame:

1. **Is the ball in play?** Out of the pitch, a long ball gap, or a tagged stoppage gives `D`;
   observed restarts end `D`.
2. **If so, which team controls it?** A team keeps possession until an opponent shows *sustained*
   control of the ball; touches, flights and contests do not change possession.

We chose rules over a learned model because no labelled data with the official H/A/D definition
exists outside the provided game, and labels on the provided game may not be used. Rules let us
state each assumption, test it on simulated and external data, and keep the output exactly
reproducible.

## 2. Data assets

| Asset | Used? | How |
|---|---|---|
| Tactical ATD (`BAR-ATH_tactical_atd.txt` + `.xml`) | **Yes** | Ball and player plane positions, team of each tracking slot, pitch size, 25 fps. Speeds are computed from position differences. Ball height, the `_s` speed column, player identity and referee signals are not used. |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`) | **Yes, `SET PIECES` only** | A set-piece tag is taken as evidence that a stoppage and a restart happened; coordinates still decide where they start and end. Phase, transition and goal tags are not used. |
| Official match boundaries / frame alignment | Yes | Through the provided `games.py` and loaders. |
| Tactical / broadcast video | No | Considered for whistle stoppages (players stopping, referee gestures). Dropped for cost and time; no video-derived signal is in the submission. |
| Broadcast tracking, homography | No | Not needed once the tactical ATD was the single timeline. |
| Public 5-minute ground truth | Pipeline check only | Used to check file format, frame alignment and evaluation flow, and to understand what the official `D` covers. Never used to select, tune or stop any experiment. |

**Training data:** none. No model is trained.

**Validation data** (all about other games, or synthetic; no labels of any kind on the provided game):

| Dataset | Source and licence | Labels | Amount | Role |
|---|---|---|---|---|
| Our simulator | Written by the team (simulation code in the development repository) | Ground truth comes from the scripted scenario, never from a candidate algorithm | 18 scenario subtypes (throw-ins, header duels, bouncing passes, loose and contested balls, stoppages and restarts), each with home/away mirrors and several ball-dropout conditions | Regression only. The simulator encodes our own assumptions, so it cannot show real-match accuracy. |
| Metrica Sports sample data, Games 1–3 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data); no explicit licence, attribution requested | Frame labels **derived automatically by us** from Metrica's event data with an event-order rule (Section 5). The events are vendor data for those games; we did not hand-label any frame. | 3 matches, about 95 min each, 25 fps | Main external check. Games 1–2 for design, Game 3 held out. |
| SkillCorner open data, match 1925299 | [SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT; data by SkillCorner and PySport | Frames outside every vendor `phases_of_play` interval = D, inside = the phase team | 1 match, 10 fps resampled to 25 fps | Secondary check, used once. |

The development repository with all experiment records is private and can be shared with the
organisers on request.

External weak labels are approximations of H/A/D, not the official definition. We use them to ask
coarse questions, such as "does this rule flip whole dead-ball stretches to live?", and not to
claim an accuracy.

## 3. Possession rules

The implementation is `algorithms/dead_ball_aware/predict.py`. The submitted parameters are in
`config/whc.yaml`.

### 3.1 Control and team changes

- A slot is a **control candidate** when the ball is within **1.3 m** of it and the ball's speed
  relative to that slot stays at or below **3 m/s** for **0.32 s** (8 intervals at 25 fps). Being
  the nearest player is not enough.
- If both teams satisfy this at the same time, it is a **contest**, and no new control is
  established.
- Possession changes **only** when an opponent confirms control. The change is backdated to the
  start of that control, with a small correction to the frame where the incoming ball's velocity
  first changes.
- **Loose balls, flights, deflections and contests keep the previous team.** This follows the
  challenge rule that a touch without control is not a possession change.
- The submission never outputs `X`. Frames outside the graded halves get the same state machine
  output and are not scored.

### 3.2 Dead ball (`D`)

`D` starts from any of:

- **Ball out:** the ball centre is beyond a line by more than the ball radius (0.11 m). We use plane
  positions only, so a high ball projected outside the line counts as out.
- **Long ball gap:** the ball is missing for more than **5 s**. The whole gap is labelled `D`,
  backdated to its start. The tracking provider removes the ball during most stoppages; short gaps
  keep the previous state.
- **Tagged stoppage** (trusted-tag mode): a `SET PIECES` tag with coordinate support starts `D`
  either at an earlier exit or long gap, or at the start of a slow-ball period of at least 0.5 s
  before the restart. A tag without such coordinate support does not create `D`.

Inside `D`, picking up, carrying, placing or walking with the ball does **not** restart play.

### 3.3 Restarts (leaving `D`)

1. **Observed release.** The ball goes from slow (≤ 3 m/s) to a release speed of at least 4 m/s,
   rises by at least 1.5 m/s within 0.12 s, and separates from a taker within reach. The taker's
   team gets possession. A throw-in restarts at the first frame the ball is no longer wholly out.
2. **Pass confirmation.** Two different players of the same team confirm control in turn, and the
   ball visibly travels between them. Play is backdated to when the ball left the first player.
3. **Moving reappearance after a long gap.** If a long gap ends with the ball already moving
   (first speed sample above 3 m/s), no release was seen, and the ball then stays visible and inside
   the pitch with no further gap until a player confirms control within 5 s, the restart happened
   while the ball was hidden. `D` ends at the reappearance frame (or, for a throw-in, the frame the
   ball enters the pitch). The team is the matched set-piece tag's team, otherwise the first team
   to confirm control within 5 s.

   This rule follows from what the official `D` covers: a `D` with a ball gap ends soon after the
   ball reappears. On our match the provider usually brings the ball back only after it has been
   played.

### 3.4 How tags and coordinates are combined

- Tags are **trusted to say that a stoppage happened**. Coordinates decide when it starts and ends
  and who restarts. Each stoppage takes the first coordinate restart candidate that has exit, gap or
  slow-ball support.
- **Team:** when both teams are within control distance at the restart, the tag's team is used.
  When coordinates clearly show only one team, coordinates win (and the conflict is counted).
- A tag is overruled only when the ball stays visible and fast with both teams contesting
  throughout. In a diagnostic run on the full match, the trusted-tag configuration used 45 of 59
  `SET PIECES` tags. 14 were unmatched and none were overruled.

## 4. How we developed it

1. **Simulation first.** Scripted scenarios with known answers, used to check that each rule does
   what it claims. Results are regression evidence only.
2. **External weak labels.** Metrica and SkillCorner were converted to the same input format as our
   match, with automatically derived frame labels. Rules were designed on Metrica Games 1–2, and
   Game 3 was looked at once after the definitions were frozen.
3. **Answer-free diagnostics on the provided match.** Count how many segments a rule would change and
   what they look like (ball state, collective player speed, nearby tag types). We fixed the
   go/no-go thresholds before running. No answers were read.

### Tried and dropped

| Idea | Why dropped |
|---|---|
| Plain sustained-control state machine without dead-ball handling | Long stoppages were labelled H/A, which was the largest error source. Superseded by `dead_ball_aware`. |
| Collective player speed as a veto (a long gap is `D` only if players slow down) and as a restart confirmation | Players' median speed separates dead from live frame by frame (AUC about 0.8–0.9 on external data). But the two rules gave no stable gain on held-out external data and hurt quick restarts. Not adopted. |
| Ignoring very short exits as tracking error; absorbing possessions shorter than 1 s | On Metrica, one real exit that bounced back cost 33 s. 11 of 12 absorbed short possessions were real. Not adopted. |
| Overruling `D` in long gaps that look live (players moving, ball reappearing far away in open play) | On our match, at most 9 gaps (67 s) look like this, and 4 of them are followed by set pieces. Too small a gain for the risk. Not adopted. |
| Learned classifier | No training labels with the official definition. |

The development repository with all experiment records is private and can be shared with the
organisers on request.

## 5. External weak-label rules (for reproducibility of our checks)

- **Metrica:** a stoppage event (ball out, shot out, foul received, card, challenge foul) starts
  `D`. The span up to the next restart (a set piece, or any event whose subtype contains
  `GOAL KICK`) is `D` unless a live event (pass, carry, a shot that is not out) occurs in between.
  A goal is `D` until the next kick-off. The team between events is the team of the latest pass,
  carry, recovery, shot or set piece.
- **SkillCorner:** outside every `phases_of_play` interval = `D`; inside exactly one phase = that
  phase's team; overlapping phases = `X` (not scored).
- Weak labels have known errors: missed stoppages, event-time offsets, and goal kicks recorded
  without a `GOAL KICK` subtype.

## 6. Known limitations

- **Whistle stoppages with the ball visible and no tag** (fouls, offsides where play is not
  restarted from a ball exit or gap) are labelled live. Coordinates carry almost no evidence for
  them. Smart Tagging covers only some of them.
- **`D` starting late:** when the whistle blows while the ball is still moving, `D` starts only when
  the ball leaves the pitch or disappears.
- **Live play with the ball hidden for more than 5 s** is labelled `D`. On our match this is at most
  a few gaps.
- **Switch timing:** possession changes are backdated to the start of sustained control. This may
  be earlier than the official moment. We did not tune a shift, because the only labels with the
  official definition are the public five minutes.
- **Restart team after a hidden restart** comes from the tag or the first control. If the first
  control is an interception, the flight is assigned to the wrong team.
- A pass to the taker during a dead ball can end `D` a few seconds early.
- Tracking slots are not stable player identities. Slot switches and plane projection of high balls
  can cause errors.

**With more time:** video-based stoppage evidence (collective halt, referee signals) for the
visible-whistle case, and a principled study of the official switch moment on external data.

## 7. Reproduction

Tested with Python 3.9.6 and 3.11.16 and the exact packages in `requirements.txt`. The repository contains
no match data. Put the three official tactical files under `data/tactical/`:

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

The run is deterministic: no randomness; it reads only local files. Expected SHA-256 of `WHC.csv`:
`dc215b076e0134a62cec7271673c34a5d7247442549347f913bd5724bd0801ef`.

Input files used (SHA-256 of the official bundle):

| File | SHA-256 |
|---|---|
| `data/tactical/BAR-ATH_tactical_atd.txt` | `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431` |
| `data/tactical/BAR-ATH_tactical_atd.xml` | `327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a` |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | `0594d35e3df01b9c5002372d8615e2e8ef5379ac8f4e84c5ef7a53f4304c9019` |

## 8. Use of AI assistants and data handling

We used AI coding assistants (Anthropic's Claude Code and OpenAI's Codex) to write code, review
pull requests, run analyses and draft documentation, including this document. All rules, design
decisions and acceptance criteria were decided by the team and recorded in the repository.

The competition data (video, tracking, Smart Tagging, ground truth) was processed only on team
members' local machines and was never uploaded to an assistant or any other third-party service. The
assistants saw the code and the output of local diagnostic runs: mostly counts and aggregate
statistics, and occasionally short excerpts such as the boundaries of a few segments or the
public-window evaluation printout. They never received the data files. The repository contains no
match data and none of our predictions (the only tracked prediction files are the official
baselines shipped with the starter repository), and no per-frame rules or constants taken from the
provided game.
