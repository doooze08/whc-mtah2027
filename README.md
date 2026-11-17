# WHC — More than a Hack 2027, Phase 1

Team WHC's submission for More than a Hack 2027, Phase 1: a rule-based state machine that labels
every frame of the Barcelona–Athletic Club tactical video as `H`, `A` or `D` from the automatic
tracking (ATD) and Smart Tagging. The technical document is [WHC.md](WHC.md).

## Data

The repository contains no match data. Put the three official tactical files here:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

## Reproduce

Tested with Python 3.9.6 and 3.11.16.

```bash
git clone https://github.com/doooze08/whc-mtah2027 && cd whc-mtah2027
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# copy the three official files into data/tactical/ (see above)
.venv/bin/python run.py --data-dir data --out WHC.csv
.venv/bin/python evaluation/validate_submission.py WHC.csv
```

Tests (no match data needed): `.venv/bin/python -m unittest discover -s tests`

The run is deterministic and reads only local files. Expected SHA-256:

| File | SHA-256 |
|---|---|
| `WHC.csv` (output) | `dc215b076e0134a62cec7271673c34a5d7247442549347f913bd5724bd0801ef` |
| `data/tactical/BAR-ATH_tactical_atd.txt` | `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431` |
| `data/tactical/BAR-ATH_tactical_atd.xml` | `327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a` |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | `0594d35e3df01b9c5002372d8615e2e8ef5379ac8f4e84c5ef7a53f4304c9019` |

## Layout

| Path | Content |
|---|---|
| `run.py` | Entry point: load ATD and Smart Tagging, predict, write `frame,label` |
| `config/whc.yaml` | Submitted parameters |
| `algorithms/dead_ball_aware/` | The algorithm |
| `observations/` | ATD loader and observation schema |
| `games.py`, `starter/`, `evaluation/` | Unchanged files from the official starter repository |
| `tests/` | Smoke test on a hand-made clip |
