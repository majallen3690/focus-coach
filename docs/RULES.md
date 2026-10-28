# Usage-state rules (spec)

The one definition of the rules. `ml/rules.py` implements it and `ml/tests/test_rules.py` checks it. Every
number is read from `shared/rules.json`; nothing is hard-coded. If code and this file disagree, this file wins.

## Inputs
A list of detections `{box: [x1, y1, x2, y2], conf, cls}` in pixel coordinates, `cls` = `head` or `phone`.

## Step 1: filter by confidence
Keep a detection when `conf >= conf.<class>` (`conf.phone`, `conf.head`). The comparison is inclusive.

## Step 2: assign each phone to one head
Each phone goes to the head whose centre is nearest, with the distance divided by that head's height (so a
larger, closer head wins a phone at the same pixel distance). A phone with no head at all is ignored: it
cannot be "used" by anyone.

## Step 3: classify one person (head `H`, assigned phones)
Let `w`, `h` be the head width and height, `(hcx, hcy)` the head centre and `(pcx, pcy)` a phone centre.

- **ON_CALL:** the phone box overlaps `H` grown by `call.head_expand × w` left and right and
  `call.head_expand × h` above and below (overlap area greater than zero). A phone at the ear or face.
- **ON_PHONE:** `pcy > hcy`, and `|pcx - hcx| <= using.max_dx_head_widths × w`, and
  `pcy - hcy <= using.max_dy_head_heights × h`. A phone held in front, below the face.
- **FOCUSED:** anything else, including no phone.

ON_CALL is checked first, because a phone at the ear also sits close to the head. If a person has several
phones, the strongest state wins (ON_CALL, then ON_PHONE, then FOCUSED).

## Step 4: image or frame decision
`classify_image(dets, cfg, mode)` returns `using` plus the state of every person.
- `mode="all"` (evaluation): `using` is true if any head is ON_CALL or ON_PHONE. FPI-Det photos often show
  several people, and the label refers to the image.
- `mode="largest"` (the app): only the largest head counts, because the laptop webcam is closest to the user.
  Ties are broken by higher confidence, then lower `x1`.

## Step 5: AWAY and time smoothing (app only, Phase 5)
- **AWAY:** no head detected for longer than `smoothing.away_ms`.
- **Smoothing:** a raw state is only committed after it fills at least `smoothing.enter_ratio` of the last
  `smoothing.window_ms` (by timestamp, so it works at any frame rate). Brief glances are ignored.
- A **pickup** is a committed change from FOCUSED or AWAY into ON_PHONE or ON_CALL. **Minutes lost** is the
  committed time in ON_PHONE plus ON_CALL.

## Binary mapping for evaluation
`using_phone = state in {ON_CALL, ON_PHONE}`. `label.csv` is binary (1 = using a phone), so only this
decision is scored, not the three-way split.

## Tuned values (tune split only)
`conf.phone 0.2`, `conf.head 0.1`, `call.head_expand 0.1`, `using.max_dx_head_widths 3.0`,
`using.max_dy_head_heights 2.0`. See `results/eval/tuning.csv` and `results/eval/summary.md`.
These were chosen for FPI-Det's surveillance-style photos. The app may need its own check on real webcam frames.
