# Mobius — Autonomous Controller Optimization

This is an experiment to have the LLM autonomously optimize a lateral steering controller for comma.ai's controls challenge simulator.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date and time (e.g. `mar13-1240`). The branch `mobius/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b mobius/<tag>` from current HEAD.
3. **Read the in-scope files**: Read these files for full context:
   - `program.md` — this file; your instructions.
   - `controller.py` — the file you modify. Controller logic.
   - `evaluate.py` — fixed evaluation harness. Do not modify.
   - `tinyphysics.py` — simulator engine. Do not modify.
4. **Verify data exists**: Check that `data/` contains CSV files. If not, run `uv run evaluate.py --num-segs 1` to trigger auto-download.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Run baseline**: Run the unmodified PID controller and record it as the first row.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment evaluates the controller against 1000 shuffled driving segments (~10-12 min on 16 cores). You launch it as: `uv run evaluate.py > run.log 2>&1`

For quick debugging/sanity checks: `uv run evaluate.py --num-segs 20` (~15s). But always record the full 1000-segment result.

**What you CAN do:**
- Modify `controller.py` — this is the only file you edit. Everything is fair game: control law, state estimation, feedforward, adaptive gains, lookup tables, optimization, predictive control, anything.
- If a new package is needed, add it as inline `uv` dependency metadata in the `# /// script` block at the top of `controller.py` so it gets installed at runtime.

**What you CANNOT do:**
- Modify `evaluate.py`, `tinyphysics.py`, `models/`, or `pyproject.toml`.

**The goal is simple: get the lowest total_cost.**

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

## Domain knowledge

Understanding the cost function and simulator is critical for making good controller design decisions.

**Cost breakdown:**
- `total_cost = lataccel_cost * 50 + jerk_cost`
- `lataccel_cost = mean((target - predicted)^2) * 100` — tracking error
- `jerk_cost = mean((diff(predicted) / 0.1)^2) * 100` — smoothness
- lataccel_cost is weighted **50x** over jerk_cost. Prioritize tracking accuracy, but smoothness still matters.

**Simulator properties:**
- 10 Hz control rate (0.1s per step)
- Controller is dormant until step 100 (CONTROL_START_IDX), active through step 500
- Steering output clipped to [-2, 2]
- Lateral acceleration change clipped to +/-0.5 per step (autoregressive model constraint)
- The simulator is **autoregressive**: controller actions affect future model predictions (error propagation)

**Available information per step:**
- `target_lataccel`: desired lateral acceleration now
- `current_lataccel`: actual lateral acceleration (from model)
- `state`: (roll_lataccel, v_ego, a_ego) — current vehicle state
- `future_plan`: 50 steps (5 sec) of lookahead with (lataccel, roll_lataccel, v_ego, a_ego)

**Example directions** (random ideas; you are NOT limited to these):
1. **Feedforward from future plan** — anticipate upcoming target changes instead of only reacting
2. **Roll compensation** — subtract `state.roll_lataccel` contribution to reduce steady-state error
3. **Velocity-dependent gains** — steer-to-lataccel sensitivity changes with speed
4. **Integrator anti-windup** — prevent integrator runaway on sustained errors
5. **Derivative filtering** — low-pass filter on error derivative to reduce noise amplification
6. **Gain optimization** — systematic search over P/I/D values using quick eval runs
7. **Predictive/MPC-style control** — multi-step optimization using the future plan
8. **Nonlinear gains** — aggressive correction when far from target, gentle when close
9. **Preview-based feedforward** — weight near-future targets more than far-future
10. **Kalman filtering** — smooth state estimation to reduce jerk from noisy measurements

**Anti-patterns:**
- Don't hardcode per-segment behavior (controller must generalize)
- Don't try heavy computation per step (keep update() fast for 1000-seg eval)
- Don't accumulate unbounded state (integrators/buffers should be bounded)

## Output format

After evaluation, the script prints a greppable summary:

```
---
total_cost:    123.456789
lataccel_cost: 2.345678
jerk_cost:     6.789012
num_segments:  1000
elapsed_secs:  22.3
```

Extract key metrics: `command grep "^total_cost:\|^lataccel_cost:\|^jerk_cost:" run.log`

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	total_cost	lataccel_cost	jerk_cost	status	description
```

1. git commit hash (short, 7 chars)
2. total_cost achieved (e.g. 123.456789) — use 0.000000 for crashes
3. lataccel_cost (e.g. 2.345678) — use 0.000000 for crashes
4. jerk_cost (e.g. 6.789012) — use 0.000000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	total_cost	lataccel_cost	jerk_cost	status	description
a1b2c3d	125.678901	2.123456	19.555678	keep	baseline PID
b2c3d4e	118.234567	1.987654	19.012345	keep	add feedforward from future plan
c3d4e5f	130.000000	2.500000	5.000000	discard	aggressive derivative gain
d4e5f6g	0.000000	0.000000	0.000000	crash	MPC with scipy (import error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `mobius/mar13-1240`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `controller.py` with an experimental idea
3. `git commit`
4. Run the experiment: `uv run evaluate.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `command grep "^total_cost:\|^lataccel_cost:\|^jerk_cost:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in results.tsv (NOTE: do not commit results.tsv, leave it untracked by git)
8. If total_cost improved (lower), you "advance" the branch, keeping the git commit
9. If total_cost is equal or worse, you `git reset --hard HEAD~1` to discard the commit

**Timeout**: Each experiment should take ~10-12 minutes for 1000 segments. If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment. If it's a typo or missing import, fix and re-run. If the idea is fundamentally broken, skip it, log "crash", and move on.

**Quick iteration**: For debugging, use `--num-segs 20` (~15s) to validate that code runs without errors before committing to a full 1000-segment evaluation.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~12 minutes then you can run approx 5/hour, for a total of about 40 over the duration of the average human sleep. The user then wakes up to experimental results and an optimized controller, all completed by you while they slept!
