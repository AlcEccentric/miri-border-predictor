# miri-border-predictor

KNN-based score-trajectory predictor for MiriShita events. Production code
reads event data + per-group configs from Cloudflare R2, runs a KNN-based
trajectory prediction for each (idol, border), writes predictions back to
R2, and uploads a step-stamped snapshot every 30 normalized steps for
later reference.

## Repository layout

```
.
├── src/                             # production code
│   ├── main.py                      # entry point: runs one prediction
│   ├── predict.py                   # prediction orchestration + CI lookup
│   ├── core/                        # pure transforms
│   │   ├── data_processing.py
│   │   ├── feature_engineering.py
│   │   ├── interpolation.py
│   │   ├── normalization.py
│   │   └── smoothing.py
│   ├── knn/                         # KNN predictor
│   │   ├── __init__.py              # public API: predict_curve_knn, get_filtered_df
│   │   ├── predictor.py             # top-level orchestration
│   │   ├── alignment.py             # scale/offset alignment of neighbours
│   │   ├── config.py                # GroupConfig + `src/knn/config.py` + overlay
│   │   ├── distance.py              # trajectory distance + nearest-neighbour search
│   │   ├── plotting.py              # optional DEBUG-level plots
│   │   └── stage.py                 # early/mid/late stage param selection
│   └── storage/                     # R2 + loaders + dynamic config
│       ├── r2_client.py
│       ├── r2_downloader.py
│       ├── loader.py                # event/border + CI R2 helpers
│       └── dynamic_config.py        # periodically-refreshed config overlay
│
├── scripts/                         # developer utilities
│   ├── batch_knn_test.py            # offline backtesting over recent events
│   ├── analyze_percentiles.py       # CI from batch-test rel-errors
│   ├── refresh_config.py            # orchestrates min_event_id + CI refresh
│   └── promote_config.py            # promotes a staged generation to latest/
│
├── logger_config.py                 # shared logging setup
├── requirements.txt
├── .env                             # R2 credentials (not committed)
│
├── confidence_intervals/            # local CI CSVs (dev/debug only; prod reads R2)
├── test_data/                       # local event_info + border_info cache
├── test_results/                    # batch_knn_test outputs
└── debug/                           # DEBUG-level prediction plots
```

## Running things

All commands run from the repo root. Use `python3 -m …` so the `src.` and
`scripts.` imports resolve.

### Production prediction

```bash
python3 -m src.main
# equivalent to:
python3 -m src.main --mode production
```

Reads the latest event metadata, loads input data from R2, loads the
dynamic config overlay from R2 (`config/latest/dynamic_config.json`) and
per-group CI files (`ci/latest/confidence_intervals_{et}_{sub}_{border}.csv`),
computes predictions, and uploads them to
`prediction/{idol_id}/{border}/predictions.json`.

### Dry-run prediction (safe for local testing)

```bash
python3 -m src.main --mode dry_run
python3 -m src.main --mode dry_run_with_cache_refresh
```

Both variants:

- Upload predictions to `dry_run/prediction/...` in R2 instead of the
  production path, so live consumers are untouched.
- Cache event/border input data at `local_cache/main/` (picked up by the
  dataframe pickles in `load_all_data`).
- Bypass the `should_skip_prediction` timing gate so you can test outside
  the active event window.

`dry_run` reuses whatever is already cached at `local_cache/main/` — fast
re-runs during iteration. `dry_run_with_cache_refresh` wipes that
directory first so the next load pulls fresh data from R2 — use when you
suspect stale cached inputs are hiding a real issue.

Pair with `--log-level DEBUG` if you want the per-neighbour weight
breakdown from the KNN core.

### Offline knn test (hyperparam optimization)

```bash
python3 -m scripts.batch_knn_test
```

Reads from `test_data/` (populate it first, see below), runs KNN on a
sliding window of steps for recent events, and writes
`test_results/batch_knn_results_{et}_{sub}_{border}.csv` plus a markdown
summary. Edit `CONFIG` inside `main()` in `scripts/batch_knn_test.py` to
pick the group, steps, and event range.

This is for tweaking the knn configs, both static and dynamic. Remember 
to update `LOOK_BACK_EVENT_CNT_BY_TYPE` in scripts.refresh_config when you
find a better `look_back_event_cnt`

### Populate local `test_data/`

```bash
python3 -m src.storage.r2_downloader --bucket mltd-border-predict
```

Downloads `event_info/event_info_all.csv` and all `border_info_*.csv` into
`test_data/`. Rerunning is idempotent; delete files you want to refresh.

### Refresh dynamic config + CI

Two modes:

```bash
# Dry-run locally. Inputs cached under local_cache/; outputs written to
# local_cache/output/<generation_id>/ only. Nothing uploaded.
python3 -m scripts.refresh_config --mode local

# Production refresh. Uploads to R2 staging paths only. Review the
# diff.md at config/staging/<gen>/diff.md in R2 before promoting. Also
# emails NOTIFICATION_TARGET via Resend with the generation id and
# promote command (set RESEND_API_KEY + EMAIL_FROM, or pass --no-notify).
python3 -m scripts.refresh_config --mode standard
python3 -m scripts.refresh_config --mode standard --no-notify
```

Each run gets a unique `generation_id` like `20260503T120000Z-ab12cd`.
Standard mode always pulls fresh `event_info` / `border_info` from R2
into `local_cache/` and points `BatchKNNTester` there directly, then
removes the cache on exit. Local mode keeps `local_cache/` around so
you can iterate quickly. Neither mode reads or writes `test_data/`.

### Promote a staged config to latest/

```bash
python3 -m scripts.promote_config --generation-id 20260503T120000Z-ab12cd
```

Copies `config/staging/<gen>/` → `config/latest/` and
`ci/staging/<gen>/` → `ci/latest/`. Before overwriting, the current
`latest/` is snapshotted into `config/history/<promote_ts>/` and
`ci/history/<promote_ts>/` as a rollback point.

## Configuration

### Static (source-controlled): `src/knn/config.py`

`GroupConfig` holds per-`(event_type, sub_event_types, border)` strategy
parameters: stage endpoints, k, lookback windows, distance metric,
alignment weights, smoothing choices, etc. These are tuned by humans and
live in source.

### Dynamic (stored in R2): `src/storage/dynamic_config.py`

Fields that change as new events ship go here. Today only
`min_event_id` is overlaid. Structure in R2 as JSON:

```json
{
  "generation_id": "20260503T120000Z-ab12cd",
  "entries": [
    {"event_type": 11.0, "sub_event_types": [1.0], "border": 2500.0, "min_event_id": 200}
  ]
}
```

At startup, `src/main.py` loads this and calls `set_dynamic_overlay(…)`
so `get_group_config(…)` returns a `GroupConfig` with `min_event_id`
filled in from the overlay.

## Data flow in R2

```
mltd-border-predict/
├── event_info/event_info_all.csv           # event metadata
├── border_info/border_info_{eid}_{iid}_{border}.csv
├── metadata/latest_event_border_info.json  # current event pointer
├── config/
│   ├── latest/                             # what production reads
│   │   ├── dynamic_config.json
│   │   └── manifest.json
│   ├── staging/<generation_id>/            # refresh_config writes here
│   │   ├── dynamic_config.json
│   │   ├── manifest.json
│   │   └── diff.md
│   └── history/<promote_ts>/               # pre-promote snapshots
├── ci/
│   ├── latest/confidence_intervals_{et}_{sub}_{border}.csv
│   ├── staging/<generation_id>/…
│   └── history/<promote_ts>/…
└── prediction/
    ├── {idol_id}/{border}/predictions.json                              # latest
    └── prediction_history/{event_id}/step_{step}/{idol_id}/{border}/… # every 30 steps
```

## Setting up from scratch

1. Install deps: `pip install -r requirements.txt`
2. Create `.env` at the repo root with:
   ```
   R2_ACCESS_KEY_ID=...
   R2_SECRET_ACCESS_KEY=...
   R2_ENDPOINT_URL=...

   # Optional: Resend email notifications after standard-mode refresh
   NOTIFICATION_TARGET=you@example.com
   RESEND_API_KEY=re_...
   EMAIL_FROM=Refresh Bot <refresh-bot@mail.yourdomain.com>
   ```
   `EMAIL_FROM` must be a verified Resend domain. See
   [Email notifications](#email-notifications) for details.
3. Download event/border data: `python3 -m src.storage.r2_downloader --bucket mltd-border-predict`
4. **First-time only**: R2 doesn't have `config/latest/` yet.
   ```bash
   python3 -m scripts.refresh_config --mode standard
   python3 -m scripts.promote_config --generation-id <printed-by-previous-step>
   ```
   After this the production path works.

## Email notifications

Standard-mode refresh sends an email via Resend when it finishes, so you
know to review the diff and promote the new generation.

Required env vars:

```
NOTIFICATION_TARGET=you@example.com
RESEND_API_KEY=re_...
EMAIL_FROM=Refresh Bot <refresh-bot@yourdomain.com>
```

Set these in `.env` and the refresh script picks them up automatically.
The notifier is a no-op (logs a warning and returns) when
`RESEND_API_KEY` or `NOTIFICATION_TARGET` is unset, so running on a
machine without email configured works without special flags.

Disable explicitly with:

```bash
python3 -m scripts.refresh_config --mode standard --no-notify
```

## Typical workflows

### "I want to tune a GroupConfig field"
Edit `src/knn/config.py`, commit, deploy. No R2 interaction.

### "I want to refresh CI / min_event_id because new events finished"
```bash
python3 -m scripts.refresh_config --mode local        # dry-run locally
# inspect local_cache/output/<gen>/diff.md
python3 -m scripts.refresh_config --mode standard     # upload staging
# review r2://…/config/staging/<gen>/diff.md
python3 -m scripts.promote_config --generation-id <gen>
```

### "I want to backtest to tune knn params"
1. Make the change in `src/knn/config.py` and `CONFIG` inside `main()` in `scripts/batch_knn_test.py`.
2. Run `python3 -m scripts.batch_knn_test` for the group you care about.
3. Inspect `test_results/batch_knn_summary_*.md`.
4. If it looks good,
   1. Save `src/knn/config.py` and update `LOOK_BACK_EVENT_CNT_BY_TYPE` in `scripts/refresh_config.py`,
   2. Commit them to Github
   3. Trigger Render job to run `scripts.refresh_config --mode standard` to get a fresh CI and dynamic config, then promote.
   4. Static take effect immediately once you push it to Github.

### "I want to roll back to a previous config"
1. List the history: `aws s3 ls s3://mltd-border-predict/config/history/` (or use R2 console).
2. Pick a `<promote_ts>` folder to restore from.
3. Copy its contents over `config/latest/` (and `ci/history/<ts>/` → `ci/latest/`) manually.

## Caveats

- **Dynamic overlay is load-once per process.** Long-running processes
  won't pick up a new generation until restart.
- **CI DataFrame cache in `src/predict.py` is also load-once per process.**
- **Log noise from matplotlib.** The plotting code sets the `matplotlib`
  logger to WARNING when DEBUG is enabled globally.
