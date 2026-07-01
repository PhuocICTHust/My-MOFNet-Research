# ==========================================
# APPLY BEST CONFIG (from BRCA grid search) TO REMAINING COHORTS
# ==========================================
# Best config found by parse_log.py over the BRCA grid search
# (5 split seeds, ranked by macro_f1):
#   conv_type=gat | fusion=cross | lr=1e-3 | mixup_alpha=0.1
#   Macro-F1 = 0.7689 +/- 0.0279
# This script reuses run_grid_search.ps1's structure but fixes the model
# config to that single best setting and sweeps it across the 4 remaining
# cohorts (COAD, GBM, LGG, OV), each with the same 5x5 split-seed protocol
# used for BRCA.
#
# FIX WINDOWS ENVIRONMENT
# ==========================================
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING     = "utf-8"
$env:PYTHONUTF8           = "1"   # belt-and-suspenders UTF-8 (Py3.7+), in case
                                   # PYTHONIOENCODING alone doesn't cover a
                                   # redirected (non-console) stdout on this box.

# ==========================================
# CONFIG PATHS
# ==========================================
$DATA_DIR    = "E:\Cancer-classification-dataset"
$LOG_DIR     = ".\logs_remaining_cohorts"

# FIX (root cause of "no .pt file ever created"): README section 2 ("Files")
# and section 5 ("How to run") both show prepare_graph.py living under
# scripts/ while main.py sits at the repo root:
#     python scripts/prepare_graph.py --cancer BRCA --split_seed 0
#     python main.py --cancer BRCA --data_path "..." --split_seed 0
# The previous version of this script called "python prepare_graph.py"
# (no scripts\ prefix). If run from the MOFNet root, Python can't find the
# file and exits immediately with an error -- which used to be silently
# discarded by "2>&1 | Out-Null" below, so prepare_graph.py never wrote any
# .pt file and nothing told you why.
# If your actual layout differs, edit this one line -- everything else
# below now fails LOUDLY if the path is wrong, instead of silently.
$PREP_SCRIPT = "scripts\prepare_graph.py"
$MAIN_SCRIPT = "main.py"

if (-Not (Test-Path -Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR | Out-Null
}

# FIX: fail fast and clearly if either script is missing, rather than
# discovering it later as a confusing FileNotFoundError on a .pt file with
# zero diagnostic trail.
foreach ($script in @($PREP_SCRIPT, $MAIN_SCRIPT)) {
    if (-Not (Test-Path -Path $script)) {
        Write-Host "FATAL: cannot find '$script' relative to $(Get-Location)." -ForegroundColor Red
        Write-Host "       Run this .ps1 from the MOFNet root folder, or fix the path at the top of this script." -ForegroundColor Red
        exit 1
    }
}

# ==========================================
# BEST CONFIG, APPLIED ACROSS REMAINING COHORTS
# ==========================================
# Single fixed config (best from the BRCA grid search) -- each list has
# exactly one value, so the nested foreach loops below run it once per
# cancer x split_seed without changing any of the surrounding logic.
$CANCERS        = @("COAD", "GBM", "LGG", "OV")
$CONV_TYPES     = @("gat")
$FUSIONS        = @("cross")
$LEARNING_RATES = @("1e-3")
$MIXUP_ALPHAS   = @("0.1")
$SPLIT_SEEDS    = @(0, 1, 2, 3, 4)

# FIX: collect every failure across the whole run instead of burying each one
# in its own log file with no overview. Printed + saved at the very end.
$failures = @()

Write-Host "APPLYING BEST CONFIG TO REMAINING COHORTS (COAD/GBM/LGG/OV)..." -ForegroundColor Cyan
Write-Host "Config: conv=gat | fusion=cross | lr=1e-3 | mixup_alpha=0.1"
Write-Host "Log directory: $LOG_DIR"
Write-Host "--------------------------------------------------------"

foreach ($CANCER in $CANCERS) {

    # ──────────────────────────────────────
    # 1. GRAPH PREPARATION
    # ──────────────────────────────────────
    Write-Host ">> [1/2] Generating graph data for $CANCER..." -ForegroundColor Magenta
    foreach ($SEED in $SPLIT_SEEDS) {
        $graphFile = Join-Path $DATA_DIR "${CANCER}_graph_s${SEED}.pt"
        $prepLog   = "$LOG_DIR\prepare_${CANCER}_s${SEED}.log"

        Write-Host "   -> Graph Split Seed: $SEED..."

        # FIX: replaced "cmd.exe /c '... 2>&1' | Out-Null" with a direct call
        # + real log file + exit-code check. *> redirects ALL streams
        # (stdout, stderr, warnings, etc.) to the file -- nothing is discarded.
        $prepArgs = @("--cancer", $CANCER, "--split_seed", $SEED)
        & python $PREP_SCRIPT @prepArgs *> $prepLog
        $prepExit = $LASTEXITCODE

        # FIX: verify the actual artifact exists, not just the exit code --
        # catches the case where the process "succeeds" but writes nowhere.
        if ($prepExit -ne 0 -or -Not (Test-Path -Path $graphFile)) {
            Write-Host "   !! FAILED building $graphFile (exit=$prepExit). See $prepLog" -ForegroundColor Red
            $failures += "PREP  | $CANCER s$SEED | exit=$prepExit | log=$prepLog"
        }
    }

    # ──────────────────────────────────────
    # 2. TRAINING
    # ──────────────────────────────────────
    Write-Host ">> [2/2] Training models..." -ForegroundColor Magenta
    foreach ($CONV in $CONV_TYPES) {
        foreach ($FUSION in $FUSIONS) {
            foreach ($LR in $LEARNING_RATES) {
                foreach ($MIXUP in $MIXUP_ALPHAS) {

                    $LOG_FILE = "$LOG_DIR\${CANCER}_${CONV}_${FUSION}_lr${LR}_mixup${MIXUP}.log"
                    "CONFIG: Cancer=$CANCER | Conv=$CONV | Fusion=$FUSION | LR=$LR | MixUp=$MIXUP" |
                        Out-File -FilePath $LOG_FILE -Encoding utf8
                    "========================================================" |
                        Out-File -FilePath $LOG_FILE -Append -Encoding utf8

                    Write-Host ">> Training: $CANCER | Net: $CONV | Fusion: $FUSION | LR: $LR | MixUp: $MIXUP" -ForegroundColor Yellow

                    foreach ($SEED in $SPLIT_SEEDS) {
                        $graphFile = Join-Path $DATA_DIR "${CANCER}_graph_s${SEED}.pt"

                        # FIX: skip cleanly with a clear log line instead of
                        # letting main.py crash 1-2s in with the same generic
                        # FileNotFoundError every time, for every cause.
                        if (-Not (Test-Path -Path $graphFile)) {
                            "`n`n=== SPLIT SEED: $SEED -- SKIPPED: $graphFile not found (graph prep failed earlier) ===" |
                                Out-File -FilePath $LOG_FILE -Append -Encoding utf8
                            Write-Host "  -> Skipping Split Seed $SEED ($graphFile missing)" -ForegroundColor DarkYellow
                            $failures += "TRAIN | $CANCER/$CONV/$FUSION/lr$LR/mix$MIXUP s$SEED | SKIPPED (no graph file)"
                            continue
                        }

                        Write-Host "  -> Running Split Seed: $SEED..."
                        "`n`n=== SPLIT SEED: $SEED ===" | Out-File -FilePath $LOG_FILE -Append -Encoding utf8

                        # FIX: dropped the cmd.exe wrapper + backtick-escaped
                        # quotes around $DATA_DIR. Splatting (@mainArgs) passes
                        # each argument to python directly and safely, no
                        # manual quoting needed even if a path has spaces.
                        $mainArgs = @(
                            "--cancer", $CANCER,
                            "--data_path", $DATA_DIR,
                            "--split_seed", $SEED,
                            "--conv_type", $CONV,
                            "--fusion", $FUSION,
                            "--lr", $LR,
                            "--mixup_alpha", $MIXUP,
                            "--epochs", "200",
                            "--patience", "40"
                        )
                        & python $MAIN_SCRIPT @mainArgs *>> $LOG_FILE
                        $trainExit = $LASTEXITCODE

                        if ($trainExit -ne 0) {
                            Write-Host "  !! Training FAILED (exit=$trainExit). See $LOG_FILE" -ForegroundColor Red
                            $failures += "TRAIN | $CANCER/$CONV/$FUSION/lr$LR/mix$MIXUP s$SEED | exit=$trainExit | log=$LOG_FILE"
                        }
                    }

                    Write-Host "Done: $CONV + $FUSION (LR=$LR, MixUp=$MIXUP). Log: $LOG_FILE" -ForegroundColor Green
                    Write-Host "--------------------------------------------------------"
                }
            }
        }
    }
}

# ==========================================
# SUMMARY  (so a failure can never again hide silently in 100+ log files)
# ==========================================
Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "ALL COHORTS COMPLETED -- all runs succeeded." -ForegroundColor Cyan
} else {
    Write-Host "COMPLETED WITH $($failures.Count) FAILURE(S):" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    $failures | Out-File -FilePath "$LOG_DIR\_FAILURES_SUMMARY.txt" -Encoding utf8
    Write-Host "Full list written to $LOG_DIR\_FAILURES_SUMMARY.txt" -ForegroundColor Red
}
