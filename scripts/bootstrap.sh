#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_EXE="${CONDA_EXE:-/home/pair/miniconda3/bin/conda}"
YAM_ENV_PREFIX="${YAM_ENV_PREFIX:-$PWD/.conda-env}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PWD/.conda-pkgs}"
if [[ ! -x "$YAM_ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXE" env create --prefix "$YAM_ENV_PREFIX" -f environment.yml
fi
# ruckig's upstream build requires this build-backend constraint.
printf 'scikit-build-core<0.10\n' > "$YAM_ENV_PREFIX/build-constraints.txt"
"$CONDA_EXE" run --no-capture-output -p "$YAM_ENV_PREFIX" python scripts/jetson_i2rt_wheel.py
"$CONDA_EXE" run --no-capture-output -p "$YAM_ENV_PREFIX" python -m pip install --build-constraint "$YAM_ENV_PREFIX/build-constraints.txt" .vendor-wheels/jetson/i2rt-1.1.2-py3-none-any.whl
"$CONDA_EXE" run -p "$YAM_ENV_PREFIX" python -m pip install ../pyzed-5.2-cp313-cp313-linux_aarch64.whl
"$CONDA_EXE" run -p "$YAM_ENV_PREFIX" python -m pip check
"$CONDA_EXE" list -p "$YAM_ENV_PREFIX" --explicit > conda-linux-aarch64.lock
"$CONDA_EXE" run -p "$YAM_ENV_PREFIX" python -m pip freeze > requirements.lock.txt
"$CONDA_EXE" run -p "$YAM_ENV_PREFIX" python scripts/smoke.py
