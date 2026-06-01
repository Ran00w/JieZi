#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${CONDA_PREFIX:-}"
if [ -z "${ENV_PREFIX}" ]; then
  echo "Error: CONDA_PREFIX is not set. Please activate the conda environment first." >&2
  exit 1
fi

CACHE_ROOT="${JIEZI_CACHE_ROOT:-$HOME/.cache/jiezi-bench}"

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  echo "Missing python in conda env: ${ENV_PREFIX}" >&2
  exit 1
fi

"${ENV_PREFIX}/bin/pip" install -r "$(dirname "$0")/../requirements-eval.txt"

mkdir -p \
  "${CACHE_ROOT}/huggingface/hub" \
  "${CACHE_ROOT}/huggingface/transformers" \
  "${CACHE_ROOT}/modelscope/datasets" \
  "${CACHE_ROOT}/matplotlib" \
  "${CACHE_ROOT}/xdg"

mkdir -p "${ENV_PREFIX}/etc/conda/activate.d" "${ENV_PREFIX}/etc/conda/deactivate.d"

cat > "${ENV_PREFIX}/etc/conda/activate.d/jiezi_cache_env.sh" <<EOF
#!/usr/bin/env bash
export _OLD_JIEZI_CACHE_ROOT="\${JIEZI_CACHE_ROOT-}"
export _OLD_HF_HOME="\${HF_HOME-}"
export _OLD_HUGGINGFACE_HUB_CACHE="\${HUGGINGFACE_HUB_CACHE-}"
export _OLD_TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE-}"
export _OLD_MODELSCOPE_CACHE="\${MODELSCOPE_CACHE-}"
export _OLD_MS_CACHE_HOME="\${MS_CACHE_HOME-}"
export _OLD_MODELSCOPE_HOME="\${MODELSCOPE_HOME-}"
export _OLD_MODELSCOPE_DATASETS_CACHE="\${MODELSCOPE_DATASETS_CACHE-}"
export _OLD_XDG_CACHE_HOME="\${XDG_CACHE_HOME-}"
export _OLD_MPLCONFIGDIR="\${MPLCONFIGDIR-}"

export JIEZI_CACHE_ROOT="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export MS_CACHE_HOME="${CACHE_ROOT}/modelscope"
export MODELSCOPE_HOME="${CACHE_ROOT}/modelscope"
export MODELSCOPE_DATASETS_CACHE="${CACHE_ROOT}/modelscope/datasets"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
EOF

cat > "${ENV_PREFIX}/etc/conda/deactivate.d/jiezi_cache_env.sh" <<'EOF'
#!/usr/bin/env bash
if [ -n "${_OLD_JIEZI_CACHE_ROOT+x}" ]; then export JIEZI_CACHE_ROOT="${_OLD_JIEZI_CACHE_ROOT}"; else unset JIEZI_CACHE_ROOT; fi
if [ -n "${_OLD_HF_HOME+x}" ]; then export HF_HOME="${_OLD_HF_HOME}"; else unset HF_HOME; fi
if [ -n "${_OLD_HUGGINGFACE_HUB_CACHE+x}" ]; then export HUGGINGFACE_HUB_CACHE="${_OLD_HUGGINGFACE_HUB_CACHE}"; else unset HUGGINGFACE_HUB_CACHE; fi
if [ -n "${_OLD_TRANSFORMERS_CACHE+x}" ]; then export TRANSFORMERS_CACHE="${_OLD_TRANSFORMERS_CACHE}"; else unset TRANSFORMERS_CACHE; fi
if [ -n "${_OLD_MODELSCOPE_CACHE+x}" ]; then export MODELSCOPE_CACHE="${_OLD_MODELSCOPE_CACHE}"; else unset MODELSCOPE_CACHE; fi
if [ -n "${_OLD_MS_CACHE_HOME+x}" ]; then export MS_CACHE_HOME="${_OLD_MS_CACHE_HOME}"; else unset MS_CACHE_HOME; fi
if [ -n "${_OLD_MODELSCOPE_HOME+x}" ]; then export MODELSCOPE_HOME="${_OLD_MODELSCOPE_HOME}"; else unset MODELSCOPE_HOME; fi
if [ -n "${_OLD_MODELSCOPE_DATASETS_CACHE+x}" ]; then export MODELSCOPE_DATASETS_CACHE="${_OLD_MODELSCOPE_DATASETS_CACHE}"; else unset MODELSCOPE_DATASETS_CACHE; fi
if [ -n "${_OLD_XDG_CACHE_HOME+x}" ]; then export XDG_CACHE_HOME="${_OLD_XDG_CACHE_HOME}"; else unset XDG_CACHE_HOME; fi
if [ -n "${_OLD_MPLCONFIGDIR+x}" ]; then export MPLCONFIGDIR="${_OLD_MPLCONFIGDIR}"; else unset MPLCONFIGDIR; fi

unset _OLD_JIEZI_CACHE_ROOT _OLD_HF_HOME _OLD_HUGGINGFACE_HUB_CACHE _OLD_TRANSFORMERS_CACHE
unset _OLD_MODELSCOPE_CACHE _OLD_MS_CACHE_HOME _OLD_MODELSCOPE_HOME _OLD_MODELSCOPE_DATASETS_CACHE
unset _OLD_XDG_CACHE_HOME _OLD_MPLCONFIGDIR
EOF

chmod +x \
  "${ENV_PREFIX}/etc/conda/activate.d/jiezi_cache_env.sh" \
  "${ENV_PREFIX}/etc/conda/deactivate.d/jiezi_cache_env.sh"

echo "JieZi-bench env is ready at: ${ENV_PREFIX}"
echo "Use: conda activate ${ENV_PREFIX}"
