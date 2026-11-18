#!/usr/bin/env bash
# Start the Prime router stack from this config. Arguments are passed to
# `vllm-sr serve`, for example --router-image vllm-sr:<commit>.
#
# The codex-lb references use provider openai, whose catalog entry falls back
# to https://api.openai.com/v1 when the endpoint is empty. So refuse to start
# unless every variable the config reads is set: an unset CODEX_LB_BASE_URL
# would otherwise send prompts and the codex-lb key to OpenAI. Config reloads
# use the environment the container got here.
set -euo pipefail

for name in SR_CLIENT_KEY SR_MGMT_TOKEN CODEX_LB_BASE_URL CODEX_LB_API_KEY MERIDIAN_BASE_URL MERIDIAN_API_KEY; do
  if [ -z "${!name:-}" ]; then
    echo "serve.sh: $name is not set" >&2
    exit 1
  fi
done

dir="$(cd "$(dirname "$0")" && pwd)"
exec "$dir/../.venv/bin/vllm-sr" serve --config "$dir/config.yaml" --minimal --gateway standalone \
  --image-pull-policy never "$@"
