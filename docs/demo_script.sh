#!/usr/bin/env bash
# Headless gif driver: asciinema rec -c "bash docs/demo_script.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
say() { printf '\033[2m# %s\033[0m\n' "$1"; sleep 0.9; }
type_cmd() { printf '\033[1;32m$\033[0m '; local s="$1"; for ((i=0;i<${#s};i++)); do printf '%s' "${s:i:1}"; sleep 0.03; done; printf '\n'; sleep 0.3; }

say "grounded-harness: every run evaluated, every run resumable, every gate able to fail"
sleep 0.5
say "an agent over grounded-mcp's vault (real MCP server, scripted model, offline)"
type_cmd "grounded-harness demo"
$PY -m grounded_harness.cli demo
sleep 1.8
say "sabotage 1: the tool breaks mid-read -> claims lose their source"
type_cmd "grounded-harness demo --sabotage broken-tool"
set +e; $PY -m grounded_harness.cli demo --sabotage broken-tool; echo "exit code: $?"; set -e
sleep 1.8
say "sabotage 2: thin the sample -> the gate refuses to pass on insufficient evidence"
type_cmd "grounded-harness demo --sabotage thin-evidence"
set +e; $PY -m grounded_harness.cli demo --sabotage thin-evidence; echo "exit code: $?"; set -e
sleep 1.6
say "a green gate only means something if you have watched it go red - github.com/rich-atkins/grounded-harness"
sleep 1.5
