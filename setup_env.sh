#!/usr/bin/env bash

conda activate commonocean-clean

export PYTHONPATH="$HOME/Documents/RL_navigation/commonocean-sim/src:$HOME/Documents/RL_navigation/RL-Autonomous-Surface-Vessels:$PYTHONPATH"
export MPLBACKEND=Agg

echo "Entorno RL_navigation activado"
echo "PYTHONPATH=$PYTHONPATH"
