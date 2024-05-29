#!/bin/bash

# Glosten_Milgrom_ModelFree.py script runner

# Fixed parameters
MAX_EPISODES=1
MAX_EPISODE_LEN=200
SPECIAL_STRING="FirstRun" # change name to reflect a different run
AGENT_TYPE="AKF" # compares AKF, KF, uniswap
AMM_TYPE="mean" # constant mean market maker
JUMP_MODE="linear" # change to log for jumps in returns instead of price directly
WINDOW_CAP=100 # window cap for truncated AKF

NOISE_VARIANCE=12 # eta
JUMP_VARIANCE=10 # sigma

VARY_INFORMED=False # change to true for non stationary eta
VARY_JUMP_PROB=False # change to true for non stationary sigma


# Outer loop for noise_variance from 5 to 25 in steps of 5
for NOISE_VARIANCE in $(seq 12 2 20)
do
    echo "Running with noise_variance=$NOISE_VARIANCE"
    
    # Inner loop through square values from 4 to 100 for jump_variance
	for i in {0..20}
	do
	    VARY_JUMP_SIZE=$((2 * i))
	    echo "    Running with jump_variance=$JUMP_VARIANCE"
	    python main.py --window_cap $WINDOW_CAP --vary_informed $VARY_INFORMED --vary_jump_prob $VARY_JUMP_PROB --agent_type $AGENT_TYPE --AMM_type $AMM_TYPE --jump_mode $JUMP_MODE --noise_variance $NOISE_VARIANCE --jump_variance $JUMP_VARIANCE --max_episodes $MAX_EPISODES --max_episode_len $MAX_EPISODE_LEN --special_string $SPECIAL_STRING
	done
done

python main.py --vary_jump_size 75 --window_cap $WINDOW_CAP --vary_informed $VARY_INFORMED --vary_jump_prob $VARY_JUMP_PROB --agent_type $AGENT_TYPE --AMM_type $AMM_TYPE --jump_mode $JUMP_MODE --noise_variance $NOISE_VARIANCE --jump_variance $JUMP_VARIANCE --max_episodes $MAX_EPISODES --max_episode_len $MAX_EPISODE_LEN --special_string $SPECIAL_STRING