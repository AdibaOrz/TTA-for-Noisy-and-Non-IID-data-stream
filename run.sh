#!/bin/bash

# Array of seeds
seeds=(0 1 2 3 4)
# Array of weighted options
weighted_options=(0 1)  # 0 for False, 1 for True

# Your conda environment name
ENV_NAME="SimpleTTA"

# Function to create a unique session name
get_session_name() {
    local seed=$1
    local weighted=$2
    echo "note_seed${seed}_weighted${weighted}"
}

# Run experiments for each seed and weighted option
for seed in "${seeds[@]}"; do
    for weighted in "${weighted_options[@]}"; do
        session_name=$(get_session_name "$seed" "$weighted")
        
        # Create new tmux session detached
        tmux new-session -d -s "$session_name"
        
        # First initialize conda and activate environment
        tmux send-keys -t "$session_name" "conda activate $ENV_NAME" C-m
        
        # Construct the command with weighted loss flag if needed
        weighted_flag=""
        if [ "$weighted" -eq 1 ]; then
            weighted_flag="--weighted_loss"
        fi
        
        # Send both commands to the session, the second will run after the first completes
        tmux send-keys -t "$session_name" "python main.py --method NOTE --iabn --tgt_use_learned_stats --optimize --adapt --device cuda:$seed --distribution dirichlet --seed $seed --use_checkpoint --conf_thresh 0.0 $weighted_flag" C-m
        tmux send-keys -t "$session_name" "python main.py --method NOTE --iabn --tgt_use_learned_stats --optimize --adapt --device cuda:$seed --distribution random --seed $seed --use_checkpoint --conf_thresh 0.0 $weighted_flag" C-m
        
        echo "Started session: $session_name on GPU $seed with weighted loss: $weighted"
    done
done