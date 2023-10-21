#!/bin/bash
OBJECTS=("church" "parachute" "tench" "garbage_truck")
METHODS=("ESD" "forget_me")
TASKS=("classifier")

START=0
END=49
GPU_START=0  # GPU starting ID
GPU_END=7    # GPU ending ID

# Helper function to check GPU availability
check_gpu() {
    GPU_ID=$1
    nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader | wc -l
}

# Main loop
for obj in "${OBJECTS[@]}"; do
    for method in "${METHODS[@]}"; do
        for task in "${TASKS[@]}"; do
            for idx in $(seq $START $END); do
                # Find an available GPU
                GPU=-1
                while [[ $GPU -lt $GPU_START ]]; do
                    for (( i=$GPU_START; i<=$GPU_END; i++ )); do
                        if [[ $(check_gpu $i) -eq 0 ]]; then
                            GPU=$i
                            break
                        fi
                    done
                    # If all GPUs are busy, sleep for a while before checking again
                    if [[ $GPU -lt $GPU_START ]]; then
                        sleep 5
                    fi
                done

                # Run the job on the available GPU
                CUDA_VISIBLE_DEVICES=$GPU python src/execs/attack.py \
                    --task.criterion l2 \
                    --config-file configs/object/$task.json \
                    --task.target_ckpt files/pretrained/"$method"_ckpt/$obj.pt \
                    --task.concept $obj \
                    --task.dataset_path files/object_dataset/"$method"_"$obj"_attack \
                    --attacker.attack_idx $idx \
                    --attacker.k 3 \
                    --attacker.insertion_location prefix_k \
                    --logger.name attack_idx_"$idx" \
                    --logger.json.root files/results/object/text_grad_$task/$method/$obj &  # Run in the background

                # Sleep for a bit to make sure the job starts
                sleep 5
            done
            echo "All $task $obj jobs completed."
        done
        echo "All $obj jobs for $method completed."
    done
    echo "All $obj jobs for all methods completed."
done
# Wait for all background jobs to complete
wait
echo "All jobs completed."