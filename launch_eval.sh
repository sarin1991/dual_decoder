accelerate launch -m \
    lighteval accelerate \
    --model_args "pretrained=<path to model>" \
    --tasks "eval_tasks.txt" \
    --override_batch_size 32 \
    --output_dir="./evals/"