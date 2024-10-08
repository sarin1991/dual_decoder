accelerate launch main.py \
  --pretrained_model "mistralai/Mistral-7B-Instruct-v0.3" \
  --config_path "config_180M_1B_opt.json" \
  --output_dir "mistral_out_180M_1B" \
  --dataset_text_field="text" \
  --max_steps=10000 \
  --gradient_checkpointing=True \
  --logging_steps=100 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --per_device_train_batch_size "128" \
  --learning_rate 2e-4 \
  --logging_dir 'logs' \
  --max_seq_length 512 \
  --bf16 True \
  --save_steps 10000 \
  --lr_scheduler_type cosine \
  --packing True \
  --model_output_path "mistral_checkpoint_180M_1B"
