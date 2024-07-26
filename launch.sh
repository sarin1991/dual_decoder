python main.py \
  --pretrained_model "mistralai/Mistral-7B-Instruct-v0.3" \
  --output_dir "mistral_out" \
  --num_train_epochs=1 \
  --gradient_checkpointing=True \
  --logging_steps=10 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --per_device_train_batch_size "32" \
  --learning_rate 2e-4 \
  --dataloader_num_workers 0 \
  --logging_dir 'logs' \
  --max_seq_length 4096 \
  --bf16 True \
  --save_steps 500 \
  --lr_scheduler_type cosine \
  --model_output_path "mistral_checkpoint"
  