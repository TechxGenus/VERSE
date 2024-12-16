Run the following program to inject the conversation template into the model:

```bash
python train/prepare_conversation.py --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base --save_path train/formatted-deepseek-coder-1.3b-base
```

Run the following program to train the model:

```bash
MODEL=$1
DATA=$2
OUTPUT=$3

python training.py \
   --model_name_or_path $MODEL \
   --data_path $DATA \
   --model_max_length 4096 \
   --per_device_train_batch_size 32 \
   --learning_rate 2e-5 \
   --weight_decay 0 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 1 \
   --gradient_checkpointing \
   --lr_scheduler_type cosine \
   --warmup_steps 15 \
   --seed 1234 \
   --output_dir $OUTPUT \
   --bf16 True \
   --evaluation_strategy "no" \
   --save_strategy "epoch" \
   --load_best_model_at_end False \
   --save_total_limit 1000 \
   --warmup_ratio 0.0 \
   --logging_steps 20 \
   --tf32 True \
   --optim adafactor
```
