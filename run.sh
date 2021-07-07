export TASK_NAME=ner
export DATA_DIR='/home/ray/project/AnswerExtract/dataset/cmrc'

accelerate launch run.py \
  --model_name_or_path bert-base-chinese \
  --train_file $DATA_DIR/cmrc_tc_train.json \
  --validation_file $DATA_DIR/cmrc_tc_dev.json \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /output/cmrc \
  --pad_to_max_length