export TASK_NAME=ner
export DATA_DIR='/home/ray/project/AnswerExtract/dataset/cmrc'

accelerate launch run_bert_crf.py \
  --model_name_or_path bert-base-chinese \
  --do_train \
  --train_file $DATA_DIR/cmrc_tc_sen_train.json \
  --validation_file $DATA_DIR/cmrc_tc_sen_dev.json \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 100 \
  --output_dir output/cmrc