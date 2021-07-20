CURRENT_DIR=`pwd`
export DATA_DIR=$CURRENT_DIR/dataset

python run_bert_qa.py \
  --model_name_or_path uer/roberta-base-chinese-extractive-qa \
  --do_train \
  --do_lower_case \
  --train_file $DATA_DIR/qa_train.json \
  --predict_file $DATA_DIR/qa_dev.json \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 128 \
  --doc_stride 64 \
  --output_dir output/temp \
  --gradient_accumulation_steps 4 \
  --early_stop_epochs 3 \
  --max_train_epochs 50 \
  --overwrite_output_dir