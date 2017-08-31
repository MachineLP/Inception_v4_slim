source env_set.sh
python -u eval_image_classifier.py \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR \
  --dataset_split_name=validation \
  --model_name=inception_v4 \
  --checkpoint_path=$TRAIN_DIR \
  --eval_dir=/tmp/eval/validation \
  --eval_interval_secs=60 \
  --batch_size=32 

