source env_set.sh

nohup python -u train_image_classifier.py \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR \
  --checkpoint_path=$CHECKPOINT_PATH \
  --model_name=inception_v4 \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
  --train_dir=$TRAIN_DIR \
  --learning_rate=0.001 \
  --learning_rate_decay_factor=0.76\
  --num_epochs_per_decay=50 \
  --moving_average_decay=0.9999 \
  --optimizer=adam \
  --ignore_missing_vars=True \
  --batch_size=32 > output.log 2>&1 &


