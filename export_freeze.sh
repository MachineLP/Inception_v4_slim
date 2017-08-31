source env_set.sh
python -u export_inference_graph.py \
  --model_name=inception_v4 \
  --output_file=./my_inception_v4.pb \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR


NEWEST_CHECKPOINT=$(ls -t1 $TRAIN_DIR/model.ckpt*| head -n1)
NEWEST_CHECKPOINT=${NEWEST_CHECKPOINT%.*}
python -u ~/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=my_inception_v4.pb \
  --input_checkpoint=$NEWEST_CHECKPOINT \
  --output_graph=./my_inception_v4_freeze.pb \
  --input_binary=True \
  --output_node_name=InceptionV4/Logits/Predictions

cp $DATASET_DIR/labels.txt ./my_inception_v4_freeze.label
