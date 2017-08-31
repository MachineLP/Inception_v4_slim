bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=/home/lwp/source/models/slim/my_inception_v4_freeze.pb \
--out_graph=/home/lwp/source/models/slim/my_inception_v4_freeze_opt.pb \
--inputs='input' \
--outputs='InceptionV4/Logits/Predictions' \
--transforms='
  strip_unused_nodes(type=float, shape="1,299,299,3")
  remove_nodes(op=Identity, op=CheckNumerics)
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms'
