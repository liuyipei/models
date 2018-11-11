# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf

from official.resnet import resnet_model
from official.utils.flags import core as flags_core
from official.utils.export import export
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
# pylint: enable=g-bad-import-order


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    examples_per_epoch: The number of examples in an epoch.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  print("process_record_dataset batch_size", batch_size)
  print("process_record_dataset shuffle_buffer", shuffle_buffer)
  print("process_record_dataset num_epochs", num_epochs)
  print("process_record_dataset num_gpus", num_gpus)
  print("process_record_dataset examples_per_epoch", examples_per_epoch)

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


def get_synth_input_fn(height, width, num_channels, num_classes):
  """Returns an input function that returns a dataset with zeroes.

  This is useful in debugging input pipeline performance, as it removes all
  elements of file reading and image preprocessing.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):  # pylint: disable=unused-argument
    return model_helpers.generate_synthetic_data(
        input_shape=tf.TensorShape([batch_size, height, width, num_channels]),
        input_dtype=tf.float32,
        label_shape=tf.TensorShape([batch_size]),
        label_dtype=tf.int32)

  return input_fn


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = 0.1 * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale,
                    loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
                    batch_norm_dict=None):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  features = tf.cast(features, dtype)

  model = model_class(resnet_size, data_format, resnet_version=resnet_version,
                      dtype=dtype, batch_norm_dict=batch_norm_dict)

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  ce_batch_subset = batch_norm_dict.get('ce_batch_subset', None)
  if ce_batch_subset is None:
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
  else:
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits[:ce_batch_subset], labels=labels[:ce_batch_subset])

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('train_cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return ('batch_normalization' not in name) and ('bfn_running' not in name) # batch free norm
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
    # loss is computed using fp32 for numerical stability.
    [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
     if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if batch_norm_dict is not None:  # also regularize inputs
    bfn_input_decay = batch_norm_dict.get('bfn_input_decay', None)
    if bfn_input_decay is not None and bfn_input_decay > 0:
      bfn_input_decay_losses = tf.get_collection('bfn_input_decay_losses')
      bfnd_input_decay_losses = tf.get_collection('bfnd_inputs_decay_losses')
      inputs_decay_losses = bfn_input_decay_losses + bfnd_input_decay_losses
      print(' %d + %d input_decay_losses' % (len(bfn_input_decay_losses), len(bfnd_input_decay_losses)), 
          [n.name for n in bfn_input_decay_losses])
      l2_bfn_loss = bfn_input_decay * tf.add_n([tf.reduce_mean(_) for _ in inputs_decay_losses])
      tf.summary.scalar('l2_bfn_loss', l2_bfn_loss)
      loss += l2_bfn_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)
    # Create a tensor named learning_rate for logging purposes
    learning_rate = tf.identity(learning_rate, name='learning_rate')
    log10_learning_rate = tf.identity(tf.log(learning_rate) / tf.log(10.), name='log10_learning_rate')
    tf.summary.scalar('log10_learning_rate', log10_learning_rate)
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in all_variables:
      v_abs = tf.cast(tf.abs(v), dtype=tf.float32) + tf.constant(1e-6, dtype=tf.float32)
      rep_name = re.sub(':', '_', v.name)
      if 'variance' in rep_name:
        v_minlogabs = tf.reduce_min(tf.log(v_abs))
        tf.summary.scalar('.'.join(['minlogabs_variance/', rep_name]), v_minlogabs)
      elif 'ygrad' in rep_name:
        v_maxlogabs = tf.reduce_max(tf.log(v_abs))
        tf.summary.scalar('.'.join(['maxlogabs_ygrads/', rep_name]), v_maxlogabs)
      else:
        continue

    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )

    #if loss_scale != 1:
    # When computing fp16 gradients, often intermediate tensor values are
    # so small, they underflow to 0. To avoid this, we multiply the loss by
    # loss_scale to make these tensor values loss_scale times bigger.
    scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

    # Once the gradient computation is complete we can scale the gradients
    # back to the correct scale before passing them to the optimizer.
    unscaled_grad_vars = [(grad / loss_scale, var)
                          for grad, var in scaled_grad_vars]
    print('loss_scale', loss_scale)
    # print(unscaled_grad_vars)
    #for i, curr_var in enumerate(unscaled_grad_vars):
      # print(i, curr_var)
      #optimizer.apply_gradients([curr_var], global_step)
    minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    #else:
    #  minimize_op = optimizer.minimize(loss, global_step)

    # for regularized_bn, the gradient update ops are generated relative to the creation of the minimize_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('update_ops len', len(update_ops))
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(labels, predictions['classes'])
  eval_metrics = {'accuracy': accuracy,
                  'cross_entropy': tf.metrics.mean(cross_entropy)}

  # Create a tensor named train_accuracy for logging purposes
  train_accuracy = tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', train_accuracy)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics)


def resnet_main(
    flags_obj, model_function, input_function, dataset_name, shape=None):
  """Shared main loop for ResNet Models.

  Args:
    flags_obj: An object containing parsed flags. See define_resnet_flags()
      for details.
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
    dataset_name: the name of the dataset for training and evaluation. This is
      used for logging purpose.
    shape: list of ints representing the shape of the images used for training.
      This is only used if flags_obj.export_dir is passed.
  """

  model_helpers.apply_clean(flags.FLAGS)
  print("flags_obj.batch_size", flags_obj.batch_size)

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Create session config based on values of inter_op_parallelism_threads and
  # intra_op_parallelism_threads. Note that we default to having
  # allow_soft_placement = True, which is required for multi-GPU and not
  # harmful for other modes.
  session_config = tf.ConfigProto(
      inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
      allow_soft_placement=True)

  distribution_strategy = distribution_utils.get_distribution_strategy(
      flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

  run_config = tf.estimator.RunConfig(
      train_distribute=distribution_strategy, session_config=session_config)

  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
      params={
          'resnet_size': int(flags_obj.resnet_size),
          'data_format': flags_obj.data_format,
          'batch_size': flags_obj.batch_size,
          'resnet_version': int(flags_obj.resnet_version),
          'loss_scale': flags_core.get_loss_scale(flags_obj),
          'dtype': flags_core.get_tf_dtype(flags_obj),
          'batch_norm_method': flags_obj.batch_norm_method,
          'bfn_mmfwd': flags_obj.mmfwd,
          'bfn_mmgrad': flags_obj.mmgrad,
          'vd_weights': flags_obj.vd_weights,
          'bfn_grad_clip': flags_obj.bfn_grad_clip,
          'rvst': flags_obj.rvst,
          'bfn_input_decay': flags_obj.bfn_input_decay,
          'batch_denom': flags_obj.batch_denom,
          'opt_mm': flags_obj.opt_mm,
          'bepm': flags_obj.bepm,
      })

  run_params = {
      'batch_size': flags_obj.batch_size,
      'dtype': flags_core.get_tf_dtype(flags_obj),
      'resnet_size': flags_obj.resnet_size,
      'resnet_version': flags_obj.resnet_version,
      'loss_scale': flags_core.get_loss_scale(flags_obj),
      'synthetic_data': flags_obj.use_synthetic_data,
      'train_epochs': flags_obj.train_epochs,
      'batch_norm_method': flags_obj.batch_norm_method,
      'bfn_mmfwd': flags_obj.mmfwd,
      'bfn_mmgrad': flags_obj.mmgrad,
      'vd_weights': flags_obj.vd_weights,
      'bfn_grad_clip': flags_obj.bfn_grad_clip,
      'rvst': flags_obj.rvst,
      'bfn_input_decay': flags_obj.bfn_input_decay,
      'batch_denom': flags_obj.batch_denom,
      'opt_mm': flags_obj.opt_mm,
      'bepm': flags_obj.bepm,
  }
  if flags_obj.use_synthetic_data:
    dataset_name = dataset_name + '-synthetic'

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('resnet', dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      model_dir=flags_obj.model_dir,
      batch_size=flags_obj.batch_size)

  def input_fn_train():
    return input_function(
        is_training=True, data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=flags_obj.epochs_between_evals,
        num_gpus=flags_core.get_num_gpus(flags_obj))

  def input_fn_eval():
    return input_function(
        is_training=False, data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=1)

  total_training_cycle = (flags_obj.train_epochs //
                          flags_obj.epochs_between_evals)
  for cycle_index in range(total_training_cycle):
    tf.logging.info('Starting a training cycle: %d/%d',
                    cycle_index, total_training_cycle)

    classifier.train(input_fn=input_fn_train, hooks=train_hooks,
                     max_steps=flags_obj.max_train_steps)

    tf.logging.info('Starting to evaluate.')

    # flags_obj.max_train_steps is generally associated with testing and
    # profiling. As a result it is frequently called with synthetic data, which
    # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
    # eval (which is generally unimportant in those circumstances) to terminate.
    # Note that eval will run for max_train_steps each loop, regardless of the
    # global_step count.
    eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                       steps=flags_obj.max_train_steps)

    benchmark_logger.log_evaluation_result(eval_results)

    if model_helpers.past_stop_threshold(
        flags_obj.stop_threshold, eval_results['accuracy']):
      break

  if flags_obj.export_dir is not None:
    # Exports a saved model for the given classifier.
    input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
        shape, batch_size=flags_obj.batch_size)
    classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)


def define_resnet_flags(resnet_size_choices=None):
  """Add flags and validators for ResNet."""
  flags_core.define_base()
  flags_core.define_performance(num_parallel_calls=False)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_enum(
    name='resnet_version', short_name='rv', default='2',
    enum_values=['1', '2'],
    help=flags_core.help_wrap(
      'Version of ResNet. (1 or 2) See README.md for details.'))

  #flags.DEFINE_enum(
  #  name='batch_norm_method', short_name='bnmethod', default='tf_layers_regular',
  #  enum_values=[
  #    'tf_layers_regular', 'tf_layers_renorm', 'identity',
  #    'batch_free_normalization_sigfunc',
  #    'batch_free_normalization_sigconst',
  #    'batch_free_normalization_sigfunc_compare_running_stats',
  #    'batch_free_direct',
  #    'bfn_like_regular',
  #    ],
  #  help=flags_core.help_wrap(
  #    'batch norm method string'))
  flags.DEFINE_string(name='batch_norm_method', short_name='bnmethod', default='tf_layers_regular', help='batch norm method string')
  flags.DEFINE_float(
    name='mmfwd', default=.99, help="bfn momentum for forward moments", lower_bound=0., upper_bound=1.)
  flags.DEFINE_float(
    name='mmgrad', default=.99, help="bfn momentum for grad moments", lower_bound=0., upper_bound=1.)
  flags.DEFINE_float(
    name='vd_weights', default=None, help="regularized bn: virtual data weights")
  flags.DEFINE_float(
    name='bfn_grad_clip', default=None, help="bfn gradient norm clipping (marginalizes away C)", lower_bound=1e-5)
  flags.DEFINE_float(
    "batch_denom", default=None, help="batch_denom in determining learning rate (was originally 128 for cifar or 256 for imagenet)")
  flags.DEFINE_float(
    "opt_mm", default=0.9, help="momentum optimizer momentum")
  flags.DEFINE_float(
    "bepm", default=1., help="Boundary epoch multiplier relative to default repo settings")
  flags.DEFINE_enum(
    "rvst", default=None, enum_values=[
      'uniform_max1',
      'uniform_near1',
      'lognorm_max1',
      'lognorm_near1',], help="random variance scaling type for bfn")
  flags.DEFINE_float(
    "bfn_input_decay", default=None, help="bfn_input_decay is l2 loss coefficient on activations that are being input into batch-free normalization")
  choice_kwargs = dict(
    name='resnet_size', short_name='rs', default='50',
    help=flags_core.help_wrap('The size of the ResNet model to use.'))

  if resnet_size_choices is None:
    flags.DEFINE_string(**choice_kwargs)
  else:
    flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)

  # The current implementation of ResNet v1 is numerically unstable when run
  # with fp16 and will produce NaN errors soon after training begins.
  msg = ('ResNet version 1 is not currently supported with fp16. '
         'Please use version 2 instead.')
  @flags.multi_flags_validator(['dtype', 'resnet_version'], message=msg)
  def _forbid_v1_fp16(flag_values):  # pylint: disable=unused-variable
    return (flags_core.DTYPE_MAP[flag_values['dtype']][0] != tf.float16 or
            flag_values['resnet_version'] != '1')
