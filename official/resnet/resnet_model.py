## Modified by Yi Liu to evaluate Batch free normalization
# Yi Liu liu.yi.pei@gmail.com 
# 

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
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DEFAULT_METHOD = 'tf_layers_regular'
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def old_batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def baseline_fallback(scope, batch_norm_method):
  # block_layer / block_fn / bn01
  if 'fast' in batch_norm_method:
    if 'bn0' not in scope or not re.match('.*/block_fn_002/.*', scope): 
      return True
  if 'block2' in batch_norm_method:
    if not re.match('.*/block_fn_002/.*', scope): 
      return True
  return False

def parse_lsqrn_Gr(batch_norm_method, C):
    # regular lsqrn: *_G_r
    # shared lsqrn:  *_G_i
    _, c, I = batch_norm_method.split('_')
    c, I = int(c), int(I)
    G = C // c # c must divide C, due to reshape
    r = c - I # ok to round here
    return G, r

def moment_normalize(x, axes, eps, **kwargs):
  u, v = tf.nn.moments(x, axes, keep_dims=True, **kwargs)
  return (x - u) * tf.rsqrt(v + eps)

def group_norm(x, G, eps=1e-5): # group_norm(x, gamma, beta, G, eps=1e−5):
  # x: input features with shape [N,C,H,W]
  # gamma, beta: scale and offset, with shape [1,C,1,1]
  # G: number of groups for GN
  # https://arxiv.org/pdf/1803.08494.pdf
  N, C, H, W = x.get_shape().as_list()
  N = -1
  print([N, C, H, W])
  x = tf.reshape(x, [N, G, C // G, H, W])
  mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
  x = (x - mean) / tf.sqrt(var + eps)
  x = tf.reshape(x, [N, C, H, W])
  return x

def batch_norm(inputs, training, data_format, batch_norm_dict=None):
  import batch_free_normalization.python.regularized_bn as regularized_bn
  # batch_norm_method=None, bfn_mmfwd=None, bfn_mmgrad=None
  print(batch_norm_dict)
  if batch_norm_dict is None:
    batch_norm_method = _BATCH_NORM_DEFAULT_METHOD
  else:
    batch_norm_method = batch_norm_dict['batch_norm_method']
  print('batch_norm_method', batch_norm_method)

  shape = tf.shape(inputs)
  shape_list = inputs.get_shape().as_list()
  C = shape_list[1] if data_format == 'channels_first' else shape_list[3]
  def _baseline_bn():
    return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True,
      renorm=False)
  if batch_norm_method=='tf_layers_regular' or batch_norm_method.startswith('sharedlsqrn'):
    return _baseline_bn()
  elif batch_norm_method.startswith('switch'):
    scope = tf.get_variable_scope().name
    if baseline_fallback(scope, batch_norm_method):
      # block_fn name is 1-based / faster running time; use fast baseline only for 
      print("Use baseline bn for {}".format(scope))
      return _baseline_bn()
    else:
      import external.lsqr_normalization.lsqr_norm as lsqr_norm
      switch_bn_treatment = 'lsqrn' if 'lsqrn' in batch_norm_method else 'base'
      print("switch_bn_treatment: {}".format(switch_bn_treatment))
      G, r = parse_lsqrn_Gr(batch_norm_method, C) if switch_bn_treatment == 'lsqrn' else [None, None]
      switchnorm_val, _ = lsqr_norm.switch_normalization(
        inputs, Gr = [G, r], training=training, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, data_format=data_format,
        bn_treatment=switch_bn_treatment)
      return switchnorm_val
  elif batch_norm_method.startswith('lsqrn'):
    scope = tf.get_variable_scope().name
    if baseline_fallback(scope, batch_norm_method):
      # block_fn name is 1-based / faster running time; use fast baseline only for 
      print("Use baseline bn for {}".format(scope))
      return _baseline_bn()
    # The first bn call of the second residual-block function in each major-block
    if '-mHW' in batch_norm_method:
      marginalize = 'HW'
    elif '-mrBHW' in batch_norm_method:
      marginalize = 'runBHW'
    else:
      marginalize = 'BHW'
    print('marginalize str = {}'.format(marginalize))
    G, r = parse_lsqrn_Gr(batch_norm_method, C)
    import external.lsqr_normalization.lsqr_norm as lsqr_norm
    inputs_lsqrn, helper_dict = lsqr_norm.lsqrn(inputs, Gr=[G, r], BHWC=None, 
      name_dict=None, marginalize=marginalize,
      training=training, momentum=_BATCH_NORM_DECAY, data_format=data_format, epsilon=_BATCH_NORM_EPSILON,)
    return inputs_lsqrn
  elif batch_norm_method.startswith('tf_sequence'):
    stshp = inputs.get_shape().as_list() # stshp stands for static shape; may contain Nones;
    for _ in stshp[1:]:
      assert _ is not None
    channel_axis = 1 if data_format == 'channels_first' else 3
    batch_axis = 0
    seq_string = batch_norm_method.split('_')[-1].upper()
    one = tf.ones(shape=[], dtype=tf.int32)
    # CuDNN has constraints on batch size (dim 0) -- can't be too large -- hence different patterning for 0 and 1
    dict_reshape = {  # dim 1 is the accumulator dimension in tf.layers.batch_normalization
      '0': tf.stack([shape[0],             one,                 shape[1], shape[2] * shape[3]]), # accumulate ...
      '2': tf.stack([shape[0] * shape[1],  shape[2],            one, shape[3]]),         # accumulate .H.
      '3': tf.stack([shape[0] * shape[1],  shape[2] * shape[3], one, one]),     # accumulate .HW
      '6': tf.stack([shape[0],             shape[1] * shape[2], one, shape[3]]),         # accumulate CH.
      # 4: C..  standard batch normalization
      # 1: ..W. batch size too large for cudnn to handle: special case  -- transpose into 2
      # 5: C.W  discontiguous accumulator dims: special case  -- transpose into 6
      # 7: ...  cannot normalize.
    }

    dict_stshp = {  # dim 1 is the accumulator dimension in tf.layers.batch_normalization
        '0': [None, 1,                   stshp[1], stshp[2] * stshp[3]], # accumulate ...
        '2': [None, stshp[2],            1, stshp[3]],                   # accumulate .H.
        '3': [None, stshp[2] * stshp[3], 1, 1],                          # accumulate .HW
        '6': [None, stshp[1] * stshp[2], 1, stshp[3]]                    # accumulate CH.
        # 4: C..  standard batch normalization
        # 1,5; transpose into BCWC and reduce to 2,6 respectively. CuDNN batch size constraint; incontiguous accumulator rank indices  
        # 7: ...  cannot normalize.
    }
    
    def partition_normalization(s, inputs):
      assert channel_axis == 1
      if s != '4': # 4 means standard BN (without affine downstream here)
        inputs = tf.reshape(inputs, dict_reshape[s]) # no transpose necessary
        inputs.set_shape(dict_stshp[s])
      inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=channel_axis,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=False,
        scale=False, training=training, fused=True,
        renorm=False)
      if s != '4':
        inputs = tf.reshape(inputs, shape)
      return inputs

    for i, s in enumerate(seq_string):
      if s == 'B':
        inputs = tf.layers.batch_normalization(
          inputs=inputs, axis=channel_axis,
          momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=False,
          scale=False, training=training, fused=True,
          renorm=False)
      elif s == 'L':
        inputs = moment_normalize(inputs, axes=sorted(set(range(4)) - set([batch_axis])), eps=_BATCH_NORM_EPSILON)
      elif s == 'I':
        inputs = moment_normalize(inputs, axes=sorted(set(range(4)) - set([batch_axis, channel_axis])), eps=_BATCH_NORM_EPSILON)
      elif s == 'G':
        assert channel_axis == 1
        inputs = group_norm(inputs, G=8, eps=_BATCH_NORM_EPSILON) # some layers have 16 channels
      elif s in '0236': 
        # imagine accumulator shapes as big endian flags for CHW respectively -- interpret in binary
        with tf.variable_scope('accum%s' % s):
          inputs = partition_normalization(s, inputs)
      elif s in '15': # accumulator of shape C.W ( big endian CHW accumulator status in binary = 5)
                      # special case because accumulator dimension is not contiguous -- needs transpose
                      # also for accumulator ..W -- batch size would be too large using the 01236 method
        with tf.variable_scope('accum%s' % s):
          swap_H_and_W = [0, 1, 3, 2]
          inputs = tf.transpose(inputs, swap_H_and_W) # B, C, W, H
          if s == '1': # accumulate ..W
            inputs = partition_normalization('2', inputs)
          elif s == '5': # accumulate C.W
            inputs = partition_normalization('6', inputs)
          else:
            raise ValueError('s=%s. not in [15]' %s)
          inputs = tf.transpose(inputs, swap_H_and_W) # invert back
      else:
        raise ValueError('Unknown batch normalization seqstring character', s)
    out = regularized_bn.beta_gamma(inputs, resulting_axes=[channel_axis])
    return out
  elif batch_norm_method=='tf_layers_renorm':
    return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True,
      renorm=True)
  elif batch_norm_method.startswith('batch_free_normalization') or batch_norm_method.startswith('bfn_like_'):
    import batch_free_normalization.python.bfn as bfn
    treat_sigma_const = not ('_sigfunc' in batch_norm_method)
    print("treat_sigma_const", treat_sigma_const)
    # backward updates are baked into the graph with control deps
    resulting_axes = [1] if data_format == 'channels_first' else [3]
    global_step = tf.train.get_or_create_global_step()
    s = tf.cast(global_step, dtype=tf.float64)
    raw_momentum = (s + 1.) / (s + 2.)
    rvst = batch_norm_dict.get('rvst', None)
    rvs = None if rvst is None else (batch_norm_dict['rvst'], batch_norm_dict.get('rvsv', None))
    bfn_input_decay_losses = tf.reduce_mean(inputs ** 2)
    tf.add_to_collection(name='bfn_input_decay_losses', value=bfn_input_decay_losses)

    ivar_type_convergence=None
    if "ivtc_batch" in batch_norm_method:
      ivar_type_convergence='batch' # deault
    elif "ivtc_loo" in batch_norm_method:
      ivar_type_convergence='loo' # only compatible with bfn_like_loo
    elif "ivtc_running" in batch_norm_method:
      ivar_type_convergence='running'
    loo_axis=None
    
    if batch_norm_method.startswith('bfn_like_regular'): # no ramp
      use_inf_accum = True
      eps = _BATCH_NORM_EPSILON
      momentum = batch_norm_dict['bfn_mmfwd']
      grad_momentum = batch_norm_dict['bfn_mmgrad']
    elif batch_norm_method.startswith('bfn_like_loo'):
      use_inf_accum = True
      eps = _BATCH_NORM_EPSILON
      momentum = batch_norm_dict['bfn_mmfwd']
      grad_momentum = batch_norm_dict['bfn_mmgrad']
      loo_axis=0
    else:
      # older batch_free_normalization formulation
      use_inf_accum = False 
      eps = 1e-3
      momentum=tf.minimum(raw_momentum, batch_norm_dict['bfn_mmfwd'])
      grad_momentum=tf.minimum(raw_momentum, batch_norm_dict['bfn_mmgrad'])
    retval = bfn.bfn_beta_gamma(
      inputs,
      treat_sigma_const=treat_sigma_const,
      resulting_axes=resulting_axes,
      momentum=momentum,
      grad_momentum=grad_momentum,
      eps=eps,
      grad_clip=batch_norm_dict.get('bfn_grad_clip', None),
      random_variance_scaling=rvs,
      training=training,
      use_inf_accum=use_inf_accum,
      loo_axis=loo_axis,
      ivar_type_convergence=ivar_type_convergence)
    if batch_norm_method.endswith('compare_running_stats'):
      with tf.variable_scope('compare_running_stats'):
        #make_baseline_accumulators_as_side_effect = \
        tf.layers.batch_normalization(
          inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
          momentum=_BATCH_NORM_DECAY, epsilon=eps, center=True,
          scale=True, training=training, fused=True)
    return retval
  elif batch_norm_method.startswith('batch_free_direct'):
    import batch_free_normalization.python.bfn_direct as bfn_direct
    resulting_axes = [1] if data_format == 'channels_first' else [3]
    global_step = tf.train.get_or_create_global_step()
    s = tf.cast(global_step, dtype=tf.float64)
    raw_momentum = (s + 1.) / (s + 2.)

    retval = bfn_direct.bfnd_beta_gamma(
      inputs,
      resulting_axes=resulting_axes,
      momentum=tf.minimum(raw_momentum, batch_norm_dict['bfn_mmfwd']),
      eps=_BATCH_NORM_EPSILON,
      training=training)
    return retval
  elif batch_norm_method.startswith('regularized_bn'):
    resulting_axes = [1] if data_format == 'channels_first' else [3]
    momentum = batch_norm_dict['bfn_mmfwd']
    grad_momentum = batch_norm_dict['bfn_mmgrad']
    vd_weights = batch_norm_dict.get('vd_weights', None)
    zero_virtual_grad = (batch_norm_method == 'regularized_bn_zero')
    y, running_stats, updates_dict = regularized_bn.regularized_batch_norm(
      inputs, resulting_axes=resulting_axes,
      momentum=momentum, grad_momentum=grad_momentum,
      eps=_BATCH_NORM_EPSILON,
      virtual_data_weights=vd_weights, training=training,
      zero_virtual_grad=zero_virtual_grad)
    y2 = regularized_bn.beta_gamma(y, resulting_axes=resulting_axes)
    return y2
    # add to collections need ot happen after optimizer is made
  elif batch_norm_method=='identity':
    return tf.identity(inputs)
  else:
    raise NotImplementedError("Unknown batch_norm_method %s" % batch_norm_method)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format, batch_norm_dict=None):
  """A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format, batch_norm_dict=batch_norm_dict)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format, batch_norm_dict=None):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  with tf.variable_scope('bn0'):
    inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    with tf.variable_scope('proj0'):
      shortcut = projection_shortcut(inputs)

  with tf.variable_scope('conv1'):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

  with tf.variable_scope('bn1'):
    inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
  inputs = tf.nn.relu(inputs)
  with tf.variable_scope('conv2'):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

  return inputs + shortcut



def _shared_lsqrn_all_in_one(inputs, shortcut, training, batch_norm_method):
  raise ValueError("stop using _shared_lsqrn_all_in_one")
  import external.lsqr_normalization.lsqr_norm as lsqr_norm
  _, [B, H, W, C] = lsqr_norm.as_BHWC(shortcut, data_format='channels_last')
  _, c, I = batch_norm_method.split('_') # 
  c, I = int(c), int(I)
  del I #  ignore I
  if '-use1g' in batch_norm_method:
    G = 1 
    explanatory_BHWGc = tf.reshape(shortcut[:, :, :, :c], [B, H, W, 1, c])
    print('%s type: %s' % (batch_norm_method, '-use1g'))
  else:
    G = C // c
    assert C == c * G # c = channels per group. ignore I
    explanatory_BHWGc = tf.reshape(shortcut, [B, H, W, G, c])

  marginalize, l2_regularizer = 'BHW', 1e-3
  if '-mHW' in batch_norm_method:
    marginalize = 'HW'
    # marginalize, l2_regularizer = 'HW', 1e-1
  if '-l2tr' in batch_norm_method:
    mult_trace_to_l2_reg = 1e-2
    # mult_trace_to_l2_reg, floor_trace_to_l2_reg = None, 1e-2
  else:
    mult_trace_to_l2_reg = None

  pinv_BHWGc = lsqr_norm.get_pinv_BHWGi(explanatory_BHWGc, 
    l2_regularizer=l2_regularizer, marginalize=marginalize, mult_trace_to_l2_reg=mult_trace_to_l2_reg)
  z_BHWC, helper_dict = lsqr_norm.shared_lsqrn(
    inputs, G=G, explanatory_pinv_BHWGi=pinv_BHWGc, BHWC=[B, H, W, C],
    name_dict = None, training=training, momentum=_BATCH_NORM_DECAY, data_format='channels_last',
    bn_residuals=True, center_bn_residuals=True, scale_bn_residuals=True,
    marginalize=marginalize, mean_variance_path=False)
  return z_BHWC


def shared_lsqrn_bnmethod(inputs, shortcut, training, batch_norm_method):

  _, c, I = batch_norm_method.split('_') # 
  c, I = int(c), int(I)
  
  if '-l2tr' in batch_norm_method:
    mult_trace_to_l2_reg = 1e-2
  else:
    mult_trace_to_l2_reg = None

  if '-1g.per.HW.BHW' in batch_norm_method or '-fullgroup.per.HW.BHW' in batch_norm_method:
    # half the channels use 1 group of HW marginalization; the other half use 1 group of BHW
    use1g = '-1g.per.HW.BHW' in batch_norm_method
    print('use1g {}'.format(use1g))
    inputs0, inputs1 = tf.split(inputs, num_or_size_splits=2, axis=3)
    shortcut0, shortcut1 = tf.split(shortcut, num_or_size_splits=2, axis=3)
    with tf.variable_scope('marg_HW'):
      out0 = _shared_lsqrn_wrap(inputs0, shortcut0, training, c, 
          use1g=use1g, marginalize='BHW', mult_trace_to_l2_reg=mult_trace_to_l2_reg)
    with tf.variable_scope('marg_BHW'):
      out1 = _shared_lsqrn_wrap(inputs1, shortcut1, training, c, 
          use1g=use1g, marginalize='HW', mult_trace_to_l2_reg=mult_trace_to_l2_reg)
    return tf.concat([out0, out1], axis=3)
  else:
    use1g = '-use1g' in batch_norm_method
    print('use1g {}'.format(use1g))
    marginalize = 'BHW'
    if '-mHW' in batch_norm_method:
      marginalize = 'HW'
    return _shared_lsqrn_wrap(inputs, shortcut, training, c, use1g, marginalize, mult_trace_to_l2_reg)


def _shared_lsqrn_wrap(inputs, shortcut, training, i, use1g, marginalize, mult_trace_to_l2_reg):
  import external.lsqr_normalization.lsqr_norm as lsqr_norm
  _, [B, H, W, C] = lsqr_norm.as_BHWC(shortcut, data_format='channels_last')
  print("C {}, i {}, marginalize {}".format(C, i, marginalize))

  if use1g:
    G = 1 
    explanatory_BHWGi = tf.reshape(shortcut[:, :, :, :i], [B, H, W, 1, i])
  else: # each group uses its own set of explanatory channels
    c = i
    G = C // c
    assert C == i * G # c = channels per group. ignore I
    explanatory_BHWGi = tf.reshape(shortcut, [B, H, W, G, i])

  l2_regularizer = 1e-3

  pinv_BHWGc = lsqr_norm.get_pinv_BHWGi(explanatory_BHWGi, 
    l2_regularizer=l2_regularizer, marginalize=marginalize, mult_trace_to_l2_reg=mult_trace_to_l2_reg)
  z_BHWC, helper_dict = lsqr_norm.shared_lsqrn(
    inputs, G=G, explanatory_pinv_BHWGi=pinv_BHWGc, BHWC=[B, H, W, C],
    name_dict = None, training=training, momentum=_BATCH_NORM_DECAY, data_format='channels_last',
    bn_residuals=True, center_bn_residuals=True, scale_bn_residuals=True,
    marginalize=marginalize, mean_variance_path=False)
  return z_BHWC




def _shared_lsqrn_building_block_v2(
    inputs, filters, training, projection_shortcut, strides,
    data_format, batch_norm_dict=None):
  """
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  print('_shared_lsqrn_building_block_v2')
  assert batch_norm_dict is not None
  batch_norm_method = batch_norm_dict['batch_norm_method']
  assert data_format == 'channels_last' # channels_first not supported
  # import external.lsqr_normalization.lsqr_norm as lsqr_norm
  # shape_list = inputs.get_shape().as_list()

  shortcut = inputs
  with tf.variable_scope('bn0'):
    inputs = old_batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    with tf.variable_scope('proj0'):
      shortcut = projection_shortcut(inputs)

  with tf.variable_scope('conv1'):
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  
  with tf.variable_scope('bn-shared1'):
    z_BHWC = shared_lsqrn_bnmethod(inputs, shortcut, training, batch_norm_method)

  inputs = tf.nn.relu(z_BHWC)
  with tf.variable_scope('conv2'):
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut



def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format, batch_norm_dict=None):
  """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format, batch_norm_dict=batch_norm_dict)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format, batch_norm_dict=None):
  """A single block for ResNet v2, without a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  with tf.variable_scope('bottleblock_initial'):
    inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
    inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  with tf.variable_scope('bottleblock_conv1'):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)
  with tf.variable_scope('bottleblock_bn1'):
    inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
    inputs = tf.nn.relu(inputs)
  with tf.variable_scope('bottleblock_conv2'):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

  with tf.variable_scope('bottleblock_bn2'):
    inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
    inputs = tf.nn.relu(inputs)
  with tf.variable_scope('bottleblock_conv3'):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

  return inputs + shortcut


def _shared_lsqrn_bottleneck_block_v2(
    inputs, filters, training, projection_shortcut, strides,
    data_format, batch_norm_dict=None):
  """
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  print('_shared_lsqrn_bottleneck_block_v2')
  assert batch_norm_dict is not None
  batch_norm_method = batch_norm_dict['batch_norm_method']
  assert data_format == 'channels_last' # channels_first not supported

  shortcut = inputs
  with tf.variable_scope('bottleblock_initial'):
    inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
    inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  with tf.variable_scope('bottleblock_conv1'):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)
  with tf.variable_scope('bottleblock_bn1'):
    inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
    inputs = tf.nn.relu(inputs)
  with tf.variable_scope('bottleblock_conv2'):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

  with tf.variable_scope('bottleblock_bn2'):
    inputs = batch_norm(inputs, training, data_format, batch_norm_dict=batch_norm_dict)
    inputs = tf.nn.relu(inputs)
  with tf.variable_scope('bottleblock_conv3'):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

  with tf.variable_scope('bn-shared-final'):
    z_BHWC = shared_lsqrn_bnmethod(inputs, shortcut, training, batch_norm_method)

  return z_BHWC + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, batch_norm_dict=None):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  with tf.variable_scope('block_initial'):
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format, batch_norm_dict=batch_norm_dict)

  for _ in range(1, blocks):
    with tf.variable_scope('block_fn_%0.3d' % _):
      inputs = block_fn(inputs, filters, training, None, 1, data_format,
                        batch_norm_dict=batch_norm_dict)

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
               kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               block_sizes, block_strides,
               final_size, resnet_version=DEFAULT_VERSION, data_format=None,
               dtype=DEFAULT_DTYPE,
               batch_norm_dict=None):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      final_size: The expected size of the model after the second pooling.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1:
        self.block_fn = _bottleneck_block_v1
      else:
        if batch_norm_dict.get('batch_norm_method', '').startswith('sharedlsqrn'):
          self.block_fn = _shared_lsqrn_bottleneck_block_v2
        else:
          self.block_fn = _bottleneck_block_v2
    else:
      if resnet_version == 1:
        self.block_fn = _building_block_v1
      else:
        if batch_norm_dict.get('batch_norm_method', '').startswith('sharedlsqrn'):
          self.block_fn = _shared_lsqrn_building_block_v2
        else:
          self.block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.final_size = final_size
    self.dtype = dtype
    self.pre_activation = resnet_version == 2
    self.batch_norm_dict = batch_norm_dict

  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope('resnet_model',
                             custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
      with tf.variable_scope('initial_conv'):
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format)
      inputs = tf.identity(inputs, 'initial_conv')

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        with tf.variable_scope('initial_bn'):
          inputs = batch_norm(inputs, training, self.data_format, batch_norm_dict=self.batch_norm_dict)
        inputs = tf.nn.relu(inputs)

      if self.first_pool_size:
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        with tf.variable_scope('block_layer_%0.2d' % (i + 1)):
          inputs = block_layer(
              inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
              block_fn=self.block_fn, blocks=num_blocks,
              strides=self.block_strides[i], training=training,
              name='block_layer_%0.2d' % (i + 1), data_format=self.data_format,
              batch_norm_dict=self.batch_norm_dict)

      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      if self.pre_activation:
        with tf.variable_scope('preactivation_bn'):
          inputs = batch_norm(inputs, training, self.data_format,batch_norm_dict=self.batch_norm_dict)
        inputs = tf.nn.relu(inputs)

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.reshape(inputs, [-1, self.final_size])
      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      
      inputs = tf.identity(inputs, 'final_dense')
      return inputs
