# This library is not yet complete.

# This library converts keras models to ruby-dnn models.
# Use of the library requires the installation of PyCall.

require "pycall/import"
require "numpy"
require_relative "numo2numpy"

include PyCall::Import

pyimport :keras

module DNN
  module Layers
    class Softmax < Layer
      def forward(x)
        Exp.(x) / Sum.(Exp.(x), axis: 1)
      end
    end
  end
end

class DNNKerasModelConvertError < DNN::DNNError; end

class DNNKerasLayerNotConvertSupportError < DNNKerasModelConvertError; end

class KerasModelConvertor
  pyfrom :"keras.models", import: :load_model

  def self.load(k_model_name, k_weights_name = nil)
    k_model = load_model(k_model_name)
    k_model.load_weights(k_weights_name) if k_weights_name
    self.new(k_model)
  end

  def initialize(k_model)
    @k_model = k_model
  end

  def convert(not_support_to_nil: false, debug_message: false)
    unless @k_model.__class__.__name__ == "Sequential"
      raise DNNKerasModelConvertError.new("#{@k_model.__class__.__name__} models do not support convert.")
    end
    layers = convert_layers(not_support_to_nil: not_support_to_nil, debug_message: debug_message)
    dnn_model = DNN::Models::Sequential.new(layers)
    dnn_model
  end

  def convert_layers(not_support_to_nil: false, debug_message: false)
    layers = []
    @k_model.layers.each do |k_layer|
      layer = if not_support_to_nil
        begin
          layer_convert(k_layer)
        rescue DNNKerasLayerNotConvertSupportError => e
          nil
        end
      else
        layer_convert(k_layer)
      end
      if layer.is_a?(Array)
        layer.each { |l| puts "Converted #{l.class.name} layer" } if debug_message
        layers += layer
      else
        puts "Converted #{layer.class.name} layer" if debug_message
        layers << layer
      end
    end
    layers
  end

  private

  def layer_convert(k_layer)
    k_layer_name = k_layer.__class__.__name__
    method_name = "convert_" + k_layer_name
    if respond_to?(method_name, true)
      send(method_name, k_layer)
    else
      raise DNNKerasLayerNotConvertSupportError.new("#{k_layer_name} layer do not support convert.")
    end
  end

  def get_k_layer_shape(k_layer)
    input_shape = k_layer.input_shape.to_a[1..-1]
    output_shape = k_layer.output_shape.to_a[1..-1]
    [input_shape, output_shape]
  end

  def build_dnn_layer(k_layer, dnn_layer)
    input_shape, output_shape = get_k_layer_shape(k_layer)
    dnn_layer.build(input_shape)
  end

  def convert_InputLayer(k_input_layer)
    input_shape, output_shape = get_k_layer_shape(k_input_layer)
    input_layer = DNN::Layers::InputLayer.new(input_shape)
    input_layer.build(input_shape)
    input_layer
  end

  def convert_Dense(k_dense)
    input_shape, output_shape = get_k_layer_shape(k_dense)
    dense = DNN::Layers::Dense.new(output_shape[0])
    dense.build(input_shape)
    dense.weight.data = Numpy.to_na(k_dense.get_weights[0])
    dense.bias.data = Numpy.to_na(k_dense.get_weights[1])
    returns = [dense]
    unless k_dense.get_config[:activation] == "linear"
      returns << activation_to_dnn_layer(k_dense.get_config[:activation], output_shape)
    end
    returns
  end

  def convert_Activation(k_activation)
    input_shape, output_shape = get_k_layer_shape(k_activation)
    activation_name = k_activation.get_config[:activation].to_s
    activation_to_dnn_layer(activation_name, input_shape)
  end

  def activation_to_dnn_layer(activation_name, shape)
    activation = case activation_name
    when "sigmoid"
       DNN::Layers::Sigmoid.new
    when "tanh"
      DNN::Layers::Tanh.new
    when "relu"
      DNN::Layers::ReLU.new
    when "softmax"
      DNN::Layers::Softmax.new
    else
      raise DNNKerasModelConvertError.new("#{activation_name} activation do not support convert.")
    end
    activation.build(shape)
    activation
  end

  def convert_Dropout(k_dropout)
    dropout_ratio = k_dropout.get_config[:rate]
    dropout = DNN::Layers::Dropout.new(dropout_ratio, use_scale: false)
    build_dnn_layer(k_dropout, dropout)
    dropout
  end

  def convert_BatchNormalization(k_batch_norm)
    momentum = k_batch_norm.get_config[momentum]
    batch_norm = DNN::Layers::BatchNormalization.new(momentum: momentum)
    build_dnn_layer(k_batch_norm, batch_norm)
    batch_norm.gamma.data = Numpy.to_na(k_batch_norm.get_weights[0])
    batch_norm.beta.data = Numpy.to_na(k_batch_norm.get_weights[1])
    batch_norm.running_mean.data = Numpy.to_na(k_batch_norm.get_weights[2])
    batch_norm.running_var.data = Numpy.to_na(k_batch_norm.get_weights[3])
    batch_norm
  end

  def convert_Conv2D(k_conv2d)
    padding = k_conv2d.get_config[:padding].to_s == "same" ? true : false
    filter_size = k_conv2d.get_config[:kernel_size].to_a
    strides = k_conv2d.get_config[:strides].to_a
    num_filters = k_conv2d.get_config[:filters]
    conv2d = DNN::Layers::Conv2D.new(num_filters, filter_size, padding: padding, strides: strides)
    build_dnn_layer(k_conv2d, conv2d)
    conv2d.filters = Numpy.to_na(k_conv2d.get_weights[0])
    conv2d.bias.data = Numpy.to_na(k_conv2d.get_weights[1])
    returns = [conv2d]
    unless k_conv2d.get_config[:activation] == "linear"
      input_shape, output_shape = get_k_layer_shape(k_conv2d)
      returns << activation_to_dnn_layer(k_conv2d.get_config[:activation], output_shape)
    end
    returns
  end

  def convert_Conv2DTranspose(k_conv2d_t)
    padding = k_conv2d_t.get_config[:padding].to_s == "same" ? true : false
    filter_size = k_conv2d_t.get_config[:kernel_size].to_a
    strides = k_conv2d_t.get_config[:strides].to_a
    num_filters = k_conv2d_t.get_config[:filters]
    conv2d_t = DNN::Layers::Conv2DTranspose.new(num_filters, filter_size, padding: padding, strides: strides)
    build_dnn_layer(k_conv2d_t, conv2d_t)
    conv2d_t.filters = Numpy.to_na(k_conv2d_t.get_weights[0])
    conv2d_t.bias.data = Numpy.to_na(k_conv2d_t.get_weights[1])
    returns = [conv2d_t]
    unless k_conv2d_t.get_config[:activation] == "linear"
      input_shape, output_shape = get_k_layer_shape(k_conv2d)
      returns << activation_to_dnn_layer(k_conv2d_t.get_config[:activation], output_shape)
    end
    returns
  end

  def convert_MaxPooling2D(k_max_pool2d)
    padding = k_max_pool2d.get_config[:padding].to_s == "same" ? true : false
    pool_size = k_max_pool2d.get_config[:pool_size].to_a
    strides = k_max_pool2d.get_config[:strides].to_a
    max_pool2d = DNN::Layers::MaxPool2D.new(pool_size, padding: padding, strides: strides)
    build_dnn_layer(k_max_pool2d, max_pool2d)
    max_pool2d
  end

  def convert_AveragePooling2D(k_avg_pool2d)
    padding = k_avg_pool2d.get_config[:padding].to_s == "same" ? true : false
    pool_size = k_avg_pool2d.get_config[:pool_size].to_a
    strides = k_avg_pool2d.get_config[:strides].to_a
    avg_pool2d = DNN::Layers::AvgPool2D.new(pool_size, padding: padding, strides: strides)
    build_dnn_layer(k_avg_pool2d, avg_pool2d)
    avg_pool2d
  end

  def convert_GlobalAveragePooling2D(k_glb_avg_pool2d)
    padding = k_glb_avg_pool2d.get_config[:padding].to_s == "same" ? true : false
    pool_size = k_glb_avg_pool2d.get_config[:pool_size].to_a
    strides = k_glb_avg_pool2d.get_config[:strides].to_a
    glb_avg_pool2d = DNN::Layers::GlobalAvgPool2D.new
    build_dnn_layer(k_glb_avg_pool2d, glb_avg_pool2d)
    glb_avg_pool2d
  end

  def convert_UpSampling2D(k_upsampling2d)
    input_shape, output_shape = get_k_layer_shape(k_upsampling2d)
    unpool_size = k_upsampling2d.get_config[:size].to_a
    unpool2d = DNN::Layers::UnPool2D.new(unpool_size)
    build_dnn_layer(k_upsampling2d, unpool2d)
    unpool2d
  end

  def convert_Flatten(k_flatten)
    flatten = DNN::Layers::Flatten.new
    build_dnn_layer(k_flatten, flatten)
    flatten
  end

  def convert_Reshape(k_reshape)
    input_shape, output_shape = get_k_layer_shape(k_reshape)
    reshape = DNN::Layers::Reshape.new(output_shape)
    build_dnn_layer(k_reshape, reshape)
    reshape
  end
end
