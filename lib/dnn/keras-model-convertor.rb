# This library is not yet complete.

# This library converts keras models to ruby-dnn models.
# Use of the library requires the installation of PyCall.

require "pycall/import"
require "numpy"
require_relative "numo2numpy"

include PyCall::Import

pyimport :numpy, as: :np
pyimport :keras
pyfrom :"keras.models", import: :Sequential
pyfrom :"keras.layers", import: [:Dense, :Dropout, :Conv2D, :Activation, :MaxPooling2D, :Flatten]
pyfrom :"keras.layers.normalization", import: :BatchNormalization

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

class KerasModelConvertor
  pyfrom :"keras.models", import: :load_model

  def self.k_load_model(k_model_name, k_weights_name)
    model = load_model(k_model_name)
    model.load_weights(k_weights_name) if k_weights_name
    model
  end

  def initialize(k_model_name, k_weights_name = nil)
    @k_model = KerasModelConvertor.k_load_model(k_model_name, k_weights_name)
  end

  def convert
    unless @k_model.__class__.__name__ == "Sequential"
      raise DNNKerasModelConvertError.new("#{@k_model.__class__.__name__} models do not support convert.")
    end
    layers = convert_layers(@k_model.layers)
    input_shape = @k_model.layers[0].input_shape.to_a[1..-1]
    input_layer = DNN::Layers::InputLayer.new(input_shape)
    input_layer.build(input_shape)
    layers.unshift(input_layer)
    dnn_model = DNN::Models::Sequential.new(layers)
    dnn_model
  end

  def convert_layers(k_layers)
    k_layers.map do |k_layer|
      layer_convert(k_layer)
    end
  end

  private

  def layer_convert(k_layer)
    k_layer_name = k_layer.__class__.__name__
    method_name = "convert_" + k_layer_name
    if respond_to?(method_name, true)
      send(method_name, k_layer)
    else
      raise DNNKerasModelConvertError.new("#{k_layer_name} layer do not support convert.")
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

  def convert_Dense(k_dense)
    input_shape, output_shape = get_k_layer_shape(k_dense)
    dense = DNN::Layers::Dense.new(output_shape[0])
    dense.build(input_shape)
    dense.weight.data = Numpy.to_na(k_dense.get_weights[0])
    dense.bias.data = Numpy.to_na(k_dense.get_weights[1])
    dense
  end

  def convert_Activation(k_activation)
    activation_name = k_activation.get_config[:activation].to_s
    activation = case k_activation.get_config[:activation].to_s
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
    build_dnn_layer(k_activation, activation)
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
    conv2d
  end

  def convert_Conv2DTranspose(k_conv2d)
    padding = k_conv2d.get_config[:padding].to_s == "same" ? true : false
    filter_size = k_conv2d.get_config[:kernel_size].to_a
    strides = k_conv2d.get_config[:strides].to_a
    num_filters = k_conv2d.get_config[:filters]
    conv2d = DNN::Layers::Conv2DTranspose.new(num_filters, filter_size, padding: padding, strides: strides)
    build_dnn_layer(k_conv2d, conv2d)
    conv2d.filters = Numpy.to_na(k_conv2d.get_weights[0])
    conv2d.bias.data = Numpy.to_na(k_conv2d.get_weights[1])
    conv2d
  end

  def convert_MaxPooling2D(k_max_pool2d)
    padding = k_max_pool2d.get_config[:padding].to_s == "same" ? true : false
    pool_size = k_max_pool2d.get_config[:pool_size].to_a
    strides = k_max_pool2d.get_config[:strides].to_a
    max_pool2d = DNN::Layers::MaxPool2D.new(pool_size, padding: padding, strides: strides)
    build_dnn_layer(k_max_pool2d, max_pool2d)
    max_pool2d
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
