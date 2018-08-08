require "test_helper"

include Numo
include DNN::Layers
include DNN::Activations
include DNN::Optimizers
Util = DNN::Util
Model = DNN::Model

class TestLayer < MiniTest::Unit::TestCase
  def test_initialize
    layer = Layer.new
    assert_equal layer.instance_variable_get(:@built), false
  end

  def test_build
    layer = Layer.new
    model = Model.new
    layer.build(model)
    assert_kind_of Model, layer.instance_variable_get(:@model)
  end

  def test_built?
    layer = Layer.new
    model = Model.new
    layer.build(model)
    assert_equal true, layer.built?
  end

  def test_shape
    model = Model.new
    input_layer = InputLayer.new(10)
    layer = Layer.new
    model << input_layer
    model << layer
    model << OutputLayer.new
    model.compile(SGD.new)
    assert_equal input_layer.shape, layer.shape
  end

  def test_to_hash
    layer = Layer.new
    expected_hash = {name: "DNN::Layers::Layer", shape: [10]}
    hash = layer.to_hash({shape: [10]})
    assert_equal expected_hash, hash
  end

  def prev_layer
    model = Model.new
    input_layer = InputLayer.new(10)
    layer = Layer.new
    model << input_layer
    model << layer
    model << OutputLayer.new
    model.compile(SGD.new)
    assert_kind_of InputLayer, layer.prev_layer
  end
end


class TestHasParamLayer < MiniTest::Unit::TestCase
  # Make sure init_param is called
  def test_build
    layer = HasParamLayer.new
    model = Model.new
    layer.define_singleton_method(:init_params) { true }
    assert_equal true, layer.build(model)
  end

  def test_update
    layer = HasParamLayer.new
    model = Model.new
    model << InputLayer.new(10)
    model << layer
    model << OutputLayer.new
    model.compile(SGD.new)
    layer.update
    assert_nil nil
  end
end


class TestInputLayer < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {name: "DNN::Layers::InputLayer", shape: [10]}
    layer = InputLayer.load_hash(hash)
    assert_equal [10], layer.shape
  end

  def test_initialize
    shape = [10, 20]
    layer = InputLayer.new(shape)
    assert_equal shape, layer.shape
  end

  def test_initialize2
    layer = InputLayer.new(10)
    assert_equal [10], layer.shape
  end

  def test_forward
    layer = InputLayer.new(10)
    x = DFloat[0, 1]
    assert_equal x, layer.forward(x)
  end

  def test_backward
    layer = InputLayer.new(10)
    dout = DFloat[0, 1]
    assert_equal dout, layer.backward(dout)
  end

  def test_to_hash
    layer = InputLayer.new(10)
    hash = {name: "DNN::Layers::InputLayer", shape: [10]}
    assert_equal hash, layer.to_hash
  end
end


class TestDense < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      name: "DNN::Layers::Dense",
      num_nodes: 100,
      weight_initializer: {
        name: "DNN::Initializers::RandomNormal",
        mean: 0,
        std: 0.05,
      },
      bias_initializer: {name: "DNN::Initializers::Zeros"},
      weight_decay: 0,
    }
    dense = Dense.load_hash(hash)
    assert_equal 100, dense.num_nodes
  end

  def test_forward
    # y = x.dot(w) + b
    dense = Dense.new(2)
    x = DFloat[[1, 2, 3], [4, 5, 6]]
    dense.params[:weight] = DFloat[[10, 20], [10, 20], [10, 20]]
    dense.params[:bias] = DFloat[5, 10]
    out = dense.forward(x)
    assert_equal DFloat[[65, 130], [155, 310]], out
  end

  def test_backward
    dense = Dense.new(2)
    x = DFloat[[1, 2, 3]]
    dense.params[:weight] = DFloat[[5], [10], [15]]
    dense.params[:bias] = DFloat[20]
    dense.forward(x)
    grad = dense.backward(DFloat[1]).round(3)
    n_grad = Util.numerical_grad(x, dense.method(:forward)).round(3)
    assert_equal n_grad.sum, grad.sum
  end

  def test_shape
    dense = Dense.new(10)
    assert_equal [10], dense.shape
  end

  def test_ridge
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(1, weight_decay: 1)
    model << dense
    model << IdentityMSE.new
    model.compile(SGD.new)
    dense.params[:weight] = SFloat.ones(*dense.params[:weight].shape)
    assert_equal 5.0, dense.ridge.round(1)
  end

  def test_to_hash
    expected_hash = {
      name: "DNN::Layers::Dense",
      num_nodes: 100,
      weight_initializer: {
        name: "DNN::Initializers::RandomNormal",
        mean: 0,
        std: 0.05,
      },
      bias_initializer: {name: "DNN::Initializers::Zeros"},
      weight_decay: 0,
    }
    dense = Dense.new(100)
    assert_equal expected_hash, dense.to_hash
  end
end


class TestReshape < MiniTest::Unit::TestCase
  def test_load
    hash = {
      name: "DNN::Layers::Reshape",
      shape: [32, 32, 3],
    }
    reshape = Reshape.load_hash(hash)
    assert_equal [32, 32, 3], reshape.shape
  end

  def test_forward
    reshape = Reshape.new([32, 32, 3])
    x = SFloat.zeros(10, 3072)
    out = reshape.forward(x)
    assert_equal [10, 32, 32, 3], out.shape
  end

  def test_backward
    reshape = Reshape.new([32, 32, 3])
    x = SFloat.zeros(10, 3072)
    reshape.forward(x)
    dout = SFloat.zeros(10, 32, 32, 3)
    assert_equal [10, 3072], reshape.backward(dout).shape
  end

  def test_to_hash
    expected_hash = {
      name: "DNN::Layers::Reshape",
      shape: [32, 32, 3],
    }
    reshape = Reshape.new([32, 32, 3])
    assert_equal expected_hash, reshape.to_hash
  end
end


class TestOutputLayer < MiniTest::Unit::TestCase
  # l2 = ->lambda, w { 0.5 * lambda * w.sum }
  def test_ridge
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(1, weight_decay: 1)
    model << dense
    dense2 = Dense.new(10, weight_decay: 1)
    model << dense2
    output_layer = OutputLayer.new
    model << output_layer
    model.compile(SGD.new)
    dense.params[:weight] = SFloat.ones(*dense.params[:weight].shape)
    dense2.params[:weight] = SFloat.ones(*dense2.params[:weight].shape)
    assert_equal 10.0, output_layer.send(:ridge).round(1)
  end
end


class TestDropout < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      name: "DNN::Layers::Dropout",
      dropout_ratio: 0.3,
    }
    dropout = Dropout.load_hash(hash)
    assert_equal 0.3, dropout.dropout_ratio
  end

  def test_forward
    model = Model.new
    model << InputLayer.new(1)
    dropout = Dropout.new
    model << dropout
    model << IdentityMSE.new
    model.compile(SGD.new)
    model.instance_variable_set(:@training, true)
    num = dropout.forward(DFloat.ones(100)).sum.round
    assert num.between?(30, 70)
  end

  def test_forward2
    model = Model.new
    model << InputLayer.new(1)
    dropout = Dropout.new
    model << dropout
    model << IdentityMSE.new
    model.compile(SGD.new)
    model.instance_variable_set(:@training, false)
    num = dropout.forward(DFloat.ones(10)).sum.round(1)
    assert_equal 5.0, num
  end

  def test_backward
    model = Model.new
    model << InputLayer.new(1)
    dropout = Dropout.new
    model << dropout
    model << IdentityMSE.new
    model.compile(SGD.new)
    out = dropout.forward(DFloat.ones(10))
    dout = dropout.backward(DFloat.ones(10))
    assert_equal out.round, dout.round
  end

  def test_backward
    model = Model.new
    model << InputLayer.new(1)
    dropout = Dropout.new(1.0)
    model << dropout
    model << IdentityMSE.new
    model.compile(SGD.new)
    model.instance_variable_set(:@training, false)
    dropout.forward(DFloat.ones(10))
    dout = dropout.backward(DFloat.ones(10))
    assert_equal 10.0, dout.sum.round(1)
  end

  def test_to_hash
    expected_hash = {
      name: "DNN::Layers::Dropout",
      dropout_ratio: 0.5
    }
    dropout = Dropout.new
    assert_equal expected_hash, dropout.to_hash
  end
end


class TestBatchNormalization < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      name: "DNN::Layers::BatchNormalization",
      momentum: 0.8,
      running_mean: SFloat.ones(10),
      running_var: SFloat.ones(10),
    }
    batch_norm = BatchNormalization.load_hash(hash)
    assert_equal 0.8, batch_norm.momentum
    assert_equal SFloat.ones(10), batch_norm.instance_variable_get(:@running_mean)
    assert_equal SFloat.ones(10), batch_norm.instance_variable_get(:@running_var)
  end

  def test_build
    batch_norm = BatchNormalization.new
    model = Model.new
    model << InputLayer.new(10)
    model << batch_norm
    batch_norm.build(model)
    assert_equal SFloat.zeros(10), batch_norm.instance_variable_get(:@running_mean)
    assert_equal SFloat.zeros(10), batch_norm.instance_variable_get(:@running_var)
  end

  # mean = ->x { x.mean(0) }
  # std = ->x { NMath.sqrt(((x - mean.(x))**2).mean(0)) }
  # batch_norm = ->x, gamma, beta { 
  #   xn = (x - mean.(x)) / std.(x)
  #   gamma * xn + beta
  # }
  def test_forward
    model = Model.new
    model << InputLayer.new(1)
    model << Dense.new(10)
    batch_norm = BatchNormalization.new
    model << batch_norm
    model << IdentityMSE.new
    model.compile(SGD.new)
    model.instance_variable_set(:@training, true)
    batch_norm.params[:gamma] = SFloat.new(10).fill(3)
    batch_norm.params[:beta] = SFloat.new(10).fill(10)
    x = DFloat.cast([DFloat.new(10).fill(10), DFloat.new(10).fill(20)])
    expected = DFloat.cast([DFloat.new(10).fill(7), DFloat.new(10).fill(13)])
    assert_equal expected, batch_norm.forward(x).round(4)
  end

  def test_forward2
    model = Model.new
    model << InputLayer.new(1)
    model << Dense.new(10)
    
    batch_norm = BatchNormalization.new(running_mean: DFloat.new(10).fill(15),
                                        running_var: DFloat.new(10).fill(25))
    model << batch_norm
    model << IdentityMSE.new
    model.compile(SGD.new)
    model.instance_variable_set(:@training, false)
    batch_norm.params[:gamma] = SFloat.new(10).fill(3)
    batch_norm.params[:beta] = SFloat.new(10).fill(10)
    x = DFloat.cast([DFloat.new(10).fill(10), DFloat.new(10).fill(20)])
    expected = DFloat.cast([DFloat.new(10).fill(7), DFloat.new(10).fill(13)])
    assert_equal expected, batch_norm.forward(x).round(4)
  end

  def test_backward
    model = Model.new
    model << InputLayer.new(1)
    model << Dense.new(10)
    batch_norm = BatchNormalization.new
    model << batch_norm
    model << IdentityMSE.new
    model.compile(SGD.new)
    model.instance_variable_set(:@training, true)
    x = DFloat.cast([DFloat.new(10).fill(10), DFloat.new(10).fill(20)])
    batch_norm.forward(x)
    grad = batch_norm.backward(DFloat.ones(*x.shape)).round(4)
    n_grad = Util.numerical_grad(x, batch_norm.method(:forward)).round(4)
    assert_equal n_grad, grad
  end

  def test_to_hash
    expected_hash = {
      name: "DNN::Layers::BatchNormalization",
      momentum: 0.9,
      running_mean: [0] * 10,
      running_var: [0] * 10,
    }
    model = Model.new
    model << InputLayer.new(1)
    model << Dense.new(10)
    batch_norm = BatchNormalization.new
    model << batch_norm
    model << IdentityMSE.new
    model.compile(SGD.new)
    assert_equal expected_hash, batch_norm.to_hash
  end
end