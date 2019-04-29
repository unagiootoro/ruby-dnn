require "test_helper"

include DNN
include Layers
include Activations
include Optimizers

class TestLayer < MiniTest::Unit::TestCase
  def test_initialize
    layer = Layer.new
    assert_equal layer.instance_variable_get(:@built), false
  end

  def test_build
    layer = Layer.new
    layer.build([10])
    assert_equal [10], layer.instance_variable_get(:@input_shape)
  end

  def test_built?
    layer = Layer.new
    layer.build([10])
    assert_equal true, layer.built?
  end

  def test_output_shape
    layer = Layer.new
    layer.build([10])
    assert_equal [10], layer.output_shape
  end

  def test_to_hash
    layer = Layer.new
    expected_hash = {class: "DNN::Layers::Layer", output_shape: [10]}
    hash = layer.to_hash({output_shape: [10]})
    assert_equal expected_hash, hash
  end
end


class TestHasParamLayer < MiniTest::Unit::TestCase
  # Make sure init_param is called
  def test_build
    layer = HasParamLayer.new
    layer.define_singleton_method(:init_params) { true }
    assert_equal true, layer.build([10])
  end

  def test_update
    layer = HasParamLayer.new
    weight = Param.new(Numo::SFloat.ones(1), Numo::SFloat.ones(1))
    layer.instance_variable_set(:@params, {weight: weight})
    layer.update(SGD.new)
    assert_equal Numo::SFloat[0.99], layer.params[:weight].data
  end
end


class TestInputLayer < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {class: "DNN::Layers::InputLayer", input_shape: [10]}
    layer = InputLayer.load_hash(hash)
    assert_equal [10], layer.output_shape
  end

  def test_initialize
    layer = InputLayer.new([10, 20])
    assert_equal [10, 20], layer.instance_variable_get(:@input_shape)
  end

  def test_initialize2
    layer = InputLayer.new(10)
    assert_equal [10], layer.instance_variable_get(:@input_shape)
  end

  def test_forward
    layer = InputLayer.new(10)
    x = Numo::SFloat[0, 1]
    assert_equal x, layer.forward(x)
  end

  def test_backward
    layer = InputLayer.new(10)
    dout = Numo::SFloat[0, 1]
    assert_equal dout, layer.backward(dout)
  end

  def test_to_hash
    layer = InputLayer.new(10)
    hash = {class: "DNN::Layers::InputLayer", input_shape: [10]}
    assert_equal hash, layer.to_hash
  end
end


class TestDense < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      class: "DNN::Layers::Dense",
      num_nodes: 100,
      weight_initializer: {
        class: "DNN::Initializers::RandomNormal",
        mean: 0,
        std: 0.05,
      },
      bias_initializer: {class: "DNN::Initializers::Zeros"},
      l1_lambda: 0,
      l2_lambda: 0,
    }
    dense = Dense.load_hash(hash)
    assert_equal 100, dense.num_nodes
  end

  def test_forward
    dense = Dense.new(2)
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense.params[:weight].data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.params[:bias].data = Numo::SFloat[5, 10]
    out = dense.forward(x)
    assert_equal Numo::SFloat[[65, 130], [155, 310]], out
  end

  def test_backward
    dense = Dense.new(2)
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense.params[:weight].data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.params[:bias].data = Numo::SFloat[5, 10]
    dense.forward(x)
    grad = dense.backward(Numo::SFloat[1])
    assert_equal Numo::SFloat[30, 30, 30], grad.round(4)
    assert_equal Numo::SFloat[5, 7, 9], dense.params[:weight].grad.round(4)
    assert_in_delta 1.0, dense.params[:bias].grad
  end

  def test_output_shape
    dense = Dense.new(10)
    assert_equal [10], dense.output_shape
  end

  def test_lasso
    dense = Dense.new(1, l1_lambda: 1)
    dense.build([10])
    dense.params[:weight].data = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    assert_equal 10, dense.lasso.round(1)
  end

  def test_ridge
    dense = Dense.new(1, l2_lambda: 1)
    dense.build([10])
    dense.params[:weight].data = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    assert_equal 5.0, dense.ridge.round(1)
  end

  def test_to_hash
    dense = Dense.new(100)
    expected_hash = {
      class: "DNN::Layers::Dense",
      num_nodes: 100,
      weight_initializer: dense.weight_initializer.to_hash,
      bias_initializer: dense.bias_initializer.to_hash,
      l1_lambda: 0,
      l2_lambda: 0,
    }
    assert_equal expected_hash, dense.to_hash
  end
end


class TestFlatten < MiniTest::Unit::TestCase
  def test_forward
    flatten = Flatten.new
    x = Numo::SFloat.zeros(10, 32, 32, 3)
    flatten.build([32, 32, 3])
    out = flatten.forward(x)
    assert_equal [10, 3072], out.shape
  end

  def test_backward
    flatten = Flatten.new
    x = Numo::SFloat.zeros(10, 32, 32, 3)
    flatten.build([32, 32, 3])
    flatten.forward(x)
    dout = Numo::SFloat.zeros(10, 3072)
    assert_equal [10, 32, 32, 3], flatten.backward(dout).shape
  end
end

class TestReshape < MiniTest::Unit::TestCase
  def test_load
    hash = {
      class: "DNN::Layers::Reshape",
      output_shape: [32, 32, 3],
    }
    reshape = Reshape.load_hash(hash)
    assert_equal [32, 32, 3], reshape.output_shape
  end

  def test_forward
    reshape = Reshape.new([32, 32, 3])
    reshape.build([3072])
    x = Numo::SFloat.zeros(10, 3072)
    out = reshape.forward(x)
    assert_equal [10, 32, 32, 3], out.shape
  end

  def test_backward
    reshape = Reshape.new([32, 32, 3])
    reshape.build([3072])
    x = Numo::SFloat.zeros(10, 3072)
    reshape.forward(x)
    dout = Numo::SFloat.zeros(10, 32, 32, 3)
    assert_equal [10, 3072], reshape.backward(dout).shape
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Layers::Reshape",
      output_shape: [32, 32, 3],
    }
    reshape = Reshape.new([32, 32, 3])
    assert_equal expected_hash, reshape.to_hash
  end
end


class TestDropout < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      class: "DNN::Layers::Dropout",
      dropout_ratio: 0.3,
      seed: 0,
    }
    dropout = Dropout.load_hash(hash)
    assert_equal 0.3, dropout.dropout_ratio
    assert_equal 0, dropout.instance_variable_get(:@seed)
  end

  def test_forward
    dropout = Dropout.new(0.5, 0)
    dropout.build([100])
    num = dropout.forward(Numo::SFloat.ones(100), true).sum.round
    assert num.between?(30, 70)
  end

  def test_forward2
    dropout = Dropout.new
    dropout.build([1])
    num = dropout.forward(Numo::SFloat.ones(10), false).sum.round(1)
    assert_equal 5.0, num
  end

  def test_backward
    dropout = Dropout.new
    dropout.build([1])
    out = dropout.forward(Numo::SFloat.ones(10), true)
    dout = dropout.backward(Numo::SFloat.ones(10), true)
    assert_equal out.round, dout.round
  end

  def test_backward2
    dropout = Dropout.new(1.0)
    dropout.build([1])
    dropout.forward(Numo::SFloat.ones(10), false)
    dout = dropout.backward(Numo::SFloat.ones(10), false)
    assert_equal 10.0, dout.sum.round(1)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Layers::Dropout",
      dropout_ratio: 0.3,
      seed: 0,
    }
    dropout = Dropout.new(0.3, 0)
    assert_equal expected_hash, dropout.to_hash
  end
end


class TestBatchNormalization < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      class: "DNN::Layers::BatchNormalization",
      momentum: 0.8,
    }
    batch_norm = BatchNormalization.load_hash(hash)
    assert_equal 0.8, batch_norm.momentum
  end

  def test_forward
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    batch_norm.params[:gamma].data = Numo::SFloat.new(10).fill(3)
    batch_norm.params[:beta].data = Numo::SFloat.new(10).fill(10)
    x = Numo::SFloat.cast([Numo::SFloat.new(10).fill(10), Numo::SFloat.new(10).fill(20)])
    expected = Numo::SFloat.cast([Numo::SFloat.new(10).fill(7), Numo::SFloat.new(10).fill(13)])
    assert_equal expected, batch_norm.forward(x, true).round(4)
  end

  def test_forward2
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    batch_norm.params[:gamma].data = Numo::SFloat.new(10).fill(3)
    batch_norm.params[:beta].data = Numo::SFloat.new(10).fill(10)
    batch_norm.params[:running_mean].data = Numo::SFloat.new(10).fill(15)
    batch_norm.params[:running_var].data = Numo::SFloat.new(10).fill(25)
    x = Numo::SFloat.cast([Numo::SFloat.new(10).fill(10), Numo::SFloat.new(10).fill(20)])
    expected = Numo::SFloat.cast([Numo::SFloat.new(10).fill(7), Numo::SFloat.new(10).fill(13)])
    assert_equal expected, batch_norm.forward(x, false).round(4)
  end

  def test_backward
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    x = Numo::SFloat.cast([Numo::SFloat.new(10).fill(10), Numo::SFloat.new(10).fill(20)])
    batch_norm.forward(x, true)
    grad = batch_norm.backward(Numo::SFloat.ones(*x.shape), true)
    assert_equal Numo::SFloat[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], grad.round(4)
    assert_equal Numo::SFloat[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], batch_norm.params[:gamma].grad
    assert_equal Numo::SFloat[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], batch_norm.params[:beta].grad
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Layers::BatchNormalization",
      momentum: 0.9,
    }
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    assert_equal expected_hash, batch_norm.to_hash
  end
end
