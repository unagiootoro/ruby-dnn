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
  def test_update
    layer = HasParamLayer.new
    weight = Param.new(Numo::SFloat.ones(1), Numo::SFloat.ones(1))
    layer.instance_variable_set(:@params, {weight: weight})
    layer.update(SGD.new)
    assert_equal Numo::SFloat[0.99], layer.params[:weight].data
  end
end


class TestInputLayer < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {class: "DNN::Layers::InputLayer", input_shape: [10]}
    layer = InputLayer.from_hash(hash)
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
    dy = Numo::SFloat[0, 1]
    assert_equal dy, layer.backward(dy)
  end

  def test_to_hash
    layer = InputLayer.new(10)
    hash = {class: "DNN::Layers::InputLayer", input_shape: [10]}
    assert_equal hash, layer.to_hash
  end
end


class TestDense < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::Dense",
      num_nodes: 100,
      weight_initializer: {
        class: "DNN::Initializers::RandomUniform",
        min: -0.05,
        max: 0.05,
      },
      bias_initializer: {
        class: "DNN::Initializers::RandomUniform",
        min: -0.05,
        max: 0.05,
      },
      l1_lambda: 0.1,
      l2_lambda: 0.2,
      use_bias: false,
    }
    dense = Dense.from_hash(hash)
    assert_equal 100, dense.num_nodes
    assert_kind_of RandomUniform, dense.weight_initializer
    assert_kind_of RandomUniform, dense.bias_initializer
    assert_equal 0.1, dense.l1_lambda
    assert_equal 0.2, dense.l2_lambda
    assert_equal false, dense.use_bias
  end

  def test_build
    dense = Dense.new(100)
    dense.build([50])
    assert_kind_of RandomNormal, dense.weight_initializer
    assert_kind_of Zeros, dense.bias_initializer
    assert_equal [50, 100], dense.params[:weight].data.shape
    assert_equal [100], dense.params[:bias].data.shape
  end

  def test_forward
    dense = Dense.new(2)
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense.params[:weight].data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.params[:bias].data = Numo::SFloat[5, 10]
    y = dense.forward(x)
    assert_equal Numo::SFloat[[65, 130], [155, 310]], y
  end

  def test_forward2
    dense = Dense.new(2, use_bias: false)
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense.params[:weight].data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    y = dense.forward(x)
    assert_equal Numo::SFloat[[60, 120], [150, 300]], y
    assert_nil dense.params[:bias]
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

  def test_backward2
    dense = Dense.new(2, use_bias: false)
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense.params[:weight].data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.forward(x)
    grad = dense.backward(Numo::SFloat[1])
    assert_equal Numo::SFloat[30, 30, 30], grad.round(4)
    assert_equal Numo::SFloat[5, 7, 9], dense.params[:weight].grad.round(4)
    assert_nil dense.params[:bias]
  end

  def test_backward3
    dense = Dense.new(2)
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense.params[:weight].data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.params[:bias].data = Numo::SFloat[5, 10]
    dense.forward(x)
    dense.forward(x)
    dense.backward(Numo::SFloat[1])
    grad = dense.backward(Numo::SFloat[1])
    assert_equal Numo::SFloat[30, 30, 30], grad.round(4)
    assert_equal Numo::SFloat[10, 14, 18], dense.params[:weight].grad.round(4)
    assert_in_delta 2.0, dense.params[:bias].grad
  end

  def test_output_shape
    dense = Dense.new(10)
    assert_equal [10], dense.output_shape
  end

  def test_regularizers
    dense = Dense.new(1, l1_lambda: 1, l2_lambda: 1)
    dense.build([10])
    assert_kind_of Lasso, dense.regularizers[0]
    assert_kind_of Ridge, dense.regularizers[1]
  end

  def test_regularizers2
    dense = Dense.new(1)
    dense.build([10])
    assert_equal [], dense.regularizers
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
      use_bias: true,
    }
    assert_equal expected_hash, dense.to_hash
  end
end


class TestFlatten < MiniTest::Unit::TestCase
  def test_forward
    flatten = Flatten.new
    x = Numo::SFloat.zeros(10, 32, 32, 3)
    flatten.build([32, 32, 3])
    y = flatten.forward(x)
    assert_equal [10, 3072], y.shape
  end

  def test_backward
    flatten = Flatten.new
    x = Numo::SFloat.zeros(10, 32, 32, 3)
    flatten.build([32, 32, 3])
    flatten.forward(x)
    dy = Numo::SFloat.zeros(10, 3072)
    assert_equal [10, 32, 32, 3], flatten.backward(dy).shape
  end
end

class TestReshape < MiniTest::Unit::TestCase
  def test_load
    hash = {
      class: "DNN::Layers::Reshape",
      output_shape: [32, 32, 3],
    }
    reshape = Reshape.from_hash(hash)
    assert_equal [32, 32, 3], reshape.output_shape
  end

  def test_forward
    reshape = Reshape.new([32, 32, 3])
    reshape.build([3072])
    x = Numo::SFloat.zeros(10, 3072)
    y = reshape.forward(x)
    assert_equal [10, 32, 32, 3], y.shape
  end

  def test_backward
    reshape = Reshape.new([32, 32, 3])
    reshape.build([3072])
    x = Numo::SFloat.zeros(10, 3072)
    reshape.forward(x)
    dy = Numo::SFloat.zeros(10, 32, 32, 3)
    assert_equal [10, 3072], reshape.backward(dy).shape
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
  def test_from_hash
    hash = {
      class: "DNN::Layers::Dropout",
      dropout_ratio: 0.3,
      seed: 0,
      use_scale: false,
    }
    dropout = Dropout.from_hash(hash)
    assert_equal 0.3, dropout.dropout_ratio
    assert_equal 0, dropout.instance_variable_get(:@seed)
    assert_equal false, dropout.use_scale
  end

  def test_forward
    dropout = Dropout.new(0.5, seed: 0)
    dropout.build([100])
    dropout.learning_phase = true
    num = dropout.forward(Numo::SFloat.ones(100)).sum.round
    assert num.between?(30, 70)
  end

  def test_forward2
    dropout = Dropout.new(0.3, use_scale: true)
    dropout.build([1])
    dropout.learning_phase = false
    num = dropout.forward(Numo::SFloat.ones(10)).sum.round(1)
    assert_equal 7.0, num
  end

  def test_forward3
    dropout = Dropout.new(0.3, use_scale: false)
    dropout.build([1])
    dropout.learning_phase = false
    num = dropout.forward(Numo::SFloat.ones(10)).sum.round(1)
    assert_equal 10.0, num
  end

  def test_backward
    dropout = Dropout.new
    dropout.build([1])
    dropout.learning_phase = true
    y = dropout.forward(Numo::SFloat.ones(10))
    dy = dropout.backward(Numo::SFloat.ones(10))
    assert_equal y.round, dy.round
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Layers::Dropout",
      dropout_ratio: 0.3,
      seed: 0,
      use_scale: false,
    }
    dropout = Dropout.new(0.3, seed: 0, use_scale: false)
    assert_equal expected_hash, dropout.to_hash
  end
end
