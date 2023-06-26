require "test_helper"

include DNN::Layers
include DNN::Optimizers
include DNN::Initializers
include DNN::Regularizers

class TestLayer < MiniTest::Unit::TestCase
  def test_initialize
    layer = Layer.new
    assert_equal layer.instance_variable_get(:@built), false
  end

  def test_build
    layer = Layer.new
    layer.build([10])
    assert_equal [[10]], layer.instance_variable_get(:@input_shapes)
  end

  def test_built?
    layer = Layer.new
    layer.build([10])
    assert_equal true, layer.built?
  end

  def test_to_hash
    layer = Layer.new
    expected_hash = { class: "DNN::Layers::Layer" }
    hash = layer.to_hash
    assert_equal expected_hash, hash
  end

  def test_clean
    dense = Dense.new(20)
    dense.build([10])
    dense.clean
    assert_equal 20, dense.num_units
  end
end


class TestInputLayer < MiniTest::Unit::TestCase
  def test_initialize
    layer = InputLayer.new([10, 20])
    assert_equal [10, 20], layer.instance_variable_get(:@input_shape)
  end

  def test_initialize2
    layer = InputLayer.new(10)
    assert_equal [10], layer.instance_variable_get(:@input_shape)
  end

  def test_forward
    layer = InputLayer.new(2)
    x = Xumo::SFloat[[0, 1]]
    out = layer.forward(DNN::Tensor.new(x))
    assert_equal x, out.data
  end

  def test_to_hash
    layer = InputLayer.new(10)
    hash = { class: "DNN::Layers::InputLayer", input_shape: [10] }
    assert_equal hash, layer.to_hash
  end
end


class TestDense < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::Dense",
      num_units: 100,
      weight_initializer: RandomUniform.new.to_hash,
      bias_initializer: RandomNormal.new.to_hash,
      weight_regularizer: L1.new.to_hash,
      bias_regularizer: L2.new.to_hash,
      use_bias: false,
    }
    dense = Dense.from_hash(hash)
    assert_equal 100, dense.num_units
    assert_kind_of RandomUniform, dense.weight_initializer
    assert_kind_of RandomNormal, dense.bias_initializer
    assert_kind_of L1, dense.weight_regularizer
    assert_kind_of L2, dense.bias_regularizer
    assert_equal false, dense.use_bias
  end

  def test_build
    dense = Dense.new(100)
    dense.build([50])
    assert_kind_of RandomNormal, dense.weight_initializer
    assert_kind_of Zeros, dense.bias_initializer
    assert_equal [50, 100], dense.weight.data.shape
    assert_equal [100], dense.bias.data.shape
  end

  def test_forward
    dense = Dense.new(2)
    dense.build([2])
    x = DNN::Tensor.new(Xumo::SFloat[[1, 2, 3], [4, 5, 6]])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    y = dense.(x)
    assert_equal Xumo::SFloat[[65, 130], [155, 310]], y.data
  end

  def test_forward2
    dense = Dense.new(2, use_bias: false)
    dense.build([2])
    x = DNN::Tensor.new(Xumo::SFloat[[1, 2, 3], [4, 5, 6]])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    y = dense.(x)
    assert_equal Xumo::SFloat[[60, 120], [150, 300]], y.data
    assert_nil dense.bias
  end

  def test_backward
    dense = Dense.new(2)
    dense.build([2])
    x = DNN::Variable.new(Xumo::SFloat[[1, 2, 3], [4, 5, 6]])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    y = dense.(x)
    y.backward(Xumo::SFloat[[1, 1], [1, 1]])
    assert_equal Xumo::SFloat[[30, 30, 30], [30, 30, 30]], x.grad
    assert_equal Xumo::SFloat[[5, 5], [7, 7], [9, 9]], dense.weight.grad
    assert_in_delta 1.0, dense.bias.grad
  end

  def test_backward2
    dense = Dense.new(2, use_bias: false)
    dense.build([2])
    x = DNN::Variable.new(Xumo::SFloat[[1, 2, 3], [4, 5, 6]])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    y = dense.(x)
    y.backward(Xumo::SFloat[[1, 1], [1, 1]])
    assert_equal Xumo::SFloat[[30, 30, 30], [30, 30, 30]], x.grad
    assert_equal Xumo::SFloat[[5, 5], [7, 7], [9, 9]], dense.weight.grad
    assert_nil dense.bias
  end

  # def test_backward3
  #   dense = Dense.new(2)
  #   x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
  #   dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
  #   dense.bias.data = Xumo::SFloat[5, 10]
  #   dense.forward(x)
  #   dense.forward(x)
  #   dense.backward(Xumo::SFloat[1])
  #   grad = dense.backward(Xumo::SFloat[1])
  #   assert_equal Xumo::SFloat[30, 30, 30], grad.round(4)
  #   assert_equal Xumo::SFloat[10, 14, 18], dense.weight.grad.round(4)
  #   assert_in_delta 2.0, dense.bias.grad
  # end

  # def test_backward4
  #   dense = Dense.new(2)
  #   dense.trainable = false
  #   x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
  #   dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
  #   dense.bias.data = Xumo::SFloat[5, 10]
  #   dense.forward(x)
  #   grad = dense.backward(Xumo::SFloat[1])
  #   assert_equal Xumo::SFloat[30, 30, 30], grad.round(4)
  #   assert_equal Xumo::SFloat[0], dense.weight.grad.round(4)
  #   assert_equal Xumo::SFloat[0], dense.bias.grad
  # end

  # def test_output_shape
  #   dense = Dense.new(10)
  #   dense.build([1])
  #   assert_equal [10], dense.output_shape
  # end

  # def test_regularizers
  #   dense = Dense.new(1, weight_regularizer: L1.new, bias_regularizer: L2.new)
  #   dense.build([10])
  #   assert_kind_of L1, dense.regularizers[0]
  #   assert_kind_of L2, dense.regularizers[1]
  # end

  # def test_regularizers2
  #   dense = Dense.new(1)
  #   dense.build([10])
  #   assert_equal [], dense.regularizers
  # end

  # def test_to_hash
  #   dense = Dense.new(100, weight_regularizer: L1.new, bias_regularizer: L2.new)
  #   expected_hash = {
  #     class: "DNN::Layers::Dense",
  #     num_units: 100,
  #     weight_initializer: dense.weight_initializer.to_hash,
  #     bias_initializer: dense.bias_initializer.to_hash,
  #     weight_regularizer: dense.weight_regularizer.to_hash,
  #     bias_regularizer: dense.bias_regularizer.to_hash,
  #     use_bias: true,
  #   }
  #   assert_equal expected_hash, dense.to_hash
  # end

  # def test_get_params
  #   dense = Dense.new(10)
  #   dense.build([10])
  #   expected_hash = {
  #     weight: dense.weight,
  #     bias: dense.bias,
  #   }
  #   assert_equal expected_hash, dense.get_variables
  # end
end


# class TestFlatten < MiniTest::Unit::TestCase
#   def test_forward
#     flatten = Flatten.new
#     x = DNN::Tensor.convert(Xumo::SFloat.zeros(10, 32, 32, 3))
#     flatten.build([32, 32, 3])
#     y = flatten.(x)
#     assert_equal [10, 3072], y.shape
#   end
# end


# class TestReshape < MiniTest::Unit::TestCase
#   def test_load
#     hash = {
#       class: "DNN::Layers::Reshape",
#       shape: [32, 32, 3],
#     }
#     reshape = Reshape.from_hash(hash)
#     reshape.build([32 * 32 * 3])
#     assert_equal [32, 32, 3], reshape.output_shape
#   end

#   def test_forward
#     reshape = Reshape.new([32, 32, 3])
#     reshape.build([3072])
#     x = Xumo::SFloat.zeros(10, 3072)
#     y = reshape.forward(x)
#     assert_equal [10, 32, 32, 3], y.shape
#   end

#   def test_backward
#     reshape = Reshape.new([32, 32, 3])
#     reshape.build([3072])
#     x = Xumo::SFloat.zeros(10, 3072)
#     reshape.forward(x)
#     dy = Xumo::SFloat.zeros(10, 32, 32, 3)
#     assert_equal [10, 3072], reshape.backward(dy).shape
#   end

#   def test_to_hash
#     expected_hash = {
#       class: "DNN::Layers::Reshape",
#       shape: [32, 32, 3],
#     }
#     reshape = Reshape.new([32, 32, 3])
#     assert_equal expected_hash, reshape.to_hash
#   end
# end

# class TestLasso < MiniTest::Unit::TestCase
#   def test_from_hash
#     hash = {
#       class: "DNN::Layers::Lasso",
#       l1_lambda: 0.1,
#     }
#     lasso = Lasso.from_hash(hash)
#     assert_equal 0.1, lasso.l1_lambda
#   end

#   def test_forward
#     lasso = Lasso.new(0.1)
#     assert_equal 0.4, lasso.forward(Xumo::SFloat[-2, 2])
#   end

#   def test_backward
#     lasso = Lasso.new(0.1)
#     lasso.forward(Xumo::SFloat[-2, 2])
#     grad = lasso.backward(1)
#     assert_equal Xumo::SFloat[-0.1, 0.1], grad.round(4)
#   end

#   def test_to_hash
#     expected_hash = {
#       class: "DNN::Layers::Lasso",
#       l1_lambda: 0.01,
#     }
#     lasso = Lasso.new
#     assert_equal expected_hash, lasso.to_hash
#   end
# end

# class TestRidge < MiniTest::Unit::TestCase
#   def test_from_hash
#     hash = {
#       class: "DNN::Layers::Ridge",
#       l2_lambda: 0.1,
#     }
#     ridge = Ridge.from_hash(hash)
#     assert_equal 0.1, ridge.l2_lambda
#   end

#   def test_forward
#     ridge = Ridge.new(0.1)
#     assert_equal 0.4, ridge.forward(Xumo::SFloat[-2, 2])
#   end

#   def test_backward
#     ridge = Ridge.new(0.1)
#     ridge.forward(Xumo::SFloat[-2, 2])
#     grad = ridge.backward(1)
#     assert_equal Xumo::SFloat[-0.2, 0.2], grad.round(4)
#   end

#   def test_to_hash
#     expected_hash = {
#       class: "DNN::Layers::Ridge",
#       l2_lambda: 0.01,
#     }
#     ridge = Ridge.new
#     assert_equal expected_hash, ridge.to_hash
#   end
# end

# class TestDropout < MiniTest::Unit::TestCase
#   def test_from_hash
#     hash = {
#       class: "DNN::Layers::Dropout",
#       dropout_ratio: 0.3,
#       seed: 0,
#       use_scale: false,
#     }
#     dropout = Dropout.from_hash(hash)
#     assert_equal 0.3, dropout.dropout_ratio
#     assert_equal 0, dropout.instance_variable_get(:@seed)
#     assert_equal false, dropout.use_scale
#   end

#   def test_forward
#     dropout = Dropout.new(0.2, seed: 0)
#     dropout.build([100])
#     dropout.set_learning_phase(true)
#     num = dropout.forward(Xumo::SFloat.ones(100)).sum.to_f.round
#     assert num.between?(70, 90)
#   end

#   def test_forward2
#     dropout = Dropout.new(0.3, use_scale: true)
#     dropout.build([1])
#     dropout.set_learning_phase(false)
#     num = dropout.forward(Xumo::SFloat.ones(10)).sum.to_f.round(1)
#     assert_equal 7.0, num
#   end

#   def test_forward3
#     dropout = Dropout.new(0.3, use_scale: false)
#     dropout.build([1])
#     dropout.set_learning_phase(false)
#     num = dropout.forward(Xumo::SFloat.ones(10)).sum.to_f.round(1)
#     assert_equal 10.0, num
#   end

#   def test_backward
#     dropout = Dropout.new
#     dropout.build([1])
#     dropout.set_learning_phase(true)
#     y = dropout.forward(Xumo::SFloat.ones(10))
#     dy = dropout.backward(Xumo::SFloat.ones(10))
#     assert_equal y.round, dy.round
#   end

#   def test_to_hash
#     expected_hash = {
#       class: "DNN::Layers::Dropout",
#       dropout_ratio: 0.3,
#       seed: 0,
#       use_scale: false,
#     }
#     dropout = Dropout.new(0.3, seed: 0, use_scale: false)
#     assert_equal expected_hash, dropout.to_hash
#   end
# end
