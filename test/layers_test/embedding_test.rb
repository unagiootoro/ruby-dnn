require "test_helper"

include DNN::Layers
include DNN::Optimizers
include DNN::Initializers
include DNN::Regularizers

class TestEmbedding < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::Embedding",
      input_shape: [10],
      input_length: 5,
      weight_initializer: RandomNormal.new.to_hash,
      weight_regularizer: L2.new.to_hash,
      mask_zero: true,
    }
    embed = Embedding.from_hash(hash)
    assert_equal [10], embed.input_shape
    assert_equal 5, embed.input_length
    assert_kind_of RandomNormal, embed.weight_initializer
    assert_kind_of L2, embed.weight_regularizer
    assert_equal true, embed.mask_zero
  end

  def test_forward_node
    embed = Embedding.new(2, 3)
    embed.build([2])
    embed.weight.data = Numo::SFloat.cast([0.1, 0.2, 0.3])
    x = Numo::Int32.cast([[0, 1], [0, 2]])
    expected = Numo::SFloat.cast([[0.1, 0.2], [0.1, 0.3]])
    assert_equal expected, embed.forward_node(x).round(4)
  end

  # Test mask zero.
  def test_forward_node2
    embed = Embedding.new(2, 3, mask_zero: true)
    embed.build([2])
    embed.weight.data = Numo::SFloat.cast([0.1, 0.2, 0.3])
    x = Numo::Int32.cast([[0, 1], [0, 2]])
    expected = Numo::SFloat.cast([[0, 0.2], [0, 0.3]])
    assert_equal expected, embed.forward_node(x).round(4)
  end

  def test_backward_node
    embed = Embedding.new(2, 3)
    embed.build([2])
    embed.weight.data = Numo::SFloat.cast([0.1, 0.2, 0.3])
    x = Numo::Int32.cast([[0, 1], [2, 2]])
    dy = Numo::SFloat.cast([[0.1, 0.2], [0.1, 0.3]])
    expected = Numo::SFloat.cast([0.1, 0.2, 0.4])
    embed.forward_node(x)
    embed.backward_node(dy)
    assert_equal expected, embed.weight.grad.round(4)
  end

  # Test mask zero.
  def test_backward_node2
    embed = Embedding.new(2, 3, mask_zero: true)
    embed.build([2])
    embed.weight.data = Numo::SFloat.cast([0.1, 0.2, 0.3])
    x = Numo::Int32.cast([[0, 1], [2, 2]])
    dy = Numo::SFloat.cast([[0.1, 0.2], [0.1, 0.3]])
    expected = Numo::SFloat.cast([0, 0.2, 0.4])
    embed.forward_node(x)
    embed.backward_node(dy)
    assert_equal expected, embed.weight.grad.round(4)
  end

  def test_regularizers
    embed = Embedding.new(2, 3, weight_regularizer: L1.new)
    embed.build([2])
    assert_kind_of L1, embed.regularizers[0]
  end

  def test_regularizers2
    embed = Embedding.new(2, 3)
    embed.build([2])
    assert_equal [], embed.regularizers
  end

  def test_to_hash
    embed = Embedding.new(10, 5)
    embed.build([10])
    expected_hash = {
      class: "DNN::Layers::Embedding",
      input_shape: [10],
      input_length: 5,
      weight_initializer: embed.weight_initializer.to_hash,
      weight_regularizer: embed.weight_regularizer&.to_hash,
      mask_zero: false,
    }
    assert_equal expected_hash, embed.to_hash
  end

  def test_get_params
    embed = Embedding.new(10, 5)
    embed.build([10])
    expected_hash = {
      weight: embed.weight,
    }
    assert_equal expected_hash, embed.get_params
  end
end
