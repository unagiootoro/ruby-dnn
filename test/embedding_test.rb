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
    }
    embed = Embedding.from_hash(hash)
    assert_equal [10], embed.input_shape
    assert_equal 5, embed.input_length
    assert_kind_of RandomNormal, embed.weight_initializer
    assert_kind_of L2, embed.weight_regularizer
  end

  def test_forward
    embed = Embedding.new(2, 3)
    embed.build
    embed.weight.data = Numo::SFloat.cast([0.1, 0.2, 0.3])
    x = Numo::Int32.cast([[0, 1], [0, 2]])
    expected = Numo::SFloat.cast([[0.1, 0.2], [0.1, 0.3]])
    assert_equal expected, embed.forward(x).round(4)
  end

  def test_backward
    embed = Embedding.new(2, 3)
    embed.build
    embed.weight.data = Numo::SFloat.cast([0.1, 0.2, 0.3])
    x = Numo::Int32.cast([[0, 1], [2, 2]])
    dy = Numo::SFloat.cast([[0.1, 0.2], [0.1, 0.3]])
    expected = Numo::SFloat.cast([0.1, 0.2, 0.4])
    embed.forward(x)
    embed.backward(dy)
    assert_equal expected, embed.weight.grad.round(4)
  end

  def test_regularizers
    embed = Embedding.new(2, 3, weight_regularizer: L1.new)
    embed.build
    assert_kind_of L1, embed.regularizers[0]
  end

  def test_regularizers2
    embed = Embedding.new(2, 3)
    embed.build
    assert_equal [], embed.regularizers
  end

  def test_to_hash
    embed = Embedding.new(10, 5)
    embed.build
    expected_hash = {
      class: "DNN::Layers::Embedding",
      name: nil,
      input_shape: [10],
      input_length: 5,
      weight_initializer: embed.weight_initializer.to_hash,
      weight_regularizer: embed.weight_regularizer&.to_hash,
    }
    assert_equal expected_hash, embed.to_hash
  end

  def test_get_params
    embed = Embedding.new(10, 5)
    embed.build
    expected_hash = {
      weight: embed.weight,
    }
    assert_equal expected_hash, embed.get_params
  end
end
