require "test_helper"

include DNN
include Activations

class TestUtils < MiniTest::Unit::TestCase
  def test_getminibatch
    x = Numo::NArray[[1], [2], [3], [4]]
    batch1, batch2 = Utils.get_minibatch(x, x, 2)
    assert_equal 2, batch1.shape[0]
  end

  def test_getminibatch2
    x = Numo::NArray[[1], [2], [3], [4]]
    batch1, batch2 = Utils.get_minibatch(x, x, 2)
    assert_equal batch1, batch2
  end

  def test_to_categorical
    y = Numo::NArray[0, 3]
    y2 = Utils.to_categorical(y, 5)
    assert_equal Numo::NArray[[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], y2
  end

  def test_to_categorical2
    y = Numo::NArray[0, 3]
    y2 = Utils.to_categorical(y, 5, Numo::SFloat)
    assert_kind_of Numo::SFloat, y2
  end

  def test_load_hash
    relu = Utils.load_hash({class: "DNN::Activations::ReLU"})
    assert_kind_of ReLU, relu
  end

  def test_load_hash2
    lrelu = Utils.load_hash({class: "DNN::Activations::LeakyReLU", alpha: 0.2})
    assert_equal 0.2, lrelu.alpha
  end
end
