require "test_helper"

include DNN
include Activations

class TestUtil < MiniTest::Unit::TestCase
  def test_getminibatch
    x = Numo::NArray[[1], [2], [3], [4]]
    batch1, batch2 = Util.get_minibatch(x, x, 2)
    assert_equal 2, batch1.shape[0]
  end

  def test_getminibatch2
    x = Numo::NArray[[1], [2], [3], [4]]
    batch1, batch2 = Util.get_minibatch(x, x, 2)
    assert_equal batch1, batch2
  end

  def test_to_categorical
    y = Numo::NArray[0, 3]
    y2 = Util.to_categorical(y, 5)
    assert_equal Numo::NArray[[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], y2
  end

  def test_to_categorical2
    y = Numo::NArray[0, 3]
    y2 = Util.to_categorical(y, 5, Numo::SFloat)
    assert_kind_of Numo::SFloat, y2
  end

  def test_numerical_grad
    x = Numo::DFloat[1, 3]
    func = ->x { x**2 }
    n_grad = Util.numerical_grad(x, func)
    assert_equal Numo::DFloat[2, 6], n_grad.round(4)
  end

  def test_load_hash
    relu = Util.load_hash({class: "DNN::Activations::ReLU"})
    assert_kind_of ReLU, relu
  end

  def test_load_hash2
    lrelu = Util.load_hash({class: "DNN::Activations::LeakyReLU", alpha: 0.2})
    assert_equal 0.2, lrelu.alpha
  end
end
