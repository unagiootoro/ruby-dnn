require "test_helper"

include Numo
include DNN::Activations
Util = DNN::Util

class TestUtil < MiniTest::Unit::TestCase
  def test_getminibatch
    x = NArray[[1], [2], [3], [4]]
    batch1, batch2 = Util.get_minibatch(x, x, 2)
    assert_equal 2, batch1.shape[0]
  end

  def test_getminibatch2
    x = NArray[[1], [2], [3], [4]]
    batch1, batch2 = Util.get_minibatch(x, x, 2)
    assert_equal batch1, batch2
  end

  def test_to_categorical
    y = NArray[0, 3]
    y2 = Util.to_categorical(y, 5)
    assert_equal NArray[[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], y2
  end

  def test_to_categorical2
    y = NArray[0, 3]
    y2 = Util.to_categorical(y, 5, SFloat)
    assert_kind_of SFloat, y2
  end

  def test_numerical_grad
    x = DFloat[1, 3]
    func = ->x { x**2 }
    n_grad = Util.numerical_grad(x, func)
    assert_equal DFloat[2, 6], n_grad.round(4)
  end

  def test_load_hash
    relu = Util.load_hash({name: "DNN::Activations::ReLU"})
    assert_kind_of ReLU, relu
  end

  def test_load_hash2
    lrelu = Util.load_hash({name: "DNN::Activations::LeakyReLU", alpha: 0.2})
    assert_equal 0.2, lrelu.alpha
  end
end
