require "test_helper"

include DNN
include Activations

class TestUtils < MiniTest::Unit::TestCase
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
    relu = Utils.from_hash({class: "DNN::Activations::ReLU"})
    assert_kind_of ReLU, relu
  end

  def test_load_hash2
    lrelu = Utils.from_hash({class: "DNN::Activations::LeakyReLU", alpha: 0.2})
    assert_equal 0.2, lrelu.alpha
  end

  def test_sigmoid
    out = Utils.sigmoid(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0.5, 0.7311], out.round(4)
  end

  def test_softmax
    out = Utils.softmax(Xumo::SFloat[[0, 1, 2]])
    assert_equal Xumo::SFloat[[0.09, 0.2447, 0.6652]], out.round(4)
  end
end
