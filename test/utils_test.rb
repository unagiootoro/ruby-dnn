require "test_helper"

Utils = DNN::Utils

class TestUtils < MiniTest::Unit::TestCase
  def test_to_categorical
    y = Xumo::NArray[0, 3]
    y2 = Utils.to_categorical(y, 5)
    assert_equal Xumo::NArray[[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], y2
  end

  def test_to_categorical2
    y = Xumo::NArray[0, 3]
    y2 = Utils.to_categorical(y, 5, Xumo::SFloat)
    assert_kind_of Xumo::SFloat, y2
  end

  def test_hash_to_obj
    relu = Utils.hash_to_obj({class: "DNN::Layers::ReLU"})
    assert_kind_of DNN::Layers::ReLU, relu
  end

  def test_hash_to_obj2
    lrelu = Utils.hash_to_obj({class: "DNN::Layers::LeakyReLU", alpha: 0.2})
    assert_equal 0.2, lrelu.alpha
  end
end
