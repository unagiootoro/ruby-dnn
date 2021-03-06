require "test_helper"

class TestSplit < MiniTest::Unit::TestCase
  def test_from_hash
    split = DNN::Layers::Split.new(axis: 0, dim: 2)
    assert_equal 0, split.axis
    assert_equal 2, split.dim
  end

  def test_forward_node
    split = DNN::Layers::Split.new(dim: 2)
    y1, y2 = split.forward_node(Xumo::SFloat[[1, 2, 3, 4, 5]])
    assert_equal Xumo::SFloat[[1, 2]], y1.round(4)
    assert_equal Xumo::SFloat[[3, 4, 5]], y2.round(4)
  end

  def test_backward_node
    split = DNN::Layers::Split.new(dim: 3)
    split.forward_node(Xumo::SFloat[[6, 7, 8, 9, 10]])
    dx = split.backward_node(Xumo::SFloat[[1, 2, 3]], Xumo::SFloat[[4, 5]])
    assert_equal Xumo::SFloat[[1, 2, 3, 4, 5]], dx
  end

  def test_to_hash
    expected_hash = { class: "DNN::Layers::Split", axis: 1, dim: 2 }
    split = DNN::Layers::Split.new(dim: 2)
    assert_equal expected_hash, split.to_hash
  end

  def test_backward
    split = DNN::Layers::Split.new(dim: 2)
    x = DNN::Param.new(Xumo::SFloat[[6, 7, 8, 9, 10]], Xumo::SFloat[0])
    y1, y2 = split.(x)
    y1.link.backward(Xumo::SFloat[[1, 2, 3]])
    y2.link.backward(Xumo::SFloat[[4, 5]])
    assert_equal Xumo::SFloat[[1, 2, 3, 4, 5]], x.grad
  end
end
