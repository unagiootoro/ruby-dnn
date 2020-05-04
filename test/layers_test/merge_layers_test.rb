require "test_helper"

class TestConcatenate < MiniTest::Unit::TestCase
  def test_from_hash
    con = DNN::Layers::Concatenate.new(axis: 0)
    assert_equal 0, con.axis
  end

  def test_forward_node
    con = DNN::Layers::Concatenate.new
    y = con.forward_node(Xumo::SFloat[[1, 2]], Xumo::SFloat[[3, 4]])
    assert_equal Xumo::SFloat[[1, 2, 3, 4]], y.round(4)
  end

  def test_backward_node
    con = DNN::Layers::Concatenate.new
    con.forward_node(Xumo::SFloat[[1, 2, 3]], Xumo::SFloat[[4, 5]])
    dx1, dx2 = con.backward_node(Xumo::SFloat[[6, 7, 8, 9, 10]])
    assert_equal Xumo::SFloat[[6, 7, 8]], dx1
    assert_equal Xumo::SFloat[[9, 10]], dx2
  end

  def test_to_hash
    expected_hash = { class: "DNN::Layers::Concatenate", axis: 1 }
    con = DNN::Layers::Concatenate.new
    assert_equal expected_hash, con.to_hash
  end
end
