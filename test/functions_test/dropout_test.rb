require "test_helper"

include DNN::Functions

class TestDropout < MiniTest::Unit::TestCase
  def test_forward
    dropout = Dropout.new(0.2, seed: 0, learning_phase: true)
    num = dropout.forward(Xumo::SFloat.ones(100)).sum.to_f.round
    assert num.between?(70, 90)
  end

  def test_forward2
    dropout = Dropout.new(0.3, use_scale: true, learning_phase: false)
    num = dropout.forward(Xumo::SFloat.ones(10)).sum.to_f.round(1)
    assert_equal 7.0, num
  end

  def test_forward3
    dropout = Dropout.new(0.3, use_scale: false, learning_phase: false)
    num = dropout.forward(Xumo::SFloat.ones(10)).sum.to_f.round(1)
    assert_equal 10.0, num
  end

  def test_backward
    dropout = Dropout.new(0.5, learning_phase: true)
    y = dropout.forward(Xumo::SFloat.ones(10))
    dy = dropout.backward(Xumo::SFloat.ones(10))
    assert_equal y.round, dy.round
  end
end
