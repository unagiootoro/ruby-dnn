require "test_helper"

include DNN::Functions
include DNN::Optimizers

class TestBatchNormalization < MiniTest::Unit::TestCase
  def test_forward
    batch_norm = BatchNormalization.new(Xumo::SFloat[0], Xumo::SFloat[0], learning_phase: true)
    gamma = Xumo::SFloat.new(10).fill(3)
    beta = Xumo::SFloat.new(10).fill(10)
    x = Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(10), Xumo::SFloat.new(10).fill(20)])
    expected = Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(7), Xumo::SFloat.new(10).fill(13)])
    assert_equal expected, batch_norm.forward(x, gamma, beta).round(4)
  end

  def test_forward2
    batch_norm = BatchNormalization.new(Xumo::SFloat.new(10).fill(15), Xumo::SFloat.new(10).fill(25), learning_phase: false)
    gamma = Xumo::SFloat.new(10).fill(3)
    beta = Xumo::SFloat.new(10).fill(10)
    x = Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(10), Xumo::SFloat.new(10).fill(20)])
    expected = Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(7), Xumo::SFloat.new(10).fill(13)])
    assert_equal expected, batch_norm.forward(x, gamma, beta).round(4)
  end

  def test_backward
    batch_norm = BatchNormalization.new(Xumo::SFloat[0], Xumo::SFloat[0], learning_phase: true)
    gamma = Xumo::SFloat.new(10).fill(3)
    beta = Xumo::SFloat.new(10).fill(10)
    x = Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(10), Xumo::SFloat.new(10).fill(20)])
    batch_norm.forward(x, gamma, beta)
    dx, dgamma, dbeta = batch_norm.backward(Xumo::SFloat.ones(*x.shape))
    assert_equal Xumo::SFloat[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dx.round(4)
    assert_equal Xumo::SFloat[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dgamma
    assert_equal Xumo::SFloat[[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], dbeta
  end
end
