require "test_helper"

class TestMeanSquaredError < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Functions::MeanSquaredError.new
    y = Xumo::SFloat[[0, 1]]
    t = Xumo::SFloat[[2, 4]]
    out_y = loss.forward(y, t)
    assert_equal 6.5, out_y.round(4)
  end

  def test_backward
    loss = DNN::Functions::MeanSquaredError.new
    y = Xumo::SFloat[[0, 1]]
    t = Xumo::SFloat[[2, 4]]
    loss.forward(y, t)
    grad = loss.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[[-2, -3]], grad.round(4)
  end
end


class TestMeanAbsoluteError < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Functions::MeanAbsoluteError.new
    y = Xumo::SFloat[[0, 1]]
    t = Xumo::SFloat[[2, 4]]
    out_y = loss.forward(y, t)
    assert_equal 5, out_y.round(4)
  end

  def test_backward
    loss = DNN::Functions::MeanAbsoluteError.new
    y = Xumo::SFloat[[-1, 2]]
    t = Xumo::SFloat[[2, 4]]
    loss.forward(y, t)
    grad = loss.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[[-1, -1]], grad.round(4)
  end
end


class TestHinge < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Functions::Hinge.new
    y = Xumo::SFloat[[1, 1]]
    t = Xumo::SFloat[[0.7, 1.5]]
    out_y = loss.forward(y, t)
    assert_equal 0.3, out_y.round(4)
  end

  def test_backward
    loss = DNN::Functions::Hinge.new
    y = Xumo::SFloat[[1, 1]]
    t = Xumo::SFloat[[0.7, 1.5]]
    loss.forward(y, t)
    grad = loss.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[[-0.7, 0]], grad.round(4)
  end
end


class TestHuberLoss < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Functions::HuberLoss.new
    y = Xumo::SFloat[[0, 1]]
    t = Xumo::SFloat[[2, 4]]
    out_y = loss.forward(y, t)
    assert_equal 5, out_y.round(4)
  end

  def test_forward2
    loss = DNN::Functions::HuberLoss.new
    y = Xumo::SFloat[[0, 1.0]]
    t = Xumo::SFloat[[0.5, 1.25]]
    out_y = loss.forward(y, t)
    assert_equal 0.1563, out_y.round(4)
  end

  def test_backward
    loss = DNN::Functions::HuberLoss.new
    y = Xumo::SFloat[[-1, 2]]
    t = Xumo::SFloat[[-3, 4]]
    loss.forward(y, t)
    grad = loss.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[[1, -1]], grad.round(4)
  end

  def test_backward2
    loss = DNN::Functions::HuberLoss.new
    y = Xumo::SFloat[[-1, 2]]
    t = Xumo::SFloat[[-0.5, 1.7]]
    loss.forward(y, t)
    grad = loss.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[[-0.5, 0.3]], grad.round(4)
  end
end


class TestSoftmaxCrossEntropy < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Functions::SoftmaxCrossEntropy.new
    y = Xumo::SFloat[[0, 1, 2]]
    t = Xumo::SFloat[[0, 0, 1]]
    out_y = loss.forward(y, t)
    assert_equal 0.4076, out_y.round(4)
  end

  def test_backward
    loss = DNN::Functions::SoftmaxCrossEntropy.new
    y = Xumo::SFloat[[0, 1, 2]]
    t = Xumo::SFloat[[0, 0, 1]]
    loss.forward(y, t)
    grad = loss.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[[0.09, 0.2447, -0.3348]], grad.round(4)
  end
end


class TestSigmoidCrossEntropy < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Functions::SigmoidCrossEntropy.new
    y = Xumo::SFloat[[0, 1]]
    t = Xumo::SFloat[[1, 0]]
    out_y = loss.forward(y, t)
    assert_equal 2.0064, out_y.round(4)
  end

  def test_backward
    loss = DNN::Functions::SigmoidCrossEntropy.new
    y = Xumo::SFloat[[0, 1]]
    t = Xumo::SFloat[[1, 0]]
    loss.forward(y, t)
    grad = loss.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[[-0.5, 0.7311]], grad.round(4)
  end
end
