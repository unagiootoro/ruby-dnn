require "test_helper"

class TestMeanSquaredError < MiniTest::Unit::TestCase
  # It is matches the expected value of forward.
  def test_forward
    loss = DNN::Losses::MeanSquaredError.new
    y = DNN::Tensor.new(Xumo::SFloat[[0, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[2, 4]])
    out_y = loss.(y, t)
    assert_equal Xumo::SFloat[6.5], out_y.data.round(4)
  end

  # It is matches the expected value of backward.
  def test_backward
    loss = DNN::Losses::MeanSquaredError.new
    y = DNN::Variable.new(Xumo::SFloat[[0, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[2, 4]])
    out_y = loss.(y, t)
    out_y.backward
    assert_equal Xumo::SFloat[[-2, -3]], y.grad.round(4)
  end

  def test_to_hash
    expected_hash = {class: "DNN::Losses::MeanSquaredError"}
    assert_equal expected_hash, DNN::Losses::MeanSquaredError.new.to_hash
  end
end


class TestMeanAbsoluteError < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Losses::MeanAbsoluteError.new
    y = DNN::Tensor.new(Xumo::SFloat[[0, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[2, 4]])
    out_y = loss.(y, t)
    assert_equal Xumo::SFloat[5], out_y.data.round(4)
  end

  def test_backward
    loss = DNN::Losses::MeanAbsoluteError.new
    y = DNN::Variable.new(Xumo::SFloat[[0, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[2, 4]])
    out_y = loss.(y, t)
    out_y.backward
    assert_equal Xumo::SFloat[[-1, -1]], y.grad.round(4)
  end
end


class TestHinge < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Losses::Hinge.new
    y = DNN::Tensor.new(Xumo::SFloat[[1, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[0.7, 1.5]])
    out_y = loss.(y, t)
    assert_equal Xumo::SFloat[0.3], out_y.data.round(4)
  end

  def test_backward
    loss = DNN::Losses::Hinge.new
    y = DNN::Variable.new(Xumo::SFloat[[1, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[0.7, 1.5]])
    out_y = loss.(y, t)
    out_y.backward
    assert_equal Xumo::SFloat[[-0.7, 0]], y.grad.round(4)
  end
end


class TestHuberLoss < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Losses::HuberLoss.new
    y = DNN::Tensor.new(Xumo::SFloat[[0, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[2, 4]])
    out_y = loss.(y, t)
    assert_equal Xumo::SFloat[5], out_y.data.round(4)
  end

  def test_forward2
    loss = DNN::Losses::HuberLoss.new
    y = DNN::Tensor.new(Xumo::SFloat[[0, 1.0]])
    t = DNN::Tensor.new(Xumo::SFloat[[0.5, 1.25]])
    out_y = loss.(y, t)
    assert_equal Xumo::SFloat[0.1563], out_y.data.round(4)
  end

  def test_backward
    loss = DNN::Losses::HuberLoss.new
    y = DNN::Variable.new(Xumo::SFloat[[-1, 2]])
    t = DNN::Tensor.new(Xumo::SFloat[[-3, 4]])
    out_y = loss.(y, t)
    out_y.backward
    assert_equal Xumo::SFloat[[1, -1]], y.grad.round(4)
  end

  def test_backward2
    loss = DNN::Losses::HuberLoss.new
    y = DNN::Variable.new(Xumo::SFloat[[-1, 2]])
    t = DNN::Tensor.new(Xumo::SFloat[[-0.5, 1.7]])
    out_y = loss.(y, t)
    out_y.backward
    assert_equal Xumo::SFloat[[-0.5, 0.3]], y.grad.round(4)
  end
end


class TestSoftmaxCrossEntropy < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {class: "DNN::Losses::SoftmaxCrossEntropy", eps: 1e-4}
    loss = DNN::Losses::SoftmaxCrossEntropy.from_hash(hash)
    assert_equal 1e-4, loss.eps
  end

  def test_forward
    loss = DNN::Losses::SoftmaxCrossEntropy.new
    y = DNN::Tensor.new(Xumo::SFloat[[0, 1, 2]])
    t = DNN::Tensor.new(Xumo::SFloat[[0, 0, 1]])
    out_y = loss.(y, t)
    assert_equal Xumo::SFloat[0.4076], out_y.data.round(4)
  end

  def test_backward
    loss = DNN::Losses::SoftmaxCrossEntropy.new
    y = DNN::Variable.new(Xumo::SFloat[[0, 1, 2]])
    t = DNN::Tensor.new(Xumo::SFloat[[0, 0, 1]])
    out_y = loss.(y, t)
    out_y.backward
    assert_equal Xumo::SFloat[[0.09, 0.2447, -0.3348]], y.grad.round(4)
  end

  def test_to_hash
    expected_hash = {class: "DNN::Losses::SoftmaxCrossEntropy", eps: 1e-7}
    assert_equal expected_hash, DNN::Losses::SoftmaxCrossEntropy.new.to_hash
  end
end


class TestSigmoidCrossEntropy < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {class: "DNN::Losses::SigmoidCrossEntropy", eps: 1e-4}
    loss = DNN::Losses::SigmoidCrossEntropy.from_hash(hash)
    assert_equal 1e-4, loss.eps
  end

  def test_forward
    loss = DNN::Losses::SigmoidCrossEntropy.new
    y = DNN::Tensor.new(Xumo::SFloat[[0, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[1, 0]])
    out_y = loss.(y, t)
    assert_equal Xumo::SFloat[2.0064], out_y.data.round(4)
  end

  def test_backward
    loss = DNN::Losses::SigmoidCrossEntropy.new
    y = DNN::Variable.new(Xumo::SFloat[[0, 1]])
    t = DNN::Tensor.new(Xumo::SFloat[[1, 0]])
    out_y = loss.(y, t)
    out_y.backward
    assert_equal Xumo::SFloat[[-0.5, 0.7311]], y.grad.round(4)
  end

  def test_to_hash
    expected_hash = {class: "DNN::Losses::SigmoidCrossEntropy", eps: 1e-7}
    assert_equal expected_hash, DNN::Losses::SigmoidCrossEntropy.new.to_hash
  end
end
