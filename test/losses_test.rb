require "test_helper"

class TestMeanSquaredError < MiniTest::Unit::TestCase
  def test_loss
    loss = DNN::Losses::MeanSquaredError.new
    y = Numo::SFloat[[0, 1]]
    t = Numo::SFloat[[2, 4]]
    assert_equal Numo::SFloat[6.5], loss.loss(y, t).round(4)
  end

  def test_loss2
    loss = DNN::Losses::MeanSquaredError.new
    dense = DNN::Layers::Dense.new(1, weight_regularizer: DNN::Regularizers::L1L2.new(1, 1))
    dense.build([10])
    dense.weight.data = Numo::SFloat.ones(*dense.weight.data.shape)
    out_y = Numo::SFloat[[0, 1]]
    t = Numo::SFloat[[0, 1]]
    assert_equal Numo::SFloat[15], loss.loss(out_y, t, [dense]).round(4)
  end

  # It is matches the expected value of forward.
  def test_forward
    loss = DNN::Losses::MeanSquaredError.new
    y = Numo::SFloat[[0, 1]]
    t = Numo::SFloat[[2, 4]]
    out_y = loss.forward(y, t)
    assert_equal 6.5, out_y.round(4)
  end

  # It is matches the expected value of backward.
  def test_backward
    loss = DNN::Losses::MeanSquaredError.new
    y = Numo::SFloat[[0, 1]]
    t = Numo::SFloat[[2, 4]]
    loss.forward(y, t)
    grad = loss.backward(y, t)
    assert_equal Numo::SFloat[[-2, -3]], grad.round(4)
  end

  # It is matches the expected value of forward and use regularizer.
  def test_regularizers_forward
    loss = DNN::Losses::MeanSquaredError.new
    dense = DNN::Layers::Dense.new(1, weight_regularizer: DNN::Regularizers::L1L2.new(1, 1))
    dense.build([10])
    dense.weight.data = Numo::SFloat.ones(*dense.weight.data.shape)
    assert_equal 15, loss.regularizers_forward([DNN::Layers::InputLayer.new(1), dense]).round(4)
  end

  # It is matches the expected value of backward and use regularizer.
  def test_regularizers_backward
    loss = DNN::Losses::MeanSquaredError.new
    dense = DNN::Layers::Dense.new(2, weight_regularizer: DNN::Regularizers::L1L2.new(1, 1))
    dense.build([1])
    dense.weight.data = Numo::SFloat[[-2, 2]]
    dense.weight.grad = Numo::SFloat.zeros(*dense.weight.data.shape)
    loss.regularizers_backward([DNN::Layers::InputLayer.new(1), dense])
    assert_equal Numo::SFloat[[-3, 3]], dense.weight.grad.round(4)
  end

  def test_to_hash
    expected_hash = {class: "DNN::Losses::MeanSquaredError"}
    assert_equal expected_hash, DNN::Losses::MeanSquaredError.new.to_hash
  end
end


class TestMeanAbsoluteError < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Losses::MeanAbsoluteError.new
    y = Numo::SFloat[[0, 1]]
    t = Numo::SFloat[[2, 4]]
    out_y = loss.forward(y, t)
    assert_equal 5, out_y.round(4)
  end

  def test_backward
    loss = DNN::Losses::MeanAbsoluteError.new
    y = Numo::SFloat[[-1, 2]]
    t = Numo::SFloat[[2, 4]]
    loss.forward(y, t)
    grad = loss.backward(y, t)
    assert_equal Numo::SFloat[[-1, -1]], grad.round(4)
  end
end


class TestHinge < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Losses::Hinge.new
    y = Numo::SFloat[[1, 1]]
    t = Numo::SFloat[[0.7, 1.5]]
    out_y = loss.forward(y, t)
    assert_equal Numo::SFloat[[0.3, 0]], out_y.round(4)
  end

  def test_backward
    loss = DNN::Losses::Hinge.new
    y = Numo::SFloat[[1, 1]]
    t = Numo::SFloat[[0.7, 1.5]]
    loss.forward(y, t)
    grad = loss.backward(y, t)
    assert_equal Numo::SFloat[[-0.7, 0]], grad.round(4)
  end
end


class TestHuberLoss < MiniTest::Unit::TestCase
  def test_forward
    loss = DNN::Losses::HuberLoss.new
    y = Numo::SFloat[[0, 1]]
    t = Numo::SFloat[[2, 4]]
    out_y = loss.forward(y, t)
    assert_equal 5, out_y.round(4)
  end

  def test_forward2
    loss = DNN::Losses::HuberLoss.new
    y = Numo::SFloat[[0, 1.0]]
    t = Numo::SFloat[[0.5, 1.25]]
    out_y = loss.forward(y, t)
    assert_equal 0.1563, out_y.round(4)
  end

  def test_backward
    loss = DNN::Losses::HuberLoss.new
    y = Numo::SFloat[[-1, 2]]
    t = Numo::SFloat[[-3, 4]]
    loss.forward(y, t)
    grad = loss.backward(y, t)
    assert_equal Numo::SFloat[[1, -1]], grad.round(4)
  end

  def test_backward2
    loss = DNN::Losses::HuberLoss.new
    y = Numo::SFloat[[-1, 2]]
    t = Numo::SFloat[[-0.5, 1.7]]
    loss.forward(y, t)
    grad = loss.backward(y, t)
    assert_equal Numo::SFloat[[-0.5, 0.3]], grad.round(4)
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
    y = Numo::SFloat[[0, 1, 2]]
    t = Numo::SFloat[[0, 0, 1]]
    out_y = loss.forward(y, t)
    assert_equal 0.4076, out_y.round(4)
  end

  def test_backward
    loss = DNN::Losses::SoftmaxCrossEntropy.new
    y = Numo::SFloat[[0, 1, 2]]
    t = Numo::SFloat[[0, 0, 1]]
    loss.forward(y, t)
    grad = loss.backward(y, t)
    assert_equal Numo::SFloat[[0.09, 0.2447, -0.3348]], grad.round(4)
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
    y = Numo::SFloat[[0, 1]]
    t = Numo::SFloat[[1, 0]]
    out_y = loss.forward(y, t)
    assert_equal Numo::SFloat[[0.6931, 1.3133]], out_y.round(4)
  end

  def test_backward
    loss = DNN::Losses::SigmoidCrossEntropy.new
    y = Numo::SFloat[[0, 1]]
    t = Numo::SFloat[[1, 0]]
    loss.forward(y, t)
    grad = loss.backward(y, t)
    assert_equal Numo::SFloat[[-0.5, 0.7311]], grad.round(4)
  end

  def test_to_hash
    expected_hash = {class: "DNN::Losses::SigmoidCrossEntropy", eps: 1e-7}
    assert_equal expected_hash, DNN::Losses::SigmoidCrossEntropy.new.to_hash
  end
end
