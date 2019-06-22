require "test_helper"

include DNN
include Activations
include Layers
include Optimizers
include Losses


class TestMeanSquaredError < MiniTest::Unit::TestCase
  def test_forward
    loss = MeanSquaredError.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[2, 4]]
    out_x = loss.forward(x, y, [])
    assert_equal 6.5, out_x.round(4)
  end

  def test_forward2
    loss = MeanSquaredError.new
    dense = Dense.new(1, weight_regularizer: L1L2.new(1, 1))
    dense.build([10])
    dense.params[:weight].data = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    out_x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[0, 1]]
    assert_equal 15, loss.forward(out_x, y, [dense]).round(4)
  end

  def test_backward
    loss = MeanSquaredError.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[2, 4]]
    loss.forward(x, y, [])
    grad = loss.backward(y, [])
    assert_equal Xumo::SFloat[[-2, -3]], grad.round(4)
  end

  def backward
    loss = MeanSquaredError.new
    dense = Dense.new(2, weight_regularizer: L1L2.new(1, 1))
    dense.build([1])
    dense.params[:weight].data = Numo::SFloat[[-2, 2]]
    dense.params[:weight].grad = Numo::SFloat.zeros(*dense.params[:weight].data.shape)
    loss.forward(0)
    loss.backward(0, [InputLayer.new(1), dense])
    assert_equal Numo::SFloat[[-3, 3]], dense.params[:weight].grad.round(4)
  end

  def test_to_hash
    expected_hash = {class: "DNN::Losses::MeanSquaredError"}
    assert_equal expected_hash, MeanSquaredError.new.to_hash
  end
end


class TestMeanAbsoluteError < MiniTest::Unit::TestCase
  def test_forward
    loss = MeanAbsoluteError.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[2, 4]]
    out_x = loss.forward(x, y, [])
    assert_equal 5, out_x.round(4)
  end

  def test_backward
    loss = MeanAbsoluteError.new
    x = Xumo::SFloat[[-1, 2]]
    y = Xumo::SFloat[[2, 4]]
    loss.forward(x, y, [])
    grad = loss.backward(y, [])
    assert_equal Xumo::SFloat[[-1, -1]], grad.round(4)
  end
end


class TestHuberLoss < MiniTest::Unit::TestCase
  def test_forward
    loss = HuberLoss.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[2, 4]]
    out_x = loss.forward(x, y, [])
    assert_equal 5, out_x.round(4)
  end

  def test_forward2
    loss = HuberLoss.new
    x = Xumo::SFloat[[0, 1.0]]
    y = Xumo::SFloat[[0.5, 1.25]]
    out_x = loss.forward(x, y, [])
    assert_equal 0.1563, out_x.round(4)
  end

  def test_backward
    loss = HuberLoss.new
    x = Xumo::SFloat[[-1, 2]]
    y = Xumo::SFloat[[-3, 4]]
    loss.forward(x, y, [])
    grad = loss.backward(y, [])
    assert_equal Xumo::SFloat[[1, -1]], grad.round(4)
  end

  def test_backward2
    loss = HuberLoss.new
    x = Xumo::SFloat[[-1, 2]]
    y = Xumo::SFloat[[-0.5, 1.7]]
    loss.forward(x, y, [])
    grad = loss.backward(y, [])
    assert_equal Xumo::SFloat[[-0.5, 0.3]], grad.round(4)
  end
end


class TestSoftmaxCrossEntropy < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {class: "DNN::Losses::SoftmaxCrossEntropy", eps: 1e-4}
    loss = SoftmaxCrossEntropy.from_hash(hash)
    assert_equal 1e-4, loss.eps
  end

  def test_forward
    loss = SoftmaxCrossEntropy.new
    x = Xumo::SFloat[[0, 1, 2]]
    y = Xumo::SFloat[[0, 0, 1]]
    out_x = loss.forward(x, y, [])
    assert_equal 0.4076, out_x.round(4)
  end

  def test_backward
    loss = SoftmaxCrossEntropy.new
    x = Xumo::SFloat[[0, 1, 2]]
    y = Xumo::SFloat[[0, 0, 1]]
    loss.forward(x, y, [])
    grad = loss.backward(y, [])
    assert_equal Xumo::SFloat[[0.09, 0.2447, -0.3348]], grad.round(4)
  end

  def test_to_hash
    expected_hash = {class: "DNN::Losses::SoftmaxCrossEntropy", eps: 1e-7}
    assert_equal expected_hash, SoftmaxCrossEntropy.new.to_hash
  end
end


class TestSigmoidCrossEntropy < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {class: "DNN::Losses::SigmoidCrossEntropy", eps: 1e-4}
    loss = SigmoidCrossEntropy.from_hash(hash)
    assert_equal 1e-4, loss.eps
  end

  def test_forward
    loss = SigmoidCrossEntropy.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[1, 0]]
    out_x = loss.forward(x, y, [])
    assert_equal Xumo::SFloat[[0.6931, 1.3133]], out_x.round(4)
  end

  def test_backward
    loss = SigmoidCrossEntropy.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[1, 0]]
    loss.forward(x, y, [])
    grad = loss.backward(y, [])
    assert_equal Xumo::SFloat[[-0.5, 0.7311]], grad.round(4)
  end

  def test_to_hash
    expected_hash = {class: "DNN::Losses::SigmoidCrossEntropy", eps: 1e-7}
    assert_equal expected_hash, SigmoidCrossEntropy.new.to_hash
  end
end
