require "test_helper"

include DNN
include Activations
include Layers
include Losses
include Optimizers

# TODO
=begin
class TestLoss < MiniTest::Unit::TestCase
  def test_ridge
    dense = Dense.new(1, l2_lambda: 1)
    dense2 = Dense.new(10, l2_lambda: 1)
    dense.build([10])
    output_layer = OutputLayer.new
    dense2.build([1])
    model << output_layer
    model.compile(SGD.new)
    dense.params[:weight].data = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    dense2.params[:weight].data = Numo::SFloat.ones(*dense2.params[:weight].data.shape)
    assert_equal 10.0, output_layer.send(:ridge).round(1)
  end
end
=end


class TestMeanSquaredError < MiniTest::Unit::TestCase
  def test_forward
    loss = MeanSquaredError.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[2, 4]]
    out = loss.forward(x, y)
    assert_equal 6.5, out.round(4)
  end

  def test_backward
    loss = MeanSquaredError.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[2, 4]]
    loss.forward(x, y)
    grad = loss.backward(y)
    assert_equal Xumo::SFloat[[-2, -3]], grad.round(4)
  end
end


class TestMeanAbsoluteError < MiniTest::Unit::TestCase
  def test_forward
    loss = MeanAbsoluteError.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[2, 4]]
    out = loss.forward(x, y)
    assert_equal 5, out.round(4)
  end

  def test_backward
    loss = MeanAbsoluteError.new
    x = Xumo::SFloat[[-1, 2]]
    y = Xumo::SFloat[[2, 4]]
    loss.forward(x, y)
    grad = loss.backward(y)
    assert_equal Xumo::SFloat[[-1, -1]], grad.round(4)
  end
end

=begin
class TestHuberLoss < MiniTest::Unit::TestCase
  def test_forward
    loss = HuberLoss.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[2, 4]]
    out = loss.forward(x, y)
    assert_equal 5, out.round(4)
  end

  def test_forward2
    loss = HuberLoss.new
    x = Xumo::SFloat[[0, 1.0]]
    y = Xumo::SFloat[[0.5, 1.25]]
    out = loss.forward(x, y)
    assert_equal 0.1563, out.round(4)
  end

  def test_backward
    loss = HuberLoss.new
    x = Xumo::SFloat[[-1, 2]]
    y = Xumo::SFloat[[-3, 4]]
    loss.forward(x, y)
    grad = loss.backward(y)
    assert_equal Xumo::SFloat[[1, -1]], grad.round(4)
  end

  def test_backward2
    loss = HuberLoss.new
    x = Xumo::SFloat[[-1, 2]]
    y = Xumo::SFloat[[-0.5, 1.7]]
    loss.forward(x, y)
    grad = loss.backward(y)
    assert_equal Xumo::SFloat[[-0.5, 0.3]], grad.round(4)
  end
end
=end


class TestSoftmaxCrossEntropy < MiniTest::Unit::TestCase
  def test_forward
    loss = SoftmaxCrossEntropy.new
    x = Xumo::SFloat[[0, 1, 2]]
    y = Xumo::SFloat[[0, 0, 1]]
    out = loss.forward(x, y)
    assert_equal 0.4076, out.round(4)
  end

  def test_backward
    loss = SoftmaxCrossEntropy.new
    x = Xumo::SFloat[[0, 1, 2]]
    y = Xumo::SFloat[[0, 0, 1]]
    loss.forward(x, y)
    grad = loss.backward(y)
    assert_equal Xumo::SFloat[[0.09, 0.2447, -0.3348]], grad.round(4)
  end
end


class TestSigmoidCrossEntropy < MiniTest::Unit::TestCase
  def test_loss
    loss = SigmoidCrossEntropy.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[1, 0]]
    out = loss.forward(x, y)
    assert_equal 2.0064, out.round(4)
  end

  def test_backward
    loss = SigmoidCrossEntropy.new
    x = Xumo::SFloat[[0, 1]]
    y = Xumo::SFloat[[1, 0]]
    loss.forward(x, y)
    grad = loss.backward(y)
    assert_equal Xumo::SFloat[[-0.5, 0.7311]], grad.round(4)
  end
end
