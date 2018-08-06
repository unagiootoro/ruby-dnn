require "test_helper"

include Numo
include DNN::Activations
include DNN::Layers
include DNN::Optimizers
Model = DNN::Model
Util = DNN::Util

class TestSigmoid < MiniTest::Unit::TestCase
  # f = ->x { 1 / (1 + NMath.exp(-x)) }
  def test_forward
    sigmoid = Sigmoid.new
    out = sigmoid.forward(DFloat[0, 1])
    assert_equal DFloat[0.5, 0.7311], out.round(4)
  end

  def test_backward
    sigmoid = Sigmoid.new
    x = DFloat[0, 1]
    sigmoid.forward(x)
    grad = sigmoid.backward(1).round(4)
    n_grad = Util.numerical_grad(x, sigmoid.method(:forward)).round(4)
    assert_equal n_grad, grad
  end
end


class TestTanh < MiniTest::Unit::TestCase
  # f = ->x { (NMath.exp(x) - NMath.exp(-x)) / (NMath.exp(x) + NMath.exp(-x)) }
  def test_forward
    tanh = Tanh.new
    out = tanh.forward(DFloat[0, 1])
    assert_equal DFloat[0, 0.7616], out.round(4)
  end

  def test_backward
    tanh = Tanh.new
    x = DFloat[0, 1]
    tanh.forward(x)
    grad = tanh.backward(1).round(4)
    n_grad = Util.numerical_grad(x, tanh.method(:forward)).round(4)
    assert_equal n_grad, grad
  end
end


class TestReLU < MiniTest::Unit::TestCase
  # f = ->x { x > 0 ? x : 0 }
  def test_forward
    relu = ReLU.new
    out = relu.forward(DFloat[-2, 0, 2])
    assert_equal DFloat[0, 0, 2], out
  end

  def test_backward
    relu = ReLU.new
    relu.forward(DFloat[-2, 0, 2])
    grad = relu.backward(1).round(4)
    assert_equal DFloat[0, 0, 1], grad
  end
end


class TestLeakyReLU < MiniTest::Unit::TestCase
  # f = ->x { x > 0 ? x : 0.3 * x }
  def test_load_hash
    hash = {alpha: 0.2}
    lrelu = LeakyReLU.load_hash(hash)
    assert_equal 0.2, lrelu.alpha
  end

  def test_forward
    lrelu = LeakyReLU.new
    out = lrelu.forward(DFloat[-2, 0, 2])
    assert_equal DFloat[-0.6, 0, 2], out.round(2)
  end

  def test_backward
    lrelu = LeakyReLU.new
    lrelu.forward(DFloat[-2, 0, 2])
    grad = lrelu.backward(1).round(4)
    assert_equal DFloat[0.3, 0.3, 1], grad
  end

  def test_backward2
    lrelu = LeakyReLU.new
    x = DFloat[-2, 2]
    lrelu.forward(x)
    grad = lrelu.backward(1).round(4)
    n_grad = Util.numerical_grad(x, lrelu.method(:forward)).round(4)
    assert_equal n_grad, grad
  end

  def test_to_hash
    lrelu = LeakyReLU.new
    expected_hash = {name: "DNN::Activations::LeakyReLU", alpha: 0.3}
    assert_equal expected_hash, lrelu.to_hash
  end
end


class TestIdentityMSE < MiniTest::Unit::TestCase
  # f = ->x { x }
  def test_forward
    identity = IdentityMSE.new
    out = identity.forward(DFloat[0, 1])
    assert_equal out, DFloat[0, 1]
  end

  # loss = ->x, y { 0.5 * (f.(x) - y)**2 }
  def test_loss
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    identity = IdentityMSE.new
    model << identity
    model.compile(SGD.new)
    out = identity.forward(DFloat[[0, 1]])
    loss = identity.loss(DFloat[[2, 4]])
    assert_equal 6.5, loss.round(1)
  end

  def test_backward
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    identity = IdentityMSE.new
    model << identity
    model.compile(SGD.new)
    x = DFloat[[0, 1]]
    y = DFloat[[2, 4]]
    identity.forward(x)
    grad = identity.backward(y).round(4)
    func = ->x do
      identity.forward(x)
      identity.loss(y)
    end
    n_grad = Util.numerical_grad(x, func).round(4)
    assert_equal n_grad, grad.sum
  end
end


class TestIdentityMAE < MiniTest::Unit::TestCase
  # f = ->x { x }
  def test_forward
    identity = IdentityMAE.new
    out = identity.forward(DFloat[0, 1])
    assert_equal out, DFloat[0, 1]
  end

  # loss = ->x, y { (f.(x) - y).abs }
  def test_loss
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    identity = IdentityMAE.new
    model << identity
    model.compile(SGD.new)
    out = identity.forward(DFloat[[0, 1]])
    loss = identity.loss(DFloat[[2, 4]])
    assert_equal 5, loss.round(1)
  end

  def test_backward
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    identity = IdentityMAE.new
    model << identity
    model.compile(SGD.new)
    x = DFloat[[-1, 2]]
    y = DFloat[[2, 4]]
    identity.forward(x)
    grad = identity.backward(y).round(4)
    func = ->x do
      identity.forward(x)
      identity.loss(y)
    end
    n_grad = Util.numerical_grad(x, func).round(4)
    assert_equal n_grad, grad.sum
  end
end


class TestIdentityHuber < MiniTest::Unit::TestCase
  # f = ->x { x }
  def test_forward
    identity = IdentityMAE.new
    out = identity.forward(DFloat[0, 1])
    assert_equal out, DFloat[0, 1]
  end

  # loss = ->x, y { (f.(x) - y).abs > 1.0 ? (f.(x) - y).abs : 0.5 * (f.(x) - y)**2 }
  def test_loss
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    identity = IdentityHuber.new
    model << identity
    model.compile(SGD.new)
    out = identity.forward(DFloat[[0, 1]])
    loss = identity.loss(DFloat[[2, 4]])
    assert_equal 5, loss.round(3)
  end

  def test_loss2
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    identity = IdentityHuber.new
    model << identity
    model.compile(SGD.new)
    out = identity.forward(DFloat[[0, 1.0]])
    loss = identity.loss(DFloat[[0.5, 1.25]])
    assert_equal 0.15625, loss.round(5)
  end

  def test_backward
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    identity = IdentityMSE.new
    model << identity
    model.compile(SGD.new)
    x = DFloat[[-1, 2]]
    y = DFloat[[-3, 4]]
    identity.forward(x)
    grad = identity.backward(y).round(4)
    func = ->x do
      identity.forward(x)
      identity.loss(y)
    end
    n_grad = Util.numerical_grad(x, func).round(4)
    assert_equal n_grad, grad.sum
  end

  def test_backward2
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    identity = IdentityMSE.new
    model << identity
    model.compile(SGD.new)
    x = DFloat[[-1, 2]]
    y = DFloat[[-0.5, 1.7]]
    identity.forward(x)
    grad = identity.backward(y).round(4)
    func = ->x do
      identity.forward(x)
      identity.loss(y)
    end
    n_grad = Util.numerical_grad(x, func).round(4)
    assert_equal n_grad, grad.sum
  end
end


class TestSoftmaxWithLoss < MiniTest::Unit::TestCase
  # f = ->x { NMath.exp(x) / NMath.exp(x).sum }
  def test_forward
    softmax = SoftmaxWithLoss.new
    out = softmax.forward(DFloat[[0, 1, 2]]).round(4)
    assert_equal out, DFloat[[0.09, 0.2447, 0.6652]]
  end

  # loss = ->x, y { -(y * NMath.log(f.(x))).sum }
  def test_loss
    model = Model.new
    model << InputLayer.new(3)
    model << Dense.new(3)
    softmax = SoftmaxWithLoss.new
    model << softmax
    model.compile(SGD.new)
    out = softmax.forward(DFloat[[0, 1, 2]]).round(4)
    loss = softmax.loss(DFloat[[0, 0, 1]]).round(4)
    assert_equal 0.4076, loss
  end

  def test_backward
    model = Model.new
    model << InputLayer.new(3)
    model << Dense.new(3)
    softmax = SoftmaxWithLoss.new
    model << softmax
    model.compile(SGD.new)
    x = DFloat[[0, 1, 2]]
    y = DFloat[[0, 0, 1]]
    softmax.forward(x)
    grad = softmax.backward(y)
    func = ->x do
      softmax.forward(x)
      softmax.loss(y)
    end
    n_grad = Util.numerical_grad(x, func).round(4)
    assert_equal n_grad, grad.sum.round(4)
  end
end


class TestSigmoidWithLoss < MiniTest::Unit::TestCase
  def test_forward
    sigmoid = SigmoidWithLoss.new
    out = sigmoid.forward(DFloat[0, 1])
    assert_equal DFloat[0.5, 0.7311], out.round(4)
  end

  #loss = ->x, y { -(y * NMath.log(f.(x)) + (1 - y) * NMath.log(1 - f.(x))).sum }
  def test_loss
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    sigmoid = SigmoidWithLoss.new
    model << sigmoid
    model.compile(SGD.new)
    out = sigmoid.forward(DFloat[[0, 1]]).round(4)
    loss = sigmoid.loss(DFloat[[1, 0]]).round(4)
    assert_equal 2.0064, loss
  end

  def test_backward
    model = Model.new
    model << InputLayer.new(2)
    model << Dense.new(2)
    sigmoid = SigmoidWithLoss.new
    model << sigmoid
    model.compile(SGD.new)
    x = DFloat[[0, 1]]
    y = DFloat[[1, 0]]
    sigmoid.forward(x)
    grad = sigmoid.backward(y)
    func = ->x do
      sigmoid.forward(x)
      sigmoid.loss(y)
    end
    n_grad = Util.numerical_grad(x, func).round(4)
    assert_equal n_grad, grad.sum.round(4)
  end
end
