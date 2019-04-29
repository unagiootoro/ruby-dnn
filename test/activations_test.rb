require "test_helper"

include DNN
include Activations
include Layers
include Optimizers

class TestSigmoid < MiniTest::Unit::TestCase
  def test_forward
    sigmoid = Sigmoid.new
    out = sigmoid.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0.5, 0.7311], out.round(4)
  end

  def test_backward
    sigmoid = Sigmoid.new
    x = Xumo::SFloat[0, 1]
    sigmoid.forward(x)
    grad = sigmoid.backward(1)
    assert_equal Xumo::SFloat[0.25, 0.1966], grad.round(4)
  end
end


class TestTanh < MiniTest::Unit::TestCase
  def test_forward
    tanh = Tanh.new
    out = tanh.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0, 0.7616], out.round(4)
  end

  def test_backward
    tanh = Tanh.new
    x = Xumo::SFloat[0, 1]
    tanh.forward(x)
    grad = tanh.backward(1)
    assert_equal Xumo::SFloat[1, 0.42], grad.round(4)
  end
end


class TestSoftsign < MiniTest::Unit::TestCase
  def test_forward
    softsign = Softsign.new
    out = softsign.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0, 0.5], out.round(4)
  end

  def test_backward
    softsign = Softsign.new
    x = Xumo::SFloat[0, 1]
    softsign.forward(x)
    grad = softsign.backward(1)
    assert_equal Xumo::SFloat[1, 0.25], grad.round(4)
  end
end


class TestSoftplus < MiniTest::Unit::TestCase
  def test_forward
    softplus = Softplus.new
    out = softplus.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0.6931, 1.3133], out.round(4)
  end

  def test_backward
    softplus = Softplus.new
    x = Xumo::SFloat[0, 1]
    softplus.forward(x)
    grad = softplus.backward(1)
    assert_equal Xumo::SFloat[0.5, 0.7311], grad.round(4)
  end
end


class TestSwish < MiniTest::Unit::TestCase
  def test_forward
    swish = Swish.new
    out = swish.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0, 0.7311], out.round(4)
  end

  def test_backward
    swish = Swish.new
    x = Xumo::SFloat[0, 1]
    swish.forward(x)
    grad = swish.backward(1)
    assert_equal Xumo::SFloat[0.5, 0.9277], grad.round(4)
  end
end


class TestReLU < MiniTest::Unit::TestCase
  def test_forward
    relu = ReLU.new
    out = relu.forward(Xumo::SFloat[-2, 0, 2])
    assert_equal Xumo::SFloat[0, 0, 2], out
  end

  def test_backward
    relu = ReLU.new
    relu.forward(Xumo::SFloat[-2, 0, 2])
    grad = relu.backward(1)
    assert_equal Xumo::SFloat[0, 0, 1], grad.round(4)
  end
end


class TestLeakyReLU < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {alpha: 0.2}
    lrelu = LeakyReLU.load_hash(hash)
    assert_equal 0.2, lrelu.alpha
  end

  def test_initialize
    lrelu = LeakyReLU.new
    assert_equal 0.3, lrelu.alpha
  end

  def test_forward
    lrelu = LeakyReLU.new
    out = lrelu.forward(Xumo::SFloat[-2, 0, 2])
    assert_equal Xumo::SFloat[-0.6, 0, 2], out.round(4)
  end

  def test_backward
    lrelu = LeakyReLU.new
    lrelu.forward(Xumo::SFloat[-2, 0, 2])
    grad = lrelu.backward(1)
    assert_equal Xumo::SFloat[0.3, 0.3, 1], grad.round(4)
  end

  def test_to_hash
    lrelu = LeakyReLU.new
    expected_hash = {class: "DNN::Activations::LeakyReLU", alpha: 0.3}
    assert_equal expected_hash, lrelu.to_hash
  end
end


class TestELU < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {alpha: 0.2}
    elu = ELU.load_hash(hash)
    assert_equal 0.2, elu.alpha
  end

  def test_initialize
    elu = ELU.new
    assert_equal 1.0, elu.alpha
  end

  def test_forward
    elu = ELU.new
    out = elu.forward(Xumo::SFloat[-2, 0, 2])
    assert_equal Xumo::SFloat[-0.8647, 0, 2], out.round(4)
  end

  def test_backward
    elu = ELU.new
    elu.forward(Xumo::SFloat[-2, 0, 2])
    grad = elu.backward(1)
    assert_equal Xumo::SFloat[0.1353, 1, 1], grad.round(4)
  end

  def test_to_hash
    elu = ELU.new
    expected_hash = {class: "DNN::Activations::ELU", alpha: 1.0}
    assert_equal expected_hash, elu.to_hash
  end
end
