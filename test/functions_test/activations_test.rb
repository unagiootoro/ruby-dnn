require "test_helper"

class TestSigmoid < MiniTest::Unit::TestCase
  def test_forward
    sigmoid = DNN::Functions::Sigmoid.new
    y = sigmoid.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0.5, 0.7311], y.round(4)
  end

  def test_backward
    sigmoid = DNN::Functions::Sigmoid.new
    x = Xumo::SFloat[0, 1]
    sigmoid.forward(x)
    grad = sigmoid.backward(1)
    assert_equal Xumo::SFloat[0.25, 0.1966], grad.round(4)
  end

  def test_backward2
    sigmoid = DNN::Functions::Sigmoid.new
    x = Xumo::DFloat[0, 1]
    sigmoid.forward(x)
    grad = sigmoid.backward(1)
    assert_equal DNN::Utils.numerical_grad(x, sigmoid.method(:forward)).round(4), grad.round(4)
  end
end

class TestTanh < MiniTest::Unit::TestCase
  def test_forward
    tanh = DNN::Functions::Tanh.new
    y = tanh.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0, 0.7616], y.round(4)
  end

  def test_backward
    tanh = DNN::Functions::Tanh.new
    x = Xumo::SFloat[0, 1]
    tanh.forward(x)
    grad = tanh.backward(1)
    assert_equal Xumo::SFloat[1, 0.42], grad.round(4)
  end

  def test_backward2
    tanh = DNN::Functions::Tanh.new
    x = Xumo::DFloat[0, 1]
    tanh.forward(x)
    grad = tanh.backward(1)
    assert_equal DNN::Utils.numerical_grad(x, tanh.method(:forward)).round(4), grad.round(4)
  end
end

class TestSoftsign < MiniTest::Unit::TestCase
  def test_forward
    softsign = DNN::Functions::Softsign.new
    y = softsign.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0, 0.5], y.round(4)
  end

  def test_backward
    softsign = DNN::Functions::Softsign.new
    x = Xumo::SFloat[0, 1]
    softsign.forward(x)
    grad = softsign.backward(1)
    assert_equal Xumo::SFloat[1, 0.25], grad.round(4)
  end

  def test_backward2
    softsign = DNN::Functions::Softsign.new
    x = Xumo::DFloat[0, 1]
    softsign.forward(x)
    grad = softsign.backward(1)
    assert_equal DNN::Utils.numerical_grad(x, softsign.method(:forward)).round(4), grad.round(4)
  end
end

class TestSoftplus < MiniTest::Unit::TestCase
  def test_forward
    softplus = DNN::Functions::Softplus.new
    y = softplus.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0.6931, 1.3133], y.round(4)
  end

  def test_backward
    softplus = DNN::Functions::Softplus.new
    x = Xumo::SFloat[0, 1]
    softplus.forward(x)
    grad = softplus.backward(1)
    assert_equal Xumo::SFloat[0.5, 0.7311], grad.round(4)
  end

  def test_backward2
    softplus = DNN::Functions::Softplus.new
    x = Xumo::DFloat[0, 1]
    softplus.forward(x)
    grad = softplus.backward(1)
    assert_equal DNN::Utils.numerical_grad(x, softplus.method(:forward)).round(4), grad.round(4)
  end
end

class TestSwish < MiniTest::Unit::TestCase
  def test_forward
    swish = DNN::Functions::Swish.new
    y = swish.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0, 0.7311], y.round(4)
  end

  def test_backward
    swish = DNN::Functions::Swish.new
    x = Xumo::SFloat[0, 1]
    swish.forward(x)
    grad = swish.backward(1)
    assert_equal Xumo::SFloat[0.5, 0.9277], grad.round(4)
  end

  def test_backward2
    swish = DNN::Functions::Swish.new
    x = Xumo::DFloat[0, 1]
    swish.forward(x)
    grad = swish.backward(1)
    assert_equal DNN::Utils.numerical_grad(x, swish.method(:forward)).round(4), grad.round(4)
  end
end

class TestReLU < MiniTest::Unit::TestCase
  def test_forward
    relu = DNN::Functions::ReLU.new
    y = relu.forward(Xumo::SFloat[-2, 0, 2])
    assert_equal Xumo::SFloat[0, 0, 2], y
  end

  def test_backward
    relu = DNN::Functions::ReLU.new
    relu.forward(Xumo::SFloat[-2, 0, 2])
    grad = relu.backward(1)
    assert_equal Xumo::SFloat[0, 0, 1], grad.round(4)
  end
end

class TestLeakyReLU < MiniTest::Unit::TestCase
  def test_initialize
    lrelu = DNN::Functions::LeakyReLU.new
    assert_equal 0.3, lrelu.alpha
  end

  def test_forward
    lrelu = DNN::Functions::LeakyReLU.new
    y = lrelu.forward(Xumo::SFloat[-2, 0, 2])
    assert_equal Xumo::SFloat[-0.6, 0, 2], y.round(4)
  end

  def test_backward
    lrelu = DNN::Functions::LeakyReLU.new
    lrelu.forward(Xumo::SFloat[-2, 0, 2])
    grad = lrelu.backward(1)
    assert_equal Xumo::SFloat[0.3, 0.3, 1], grad.round(4)
  end

  def test_backward2
    lrelu = DNN::Functions::LeakyReLU.new
    x = Xumo::DFloat[-2, 1, 2]
    lrelu.forward(x)
    grad = lrelu.backward(1)
    assert_equal DNN::Utils.numerical_grad(x, lrelu.method(:forward)).round(4), Xumo::DFloat.cast(grad).round(4)
  end
end

class TestELU < MiniTest::Unit::TestCase
  def test_initialize
    elu = DNN::Functions::ELU.new
    assert_equal 1.0, elu.alpha
  end

  def test_forward
    elu = DNN::Functions::ELU.new
    y = elu.forward(Xumo::SFloat[-2, 0, 2])
    assert_equal Xumo::SFloat[-0.8647, 0, 2], y.round(4)
  end

  def test_backward
    elu = DNN::Functions::ELU.new
    elu.forward(Xumo::SFloat[-2, 0, 2])
    grad = elu.backward(1)
    assert_equal Xumo::SFloat[0.1353, 1, 1], grad.round(4)
  end

  def test_backward2
    elu = DNN::Functions::ELU.new
    x = Xumo::DFloat[-2, 1, 2]
    elu.forward(x)
    grad = elu.backward(1)
    assert_equal DNN::Utils.numerical_grad(x, elu.method(:forward)).round(4), grad.round(4)
  end
end

class TestMish < MiniTest::Unit::TestCase
  def test_forward
    mish = DNN::Functions::Mish.new
    y = mish.forward(Xumo::SFloat[0, 1])
    assert_equal Xumo::SFloat[0, 0.8651], y.round(4)
  end

  def test_backward
    mish = DNN::Functions::Mish.new
    x = Xumo::SFloat[0, 1]
    mish.forward(x)
    grad = mish.backward(1)
    assert_equal Xumo::SFloat[0.6, 1.049], grad.round(4)
  end

  def test_backward2
    mish = DNN::Functions::Mish.new
    x = Xumo::DFloat[0, 1]
    mish.forward(x)
    grad = mish.backward(1)
    assert_equal DNN::Utils.numerical_grad(x, mish.method(:forward)).round(4), grad.round(4)
  end
end
