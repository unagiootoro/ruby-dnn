require "test_helper"

class TestSigmoid < MiniTest::Unit::TestCase
  def test_forward
    sigmoid = DNN::Activations::Sigmoid.new
    y = sigmoid.forward(Numo::SFloat[0, 1])
    assert_equal Numo::SFloat[0.5, 0.7311], y.round(4)
  end

  def test_backward
    sigmoid = DNN::Activations::Sigmoid.new
    x = Numo::SFloat[0, 1]
    sigmoid.forward(x)
    grad = sigmoid.backward(1)
    assert_equal Numo::SFloat[0.25, 0.1966], grad.round(4)
  end
end


class TestTanh < MiniTest::Unit::TestCase
  def test_forward
    tanh = DNN::Activations::Tanh.new
    y = tanh.forward(Numo::SFloat[0, 1])
    assert_equal Numo::SFloat[0, 0.7616], y.round(4)
  end

  def test_backward
    tanh = DNN::Activations::Tanh.new
    x = Numo::SFloat[0, 1]
    tanh.forward(x)
    grad = tanh.backward(1)
    assert_equal Numo::SFloat[1, 0.42], grad.round(4)
  end
end


class TestSoftsign < MiniTest::Unit::TestCase
  def test_forward
    softsign = DNN::Activations::Softsign.new
    y = softsign.forward(Numo::SFloat[0, 1])
    assert_equal Numo::SFloat[0, 0.5], y.round(4)
  end

  def test_backward
    softsign = DNN::Activations::Softsign.new
    x = Numo::SFloat[0, 1]
    softsign.forward(x)
    grad = softsign.backward(1)
    assert_equal Numo::SFloat[1, 0.25], grad.round(4)
  end
end


class TestSoftplus < MiniTest::Unit::TestCase
  def test_forward
    softplus = DNN::Activations::Softplus.new
    y = softplus.forward(Numo::SFloat[0, 1])
    assert_equal Numo::SFloat[0.6931, 1.3133], y.round(4)
  end

  def test_backward
    softplus = DNN::Activations::Softplus.new
    x = Numo::SFloat[0, 1]
    softplus.forward(x)
    grad = softplus.backward(1)
    assert_equal Numo::SFloat[0.5, 0.7311], grad.round(4)
  end
end


class TestSwish < MiniTest::Unit::TestCase
  def test_forward
    swish = DNN::Activations::Swish.new
    y = swish.forward(Numo::SFloat[0, 1])
    assert_equal Numo::SFloat[0, 0.7311], y.round(4)
  end

  def test_backward
    swish = DNN::Activations::Swish.new
    x = Numo::SFloat[0, 1]
    swish.forward(x)
    grad = swish.backward(1)
    assert_equal Numo::SFloat[0.5, 0.9277], grad.round(4)
  end
end


class TestReLU < MiniTest::Unit::TestCase
  def test_forward
    relu = DNN::Activations::ReLU.new
    y = relu.forward(Numo::SFloat[-2, 0, 2])
    assert_equal Numo::SFloat[0, 0, 2], y
  end

  def test_backward
    relu = DNN::Activations::ReLU.new
    relu.forward(Numo::SFloat[-2, 0, 2])
    grad = relu.backward(1)
    assert_equal Numo::SFloat[0, 0, 1], grad.round(4)
  end
end


class TestLeakyReLU < MiniTest::Unit::TestCase
  def test_from_hash
    hash = { class: "DNN::Activations::LeakyReLU", alpha: 0.2 }
    lrelu = DNN::Activations::LeakyReLU.from_hash(hash)
    assert_equal 0.2, lrelu.alpha
  end

  def test_initialize
    lrelu = DNN::Activations::LeakyReLU.new
    assert_equal 0.3, lrelu.alpha
  end

  def test_forward
    lrelu = DNN::Activations::LeakyReLU.new
    y = lrelu.forward(Numo::SFloat[-2, 0, 2])
    assert_equal Numo::SFloat[-0.6, 0, 2], y.round(4)
  end

  def test_backward
    lrelu = DNN::Activations::LeakyReLU.new
    lrelu.forward(Numo::SFloat[-2, 0, 2])
    grad = lrelu.backward(1)
    assert_equal Numo::SFloat[0.3, 0.3, 1], grad.round(4)
  end

  def test_to_hash
    lrelu = DNN::Activations::LeakyReLU.new
    expected_hash = { class: "DNN::Activations::LeakyReLU", alpha: 0.3 }
    assert_equal expected_hash, lrelu.to_hash
  end
end


class TestELU < MiniTest::Unit::TestCase
  def test_from_hash
    hash = { class: "DNN::Activations::ELU", alpha: 0.2 }
    elu = DNN::Activations::ELU.from_hash(hash)
    assert_equal 0.2, elu.alpha
  end

  def test_initialize
    elu = DNN::Activations::ELU.new
    assert_equal 1.0, elu.alpha
  end

  def test_forward
    elu = DNN::Activations::ELU.new
    y = elu.forward(Numo::SFloat[-2, 0, 2])
    assert_equal Numo::SFloat[-0.8647, 0, 2], y.round(4)
  end

  def test_backward
    elu = DNN::Activations::ELU.new
    elu.forward(Numo::SFloat[-2, 0, 2])
    grad = elu.backward(1)
    assert_equal Numo::SFloat[0.1353, 1, 1], grad.round(4)
  end

  def test_to_hash
    elu = DNN::Activations::ELU.new
    expected_hash = { class: "DNN::Activations::ELU", alpha: 1.0 }
    assert_equal expected_hash, elu.to_hash
  end
end
