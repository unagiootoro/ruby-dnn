require "test_helper"

class TestSigmoid < MiniTest::Unit::TestCase
  def test_forward
    sigmoid = DNN::Layers::Sigmoid.new
    y = sigmoid.(DNN::Tensor.new(Xumo::SFloat[0, 1]))
    assert_equal Xumo::SFloat[0.5, 0.7311], y.data.round(4)
  end
end

class TestTanh < MiniTest::Unit::TestCase
  def test_forward
    tanh = DNN::Layers::Tanh.new
    y = tanh.(DNN::Tensor.new(Xumo::SFloat[0, 1]))
    assert_equal Xumo::SFloat[0, 0.7616], y.data.round(4)
  end
end

class TestSoftsign < MiniTest::Unit::TestCase
  def test_forward
    softsign = DNN::Layers::Softsign.new
    y = softsign.(DNN::Tensor.new(Xumo::SFloat[0, 1]))
    assert_equal Xumo::SFloat[0, 0.5], y.data.round(4)
  end
end

class TestSoftplus < MiniTest::Unit::TestCase
  def test_forward
    softplus = DNN::Layers::Softplus.new
    y = softplus.(DNN::Tensor.new(Xumo::SFloat[0, 1]))
    assert_equal Xumo::SFloat[0.6931, 1.3133], y.data.round(4)
  end
end

class TestSwish < MiniTest::Unit::TestCase
  def test_forward
    swish = DNN::Layers::Swish.new
    y = swish.(DNN::Tensor.new(Xumo::SFloat[0, 1]))
    assert_equal Xumo::SFloat[0, 0.7311], y.data.round(4)
  end
end

class TestReLU < MiniTest::Unit::TestCase
  def test_forward
    relu = DNN::Layers::ReLU.new
    y = relu.(DNN::Tensor.new(Xumo::SFloat[-2, 0, 2]))
    assert_equal Xumo::SFloat[0, 0, 2], y.data
  end
end

class TestLeakyReLU < MiniTest::Unit::TestCase
  def test_initialize
    lrelu = DNN::Layers::LeakyReLU.new
    assert_equal 0.3, lrelu.alpha
  end

  def test_forward
    lrelu = DNN::Layers::LeakyReLU.new
    y = lrelu.(DNN::Tensor.new(Xumo::SFloat[-2, 0, 2]))
    assert_equal Xumo::SFloat[-0.6, 0, 2], y.data.round(4)
  end
end

class TestELU < MiniTest::Unit::TestCase
  def test_initialize
    elu = DNN::Layers::ELU.new
    assert_equal 1.0, elu.alpha
  end
end

class TestMish < MiniTest::Unit::TestCase
  def test_forward
    mish = DNN::Layers::Mish.new
    y = mish.(DNN::Tensor.new(Xumo::SFloat[0, 1]))
    assert_equal Xumo::SFloat[0, 0.8651], y.data.round(4)
  end
end
