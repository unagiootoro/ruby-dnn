require "test_helper"

class TestMathUtils < MiniTest::Unit::TestCase
  def test_align_ndim1
    shape1 = [1, 2, 3, 4]
    shape2 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Functions::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 2, 3, 4], shape1
    assert_equal [1, 2, 3, 4], shape2
  end

  def test_align_ndim2
    shape1 = [3, 4]
    shape2 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Functions::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 1, 3, 4], shape1
    assert_equal [1, 2, 3, 4], shape2
  end

  def test_align_ndim3
    shape1 = [1, 2]
    shape2 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Functions::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 2, 1, 1], shape1
    assert_equal [1, 2, 3, 4], shape2
  end

  def test_align_ndim4
    shape2 = [3, 4]
    shape1 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Functions::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 1, 3, 4], shape2
    assert_equal [1, 2, 3, 4], shape1
  end

  def test_align_ndim5
    shape2 = [1, 2]
    shape1 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Functions::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 2, 1, 1], shape2
    assert_equal [1, 2, 3, 4], shape1
  end

  def test_broadcast_to
    y = Xumo::SFloat.new(1, 2, 3, 4).seq
    x = y.sum(axis: 2)
    x = DNN::Functions::MathUtils.broadcast_to(x, y.shape)
    assert_equal y.shape, x.shape
  end

  def test_sum_to
    x = Xumo::SFloat.new(1, 2, 3, 4).seq
    y = x.sum(axis: 2, keepdims: true)
    x = DNN::Functions::MathUtils.sum_to(x, y.shape)
    assert_equal y.shape, x.shape
  end
end

class TestNeg < MiniTest::Unit::TestCase
  def test_forward
    neg = DNN::Functions::Neg.new
    y = neg.forward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[-1, -2], y.round(4)
  end

  def test_backward
    neg = DNN::Functions::Neg.new
    neg.forward(Xumo::SFloat[1, 2])
    dx = neg.backward(Xumo::SFloat[3, 4])
    assert_equal Xumo::SFloat[-3, -4], dx
  end
end

class TestAdd < MiniTest::Unit::TestCase
  def test_forward
    add = DNN::Functions::Add.new
    y = add.forward(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    assert_equal Xumo::SFloat[4, 6], y.round(4)
  end

  def test_backward
    add = DNN::Functions::Add.new
    add.forward(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    dx1, dx2 = add.backward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[1, 2], dx1
    assert_equal Xumo::SFloat[1, 2], dx2
  end
end

class TestSub < MiniTest::Unit::TestCase
  def test_forward
    sub = DNN::Functions::Sub.new
    y = sub.forward(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    assert_equal Xumo::SFloat[-2, -2], y.round(4)
  end

  def test_backward
    sub = DNN::Functions::Sub.new
    sub.forward(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    dx1, dx2 = sub.backward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[1, 2], dx1
    assert_equal Xumo::SFloat[-1, -2], dx2
  end
end

class TestMul < MiniTest::Unit::TestCase
  def test_forward
    mul = DNN::Functions::Mul.new
    y = mul.forward(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    assert_equal Xumo::SFloat[3, 8], y.round(4)
  end

  def test_backward
    mul = DNN::Functions::Mul.new
    mul.forward(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    dx1, dx2 = mul.backward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[3, 8], dx1
    assert_equal Xumo::SFloat[1, 4], dx2
  end
end

class TestDiv < MiniTest::Unit::TestCase
  def test_forward
    div = DNN::Functions::Div.new
    y = div.forward(Xumo::SFloat[2, 4], Xumo::SFloat[4, 2])
    assert_equal Xumo::SFloat[0.5, 2], y.round(4)
  end

  def test_backward
    div = DNN::Functions::Div.new
    div.forward(Xumo::SFloat[2, 4], Xumo::SFloat[4, 2])
    dx1, dx2 = div.backward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[0.25, 1], dx1
    assert_equal Xumo::SFloat[-0.125, -2], dx2
  end
end

class TestExp < MiniTest::Unit::TestCase
  def test_forward
    exp = DNN::Functions::Exp.new
    y = exp.forward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[2.7183, 7.3891], y.round(4)
  end

  def test_backward
    exp = DNN::Functions::Exp.new
    exp.forward(Xumo::SFloat[1, 2])
    dx = exp.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[2.7183, 7.3891], dx.round(4)
  end
end

class TestLog < MiniTest::Unit::TestCase
  def test_forward
    log = DNN::Functions::Log.new
    y = log.forward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[0, 0.6931], y.round(4)
  end

  def test_backward
    log = DNN::Functions::Log.new
    log.forward(Xumo::SFloat[1, 2])
    dx = log.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[1, 0.5], dx.round(4)
  end
end

class TestPow < MiniTest::Unit::TestCase
  def test_forward
    pow = DNN::Functions::Pow.new(2)
    y = pow.forward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[1, 4], y.round(4)
  end

  def test_backward
    pow = DNN::Functions::Pow.new(2)
    pow.forward(Xumo::SFloat[1, 2])
    dx = pow.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[2, 4], dx.round(4)
  end
end

class TestSqrt < MiniTest::Unit::TestCase
  def test_forward
    sqrt = DNN::Functions::Sqrt.new
    y = sqrt.forward(Xumo::SFloat[4, 6])
    assert_equal Xumo::SFloat[2, 2.4495], y.round(4)
  end

  def test_backward
    sqrt = DNN::Functions::Sqrt.new
    sqrt.forward(Xumo::SFloat[4, 6])
    dx = sqrt.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[1, 1.2247], dx.round(4)
  end
end

class TestSum < MiniTest::Unit::TestCase
  def test_forward
    sum = DNN::Functions::Sum.new(axis: 0)
    y = sum.forward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[3], y.round(4)
  end

  def test_forward2
    sum = DNN::Functions::Sum.new(axis: nil)
    y = sum.forward(Xumo::SFloat[1, 2])
    assert_equal 3, y.round(4)
  end

  def test_backward
    sum = DNN::Functions::Sum.new(axis: 0)
    sum.forward(Xumo::SFloat[1, 2])
    dx = sum.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[1, 1], dx.round(4)
  end

  def test_backward2
    sum = DNN::Functions::Sum.new(axis: 0)
    sum.forward(Xumo::SFloat[1, 2])
    dx = sum.backward(Xumo::SFloat[1, 1])
    assert_equal Xumo::SFloat[1, 1], dx.round(4)
  end

  def test_backward3
    sum = DNN::Functions::Sum.new(axis: nil)
    sum.forward(Xumo::SFloat[1, 2])
    dx = sum.backward(Xumo::SFloat[1, 1])
    assert_equal Xumo::SFloat[1, 1], dx.round(4)
  end
end

class TestMean < MiniTest::Unit::TestCase
  def test_forward
    mean = DNN::Functions::Mean.new(axis: 0)
    y = mean.forward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[1.5], y.round(4)
  end

  def test_forward2
    mean = DNN::Functions::Mean.new(axis: nil, keepdims: false)
    y = mean.forward(Xumo::SFloat[1, 2])
    assert_equal 1.5, y.round(4)
  end

  def test_backward
    mean = DNN::Functions::Mean.new(axis: 0)
    mean.forward(Xumo::SFloat[1, 2])
    dx = mean.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[0.5, 0.5], dx.round(4)
  end

  def test_backward2
    mean = DNN::Functions::Mean.new(axis: 0)
    mean.forward(Xumo::SFloat[1, 2])
    dx = mean.backward(Xumo::SFloat[1, 1])
    assert_equal Xumo::SFloat[0.5, 0.5], dx.round(4)
  end

  def test_backward3
    mean = DNN::Functions::Mean.new(axis: nil, keepdims: false)
    mean.forward(Xumo::SFloat[1, 2])
    dx = mean.backward(Xumo::SFloat[1, 1])
    assert_equal Xumo::SFloat[0.5, 0.5], dx.round(4)
  end
end

class TestMax < MiniTest::Unit::TestCase
  def test_forward
    max = DNN::Functions::Max.new(axis: 0)
    y = max.forward(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[2], y.round(4)
  end

  def test_forward2
    max = DNN::Functions::Max.new(axis: nil, keepdims: false)
    y = max.forward(Xumo::SFloat[1, 2])
    assert_equal 2, y.round(4)
  end

  def test_backward
    max = DNN::Functions::Max.new(axis: 0)
    max.forward(Xumo::SFloat[1, 2])
    dx = max.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[0, 1], dx.round(4)
  end

  def test_backward2
    max = DNN::Functions::Max.new(axis: nil, keepdims: false)
    max.forward(Xumo::SFloat[1, 2])
    dx = max.backward(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[0, 1], dx.round(4)
  end
end

class TestBroadcastTo < MiniTest::Unit::TestCase
  def test_forward
    broadcast_to = DNN::Functions::BroadcastTo.new([3, 3])
    y = broadcast_to.forward(Xumo::SFloat[[1, 2, 3]])
    assert_equal Xumo::SFloat[[1, 2, 3], [1, 2, 3], [1, 2, 3]], y.round(4)
  end

  def test_backward
    broadcast_to = DNN::Functions::BroadcastTo.new([3, 3])
    broadcast_to.forward(Xumo::SFloat[[1, 2, 3]])
    dx = broadcast_to.backward(Xumo::SFloat[[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert_equal Xumo::SFloat[[6, 6, 6]], dx.round(4)
  end
end
