require "test_helper"

class TestMathUtils < MiniTest::Unit::TestCase
  def test_align_ndim1
    shape1 = [1, 2, 3, 4]
    shape2 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Layers::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 2, 3, 4], shape1
    assert_equal [1, 2, 3, 4], shape2
  end

  def test_align_ndim2
    shape1 = [3, 4]
    shape2 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Layers::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 1, 3, 4], shape1
    assert_equal [1, 2, 3, 4], shape2
  end

  def test_align_ndim3
    shape1 = [1, 2]
    shape2 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Layers::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 2, 1, 1], shape1
    assert_equal [1, 2, 3, 4], shape2
  end

  def test_align_ndim4
    shape2 = [3, 4]
    shape1 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Layers::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 1, 3, 4], shape2
    assert_equal [1, 2, 3, 4], shape1
  end

  def test_align_ndim5
    shape2 = [1, 2]
    shape1 = [1, 2, 3, 4]
    shape1, shape2 = DNN::Layers::MathUtils.align_ndim(shape1, shape2)
    assert_equal [1, 2, 1, 1], shape2
    assert_equal [1, 2, 3, 4], shape1
  end

  def test_broadcast_to
    y = Xumo::SFloat.new(1, 2, 3, 4).seq
    x = y.sum(axis: 2)
    x = DNN::Layers::MathUtils.broadcast_to(x, y.shape)
    assert_equal y.shape, x.shape
  end

  def test_sum_to
    x = Xumo::SFloat.new(1, 2, 3, 4).seq
    y = x.sum(axis: 2, keepdims: true)
    x = DNN::Layers::MathUtils.sum_to(x, y.shape)
    assert_equal y.shape, x.shape
  end
end

class TestNeg < MiniTest::Unit::TestCase
  def test_forward_node
    neg = DNN::Layers::Neg.new
    y = neg.forward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[-1, -2], y.round(4)
  end

  def test_backward_node
    neg = DNN::Layers::Neg.new
    neg.forward_node(Xumo::SFloat[1, 2])
    dx = neg.backward_node(Xumo::SFloat[3, 4])
    assert_equal Xumo::SFloat[-3, -4], dx
  end
end

class TestAdd < MiniTest::Unit::TestCase
  def test_forward_node
    add = DNN::Layers::Add.new
    y = add.forward_node(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    assert_equal Xumo::SFloat[4, 6], y.round(4)
  end

  def test_backward_node
    add = DNN::Layers::Add.new
    add.forward_node(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    dx1, dx2 = add.backward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[1, 2], dx1
    assert_equal Xumo::SFloat[1, 2], dx2
  end
end

class TestSub < MiniTest::Unit::TestCase
  def test_forward_node
    sub = DNN::Layers::Sub.new
    y = sub.forward_node(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    assert_equal Xumo::SFloat[-2, -2], y.round(4)
  end

  def test_backward_node
    sub = DNN::Layers::Sub.new
    sub.forward_node(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    dx1, dx2 = sub.backward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[1, 2], dx1
    assert_equal Xumo::SFloat[-1, -2], dx2
  end
end

class TestMul < MiniTest::Unit::TestCase
  def test_forward_node
    mul = DNN::Layers::Mul.new
    y = mul.forward_node(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    assert_equal Xumo::SFloat[3, 8], y.round(4)
  end

  def test_backward_node
    mul = DNN::Layers::Mul.new
    mul.forward_node(Xumo::SFloat[1, 2], Xumo::SFloat[3, 4])
    dx1, dx2 = mul.backward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[3, 8], dx1
    assert_equal Xumo::SFloat[1, 4], dx2
  end
end

class TestDiv < MiniTest::Unit::TestCase
  def test_forward_node
    div = DNN::Layers::Div.new
    y = div.forward_node(Xumo::SFloat[2, 4], Xumo::SFloat[4, 2])
    assert_equal Xumo::SFloat[0.5, 2], y.round(4)
  end

  def test_backward_node
    div = DNN::Layers::Div.new
    div.forward_node(Xumo::SFloat[2, 4], Xumo::SFloat[4, 2])
    dx1, dx2 = div.backward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[0.25, 1], dx1
    assert_equal Xumo::SFloat[-0.125, -2], dx2
  end
end

class TestDot < MiniTest::Unit::TestCase
  def test_forward_node
    dot = DNN::Layers::Dot.new
    y = dot.forward_node(Xumo::SFloat[[1, 2, 3], [4, 5, 6]], Xumo::SFloat[[10, 20], [10, 20], [10, 20]])
    assert_equal Xumo::SFloat[[60, 120], [150, 300]], y.round(4)
  end

  def test_backward_node
    dot = DNN::Layers::Dot.new
    dot.forward_node(Xumo::SFloat[[1, 2, 3], [4, 5, 6]], Xumo::SFloat[[10, 20], [10, 20], [10, 20]])
    dx1, dx2 = dot.backward_node(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[30, 30, 30], dx1
    assert_equal Xumo::SFloat[5, 7, 9], dx2
  end
end

class TestExp < MiniTest::Unit::TestCase
  def test_forward_node
    exp = DNN::Layers::Exp.new
    y = exp.forward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[2.7183, 7.3891], y.round(4)
  end

  def test_backward_node
    exp = DNN::Layers::Exp.new
    exp.forward_node(Xumo::SFloat[1, 2])
    dx = exp.backward_node(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[2.7183, 7.3891], dx.round(4)
  end
end

class TestLog < MiniTest::Unit::TestCase
  def test_forward_node
    log = DNN::Layers::Log.new
    y = log.forward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[0, 0.6931], y.round(4)
  end

  def test_backward_node
    log = DNN::Layers::Log.new
    log.forward_node(Xumo::SFloat[1, 2])
    dx = log.backward_node(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[1, 0.5], dx.round(4)
  end
end

class TestPow < MiniTest::Unit::TestCase
  def test_forward_node
    pow = DNN::Layers::Pow.new(2)
    y = pow.forward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[1, 4], y.round(4)
  end

  def test_backward_node
    pow = DNN::Layers::Pow.new(2)
    pow.forward_node(Xumo::SFloat[1, 2])
    dx = pow.backward_node(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[2, 4], dx.round(4)
  end
end

class TestSqrt < MiniTest::Unit::TestCase
  def test_forward_node
    sqrt = DNN::Layers::Sqrt.new
    y = sqrt.forward_node(Xumo::SFloat[4, 6])
    assert_equal Xumo::SFloat[2, 2.4495], y.round(4)
  end

  def test_backward_node
    sqrt = DNN::Layers::Sqrt.new
    sqrt.forward_node(Xumo::SFloat[4, 6])
    dx = sqrt.backward_node(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[1, 1.2247], dx.round(4)
  end
end

class TestSum < MiniTest::Unit::TestCase
  def test_from_hash
    sum = DNN::Layers::Sum.new(axis: 1, keepdims: false)
    assert_equal 1, sum.axis
    assert_equal false, sum.keepdims
  end

  def test_forward_node
    sum = DNN::Layers::Sum.new(axis: 0)
    y = sum.forward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[3], y.round(4)
  end

  def test_forward_node2
    sum = DNN::Layers::Sum.new(axis: nil)
    y = sum.forward_node(Xumo::SFloat[1, 2])
    assert_equal 3, y.round(4)
  end

  def test_backward_node
    sum = DNN::Layers::Sum.new(axis: 0)
    sum.forward_node(Xumo::SFloat[1, 2])
    dx = sum.backward_node(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[1, 1], dx.round(4)
  end

  def test_backward_node2
    sum = DNN::Layers::Sum.new(axis: 0)
    sum.forward_node(Xumo::SFloat[1, 2])
    dx = sum.backward_node(Xumo::SFloat[1, 1])
    assert_equal Xumo::SFloat[1, 1], dx.round(4)
  end

  def test_backward_node3
    sum = DNN::Layers::Sum.new(axis: nil)
    sum.forward_node(Xumo::SFloat[1, 2])
    dx = sum.backward_node(Xumo::SFloat[1, 1])
    assert_equal Xumo::SFloat[1, 1], dx.round(4)
  end

  def test_to_hash
    expected_hash = { class: "DNN::Layers::Sum", axis: 0, keepdims: true }
    sum = DNN::Layers::Sum.new
    assert_equal expected_hash, sum.to_hash
  end
end

class TestMean < MiniTest::Unit::TestCase
  def test_from_hash
    sum = DNN::Layers::Sum.new(axis: 1, keepdims: false)
    assert_equal 1, sum.axis
    assert_equal false, sum.keepdims
  end

  def test_forward_node
    mean = DNN::Layers::Mean.new(axis: 0)
    y = mean.forward_node(Xumo::SFloat[1, 2])
    assert_equal Xumo::SFloat[1.5], y.round(4)
  end

  def test_forward_node2
    mean = DNN::Layers::Mean.new(axis: nil)
    y = mean.forward_node(Xumo::SFloat[1, 2])
    assert_equal 1.5, y.round(4)
  end

  def test_backward_node
    mean = DNN::Layers::Mean.new(axis: 0)
    mean.forward_node(Xumo::SFloat[1, 2])
    dx = mean.backward_node(Xumo::SFloat[1])
    assert_equal Xumo::SFloat[0.5, 0.5], dx.round(4)
  end

  def test_backward_node2
    mean = DNN::Layers::Mean.new(axis: 0)
    mean.forward_node(Xumo::SFloat[1, 2])
    dx = mean.backward_node(Xumo::SFloat[1, 1])
    assert_equal Xumo::SFloat[0.5, 0.5], dx.round(4)
  end

  def test_backward_node3
    mean = DNN::Layers::Mean.new(axis: nil)
    mean.forward_node(Xumo::SFloat[1, 2])
    dx = mean.backward_node(Xumo::SFloat[1, 1])
    assert_equal Xumo::SFloat[0.5, 0.5], dx.round(4)
  end

  def test_to_hash
    expected_hash = { class: "DNN::Layers::Mean", axis: 0, keepdims: true }
    mean = DNN::Layers::Mean.new
    assert_equal expected_hash, mean.to_hash
  end
end
