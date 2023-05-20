require "test_helper"

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

  def test_backward2
    add = DNN::Functions::Add.new
    add.forward(Xumo::SFloat[[1, 2], [3, 4]], Xumo::SFloat[5])
    dx1, dx2 = add.backward(Xumo::SFloat[[1, 1], [1, 1]])
    assert_equal Xumo::SFloat[[1, 1], [1, 1]], dx1
    assert_equal Xumo::SFloat[4], dx2
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

class TestDot < MiniTest::Unit::TestCase
  def test_forward
    dot = DNN::Functions::Dot.new
    y = dot.forward(Xumo::SFloat[[1, 2, 3], [4, 5, 6]], Xumo::SFloat[[10, 20], [10, 20], [10, 20]])
    assert_equal Xumo::SFloat[[60, 120], [150, 300]], y.round(4)
  end

  def test_backward
    dot = DNN::Functions::Dot.new
    dot.forward(Xumo::SFloat[[1, 2, 3], [4, 5, 6]], Xumo::SFloat[[10, 20], [10, 20], [10, 20]])
    dx1, dx2 = dot.backward(Xumo::SFloat[[1, 1], [1, 1]])
    assert_equal Xumo::SFloat[[30, 30, 30], [30, 30, 30]], dx1
    assert_equal Xumo::SFloat[[5, 5], [7, 7], [9, 9]], dx2
  end
end

class TestFlatten < MiniTest::Unit::TestCase
  def test_forward
    flatten = DNN::Functions::Flatten.new
    y = flatten.forward(Xumo::SFloat.zeros(10, 32, 32, 3))
    assert_equal [30720], y.shape
  end

  def test_backward
    flatten = DNN::Functions::Flatten.new
    flatten.forward(Xumo::SFloat.zeros(10, 32, 32, 3))
    dx = flatten.backward(Xumo::SFloat.ones(30720))
    assert_equal [10, 32, 32, 3], dx.shape
  end
end

class TestReshape < MiniTest::Unit::TestCase
  def test_forward
    reshape = DNN::Functions::Reshape.new([10, 32, 32, 3])
    x = Xumo::SFloat.zeros(30720)
    y = reshape.forward(x)
    assert_equal [10, 32, 32, 3], y.shape
  end

  def test_backward
    reshape = DNN::Functions::Reshape.new([10, 32, 32, 3])
    x = Xumo::SFloat.zeros(30720)
    reshape.forward(x)
    dy = Xumo::SFloat.ones(10, 32, 32, 3)
    dx = reshape.backward(dy)
    assert_equal [30720], dx.shape
  end
end

class TestTranspose < MiniTest::Unit::TestCase
  def test_forward
    x = Xumo::SFloat.zeros(10, 20, 30, 40)
    transpose = DNN::Functions::Transpose.new(2, 3, 1, 0)
    y = transpose.forward(x)
    assert_equal [30, 40, 20, 10], y.shape
  end

  def test_forward2
    x = Xumo::SFloat.zeros(10, 20, 30, 40)
    transpose = DNN::Functions::Transpose.new
    y = transpose.forward(x)
    assert_equal [40, 30, 20, 10], y.shape
  end

  def test_backward
    x = Xumo::SFloat.zeros(10, 20, 30, 40)
    transpose = DNN::Functions::Transpose.new(2, 3, 1, 0)
    transpose.forward(x)
    dx = transpose.backward(Xumo::SFloat.zeros(30, 40, 20, 10))
    assert_equal [10, 20, 30, 40], dx.shape
  end

  def test_backward2
    x = Xumo::SFloat.zeros(10, 20, 30, 40)
    transpose = DNN::Functions::Transpose.new
    transpose.forward(x)
    dx = transpose.backward(Xumo::SFloat.zeros(40, 30, 20, 10))
    assert_equal [10, 20, 30, 40], dx.shape
  end
end

class TestConcatenate < MiniTest::Unit::TestCase
  def test_forward
    con = DNN::Functions::Concatenate.new(axis: 1)
    y = con.forward(Xumo::SFloat[[1, 2, 3]], Xumo::SFloat[[4, 5]])
    assert_equal Xumo::SFloat[[1, 2, 3, 4, 5]], y.round(4)
  end

  def test_backward
    con = DNN::Functions::Concatenate.new(axis: 1)
    con.forward(Xumo::SFloat[[1, 2, 3]], Xumo::SFloat[[4, 5]])
    dx1, dx2 = con.backward(Xumo::SFloat[[6, 7, 8, 9, 10]])
    assert_equal Xumo::SFloat[[6, 7, 8]], dx1
    assert_equal Xumo::SFloat[[9, 10]], dx2
  end
end

class TestSplit < MiniTest::Unit::TestCase
  def test_forward
    split = DNN::Functions::Split.new(2, axis: 1)
    y1, y2 = split.forward(Xumo::SFloat[[1, 2, 3, 4, 5]])
    assert_equal Xumo::SFloat[[1, 2, 3]], y1.round(4)
    assert_equal Xumo::SFloat[[4, 5]], y2.round(4)
  end

  def test_forward2
    split = DNN::Functions::Split.new([2, 5], axis: 1)
    y1, y2 = split.forward(Xumo::SFloat[[1, 2, 3, 4, 5]])
    assert_equal Xumo::SFloat[[1, 2]], y1.round(4)
    assert_equal Xumo::SFloat[[3, 4, 5]], y2.round(4)
  end

  def test_backward
    split = DNN::Functions::Split.new(2, axis: 1)
    split.forward(Xumo::SFloat[[1, 2, 3, 4, 5]])
    dx = split.backward(Xumo::SFloat[[6, 7, 8]], Xumo::SFloat[[9, 10]])
    assert_equal Xumo::SFloat[[6, 7, 8, 9, 10]], dx
  end

  def test_backward2
    split = DNN::Functions::Split.new([2, 5], axis: 1)
    split.forward(Xumo::SFloat[[1, 2, 3, 4, 5]])
    dx = split.backward(Xumo::SFloat[[6, 7]], Xumo::SFloat[[8, 9, 10]])
    assert_equal Xumo::SFloat[[6, 7, 8, 9, 10]], dx
  end
end
