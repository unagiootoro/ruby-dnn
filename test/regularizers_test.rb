include DNN

class TestLasso < MiniTest::Unit::TestCase
  def test_forward
    param = Param.new(Numo::SFloat[-2, 2])
    lasso = Lasso.new(0.1, param)
    assert_equal 1.4, lasso.forward(1)
  end

  def test_backward
    x = Numo::SFloat.new(10).fill(-1)
    x[0..4] = 1
    param = Param.new(Numo::SFloat[-2, 2], Numo::SFloat[-1, 1])
    lasso = Lasso.new(0.1, param)
    lasso.backward
    assert_equal Numo::SFloat[-1.1, 1.1], lasso.instance_variable_get(:@param).grad
  end
end


class TestRidge < MiniTest::Unit::TestCase
  def test_forward
    param = Param.new(Numo::SFloat[-2, 2])
    ridge = Ridge.new(0.1, param)
    assert_equal 1.4, ridge.forward(1)
  end

  def test_backward
    param = Param.new(Numo::SFloat[-2, 2], Numo::SFloat[-1, 1])
    ridge = Ridge.new(0.1, param)
    ridge.backward
    assert_equal Numo::SFloat[-1.2, 1.2], ridge.instance_variable_get(:@param).grad
  end
end
