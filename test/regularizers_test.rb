include DNN
include Regularizers

class TestL1 < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Regularizers::L1",
      l1_lambda: 0.1,
    }
    l1 = L1.from_hash(hash)
    assert_equal 0.1, l1.l1_lambda
  end

  def test_forward
    param = Param.new(Numo::SFloat[-2, 2])
    l1 = L1.new(0.1)
    l1.param = param
    assert_equal 1.4, l1.forward(1)
  end

  def test_backward
    x = Numo::SFloat.new(10).fill(-1)
    x[0..4] = 1
    param = Param.new(Numo::SFloat[-2, 2], Numo::SFloat[-1, 1])
    l1 = L1.new(0.1)
    l1.param = param
    l1.backward
    assert_equal Numo::SFloat[-1.1, 1.1], l1.instance_variable_get(:@param).grad.round(4)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Regularizers::L1",
      l1_lambda: 0.01,
    }
    l1 = L1.new
    assert_equal expected_hash, l1.to_hash
  end
end


class TestL2 < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Regularizers::L2",
      l2_lambda: 0.1,
    }
    l2 = L2.from_hash(hash)
    assert_equal 0.1, l2.l2_lambda
  end

  def test_forward
    param = Param.new(Numo::SFloat[-2, 2])
    l2 = L2.new(0.1)
    l2.param = param
    assert_equal 1.4, l2.forward(1)
  end

  def test_backward
    param = Param.new(Numo::SFloat[-2, 2], Numo::SFloat[-1, 1])
    l2 = L2.new(0.1)
    l2.param = param
    l2.backward
    assert_equal Numo::SFloat[-1.2, 1.2], l2.instance_variable_get(:@param).grad.round(4)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Regularizers::L2",
      l2_lambda: 0.01,
    }
    l2 = L2.new
    assert_equal expected_hash, l2.to_hash
  end
end


class TestL1L2 < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Regularizers::L1L2",
      l1_lambda: 0.1,
      l2_lambda: 0.2,
    }
    l1l2 = L1L2.from_hash(hash)
    assert_equal 0.1, l1l2.l1_lambda
    assert_equal 0.2, l1l2.l2_lambda
  end

  def test_forward
    param = Param.new(Numo::SFloat[-2, 2])
    l1l2 = L1L2.new(0.1, 0.1)
    l1l2.param = param
    assert_in_delta 1.8, l1l2.forward(1)
  end

  def test_backward
    param = Param.new(Numo::SFloat[-2, 2], Numo::SFloat[-1, 1])
    l1l2 = L1L2.new(0.1, 0.1)
    l1l2.param = param
    l1l2.backward
    assert_equal Numo::SFloat[-1.3, 1.3], l1l2.instance_variable_get(:@param).grad.round(4)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Regularizers::L1L2",
      l1_lambda: 0.01,
      l2_lambda: 0.01,
    }
    l1l2 = L1L2.new
    assert_equal expected_hash, l1l2.to_hash
  end
end
