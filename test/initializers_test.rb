require "test_helper"

include DNN::Layers
include DNN::Optimizers
include DNN::Initializers

class TestInitializer < MiniTest::Unit::TestCase
  def test_to_hash
    initializer = Initializer.new
    hash = initializer.to_hash({mean: 1, std: 2})
    expected_hash = {
      class: "DNN::Initializers::Initializer",
      mean: 1,
      std: 2,
      seed: false,
    }
    assert_equal expected_hash, hash
  end
end


class TestConst < MiniTest::Unit::TestCase
  def test_from_hash
    hash = { class: "DNN::Initializers::Const", const: 1 }
    initializer = Const.from_hash(hash)
    assert_equal 1, initializer.const
  end

  def test_init_param
    initializer = Const.new(1)
    dense = Dense.new(10)
    dense.build([10])
    zeros = Xumo::SFloat.ones(dense.weight.data.shape)
    initializer.init_param(dense.weight, dense.input_shapes)
    assert_equal zeros, dense.weight.data
  end

  def test_to_hash
    initializer = Const.new(1)
    expected_hash = {
      class: "DNN::Initializers::Const",
      const: 1,
      seed: false,
    }
    assert_equal expected_hash, initializer.to_hash
  end
end


class TestZeros < MiniTest::Unit::TestCase
  def test_init_param
    initializer = Zeros.new
    dense = Dense.new(10)
    dense.build([10])
    zeros = Xumo::SFloat.zeros(dense.weight.data.shape)
    initializer.init_param(dense.weight, dense.input_shapes)
    assert_equal zeros, dense.weight.data
  end
end


class TestRandomNormal < MiniTest::Unit::TestCase
  def test_from_hash
    hash = { class: "DNN::Initializers::RandomNormal", mean: 1, std: 2, seed: 3 }
    initializer = RandomNormal.from_hash(hash)
    assert_equal 1, initializer.mean
    assert_equal 2, initializer.std
    assert_equal 3, initializer.instance_variable_get(:@seed)
  end

  def test_initialize
    initializer = RandomNormal.new
    assert_equal 0, initializer.mean
    assert_equal 0.05, initializer.std
  end

  def test_init_param
    initializer = RandomNormal.new(0, 0.05, seed: 0)
    dense = Dense.new(10)
    dense.build([10])
    initializer.init_param(dense.weight, dense.input_shapes)

    Xumo::SFloat.srand(0)
    expected = Xumo::SFloat.new(10, 10).rand_norm(0, 0.05).round(4)
    assert_equal expected, dense.weight.data.round(4)
  end

  def test_to_hash
    initializer = RandomNormal.new(1, 2, seed: 3)
    expected_hash = {
      class: "DNN::Initializers::RandomNormal",
      mean: 1,
      std: 2,
      seed: 3,
    }
    assert_equal expected_hash, initializer.to_hash
  end
end


class TestRandomUniform < MiniTest::Unit::TestCase
  def test_from_hash
    hash = { class: "DNN::Initializers::RandomUniform", min: -0.1, max: 0.1, seed: 3 }
    initializer = RandomUniform.from_hash(hash)
    assert_equal(-0.1, initializer.min)
    assert_equal 0.1, initializer.max
    assert_equal 3, initializer.instance_variable_get(:@seed)
  end

  def test_init_param
    initializer = RandomUniform.new(-0.05, 0.05, seed: 0)
    dense = Dense.new(10)
    dense.build([10])
    initializer.init_param(dense.weight, dense.input_shapes)

    Xumo::SFloat.srand(0)
    expected = Xumo::SFloat.new(10, 10).rand(-0.05, 0.05).round(4)
    assert_equal  expected, dense.weight.data.round(4)
  end

  def test_to_hash
    initializer = RandomUniform.new(-0.1, 0.1, seed: 3)
    expected_hash = {
      class: "DNN::Initializers::RandomUniform",
      min: -0.1,
      max: 0.1,
      seed: 3,
    }
    assert_equal expected_hash, initializer.to_hash
  end
end


class TestXavier < MiniTest::Unit::TestCase
  def test_init_param
    initializer = Xavier.new(seed: 0)
    dense = Dense.new(10)
    dense.build([10])
    initializer.init_param(dense.weight, dense.input_shapes)

    Xumo::SFloat.srand(0)
    expected = (Xumo::SFloat.new(10, 10).rand_norm / Math.sqrt(10)).round(4)
    assert_equal expected, dense.weight.data.round(4)
  end
end


class TestHe < MiniTest::Unit::TestCase
  def test_init_param
    initializer = He.new(seed: 0)
    dense = Dense.new(10)
    dense.build([10])
    initializer.init_param(dense.weight, dense.input_shapes)

    Xumo::SFloat.srand(0)
    expected = (Xumo::SFloat.new(10, 10).rand_norm / Math.sqrt(10) * Math.sqrt(2)).round(4)
    assert_equal expected, dense.weight.data.round(4)
  end
end
