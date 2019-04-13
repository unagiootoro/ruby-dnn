require "test_helper"

include DNN
include Layers
include Activations
include Optimizers
include Initializers

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
  def test_load_hash
    hash = {const: 1}
    initializer = Const.load_hash(hash)
    assert_equal 1, initializer.const
  end

  def test_init_param
    initializer = Const.new(1)
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10)
    model << dense
    model << IdentityMSE.new
    model.compile(SGD.new)
    zeros = Numo::SFloat.ones(dense.params[:weight].data.shape)
    initializer.init_param(self, dense.params[:weight])
    assert_equal zeros, dense.params[:weight].data
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
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10)
    model << dense
    model << IdentityMSE.new
    model.compile(SGD.new)
    zeros = Numo::SFloat.zeros(dense.params[:weight].data.shape)
    initializer.init_param(self, dense.params[:weight])
    assert_equal zeros, dense.params[:weight].data
  end
end


class TestRandomNorm < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {mean: 1, std: 2, seed: 3}
    initializer = RandomNormal.load_hash(hash)
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
    initializer = RandomNormal.new(0, 0.05, 0)
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10)
    model << dense
    model << IdentityMSE.new
    model.compile(SGD.new)
    initializer.init_param(dense, dense.params[:weight])

    Numo::SFloat.srand(0)
    expected = Numo::SFloat.new(10, 10).rand_norm(0, 0.05).round(4)
    assert_equal expected, Numo::SFloat, dense.params[:weight].data.round(4)
  end

  def test_to_hash
    initializer = RandomNormal.new(1, 2, 3)
    expected_hash = {
      class: "DNN::Initializers::RandomNormal",
      mean: 1,
      std: 2,
      seed: 3,
    }
    assert_equal expected_hash, initializer.to_hash
  end
end


class TestRandomNorm < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {min: -0.1, max: 0.1, seed: 3}
    initializer = RandomUniform.load_hash(hash)
    assert_equal -0.1, initializer.min
    assert_equal 0.1, initializer.max
    assert_equal 3, initializer.instance_variable_get(:@seed)
  end

  def test_init_param
    initializer = RandomUniform.new(-0.05, 0.05, 0)
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10)
    model << dense
    model << IdentityMSE.new
    model.compile(SGD.new)
    initializer.init_param(dense, dense.params[:weight])

    Numo::SFloat.srand(0)
    expected = Numo::SFloat.new(10, 10).rand(-0.05, 0.05).round(4)
    assert_equal  expected, dense.params[:weight].data.round(4)
  end

  def test_to_hash
    initializer = RandomUniform.new(-0.1, 0.1, 3)
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
    initializer = Xavier.new(0)
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10)
    model << dense
    model << IdentityMSE.new
    model.compile(SGD.new)
    initializer.init_param(dense, dense.params[:weight])

    Numo::SFloat.srand(0)
    expected = (Numo::SFloat.new(10, 10).rand_norm / Math.sqrt(10)).round(4)
    assert_equal expected, dense.params[:weight].data.round(4)
  end
end


class TestHe < MiniTest::Unit::TestCase
  def test_init_param
    initializer = He.new
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10)
    model << dense
    model << IdentityMSE.new
    model.compile(SGD.new)
    initializer.init_param(dense, dense.params[:weight])

    Numo::SFloat.srand(0)
    expected = (Numo::SFloat.new(10, 10).rand_norm / Math.sqrt(10) * Math.sqrt(2)).round(4)
    assert_equal expected, dense.params[:weight].data.round(4)
  end
end
