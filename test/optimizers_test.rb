require "test_helper"

include DNN
include Activations
include Layers
include Initializers
include Optimizers

class TestOptimizer < MiniTest::Unit::TestCase
  def test_initialize
    optimizer = Optimizer.new(0.1)
    assert_equal 0.1, optimizer.learning_rate
  end

  def test_to_hash
    optimizer = Optimizer.new(0.1)
    hash = optimizer.to_hash({momentum: 0.9})
    expected_hash = {
      class: "DNN::Optimizers::Optimizer",
      learning_rate: 0.1,
      momentum: 0.9,
      clip_norm: nil,
    }
    assert_equal expected_hash, hash
  end
end


class TestSGD < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Optimizers::SGD",
      learning_rate: 0.1,
      momentum: 0.9,
      clip_norm: 1.0,
    }
    sgd = SGD.from_hash(hash)
    assert_equal 0.1, sgd.learning_rate
    assert_equal 0.9, sgd.momentum
    assert_equal 1.0, sgd.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    sgd = SGD.new(0.1)
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    sgd.update([dense])
    assert_equal -0.1, dense.params[:weight].data.mean.round(2)
  end

  def test_update2
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    sgd = SGD.new(0.1, momentum: 1)
    sgd.update([dense])
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    sgd.update([dense])
    assert_equal -0.3, dense.params[:weight].data.mean.round(2)
  end

  def test_update3
    dense = Dense.new(2, weight_initializer: Zeros.new)
    dense.build([1])
    sgd = SGD.new(0.1, clip_norm: Math.sqrt(16) / 2)
    dense.params[:weight].grad = Numo::SFloat.new(*dense.params[:weight].data.shape).fill(2)
    dense.params[:bias].grad = Numo::SFloat.new(*dense.params[:bias].data.shape).fill(2)
    sgd.update([dense])
    assert_equal -0.1, dense.params[:weight].data.mean.round(2)
    assert_equal -0.1, dense.params[:weight].data.mean.round(2)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::SGD",
      learning_rate: 0.01,
      momentum: 0,
      clip_norm: nil,
    }
    sgd = SGD.new
    assert_equal expected_hash, sgd.to_hash
  end
end


class TestNesterov < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Optimizers::Nesterov",
      learning_rate: 0.1,
      momentum: 0.8,
      clip_norm: 1.0,
    }
    nesterov = Nesterov.from_hash(hash)
    assert_equal 0.1, nesterov.learning_rate
    assert_equal 0.8, nesterov.momentum
    assert_equal 1.0, nesterov.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    nesterov = Nesterov.new(0.1, momentum: 0.9)
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    nesterov.update([dense])
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    nesterov.update([dense])
    assert_equal -0.6149, dense.params[:weight].data.mean.round(5)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::Nesterov",
      learning_rate: 0.01,
      momentum: 0.9,
      clip_norm: nil,
    }
    nesterov = Nesterov.new
    assert_equal expected_hash, nesterov.to_hash
  end
end


class TestAdaGrad < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Optimizers::AdaGrad",
      learning_rate: 0.001,
      eps: 1e-4,
      clip_norm: 1.0,
    }
    adagrad = AdaGrad.from_hash(hash)
    assert_equal 0.001, adagrad.learning_rate
    assert_equal 1e-4, adagrad.eps
    assert_equal 1.0, adagrad.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    adagrad = AdaGrad.new
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    adagrad.update([dense])
    assert_equal -0.01, dense.params[:weight].data.mean.round(3)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::AdaGrad",
      learning_rate: 0.01,
      eps: 1e-7,
      clip_norm: nil,
    }
    adagrad = AdaGrad.new
    assert_equal expected_hash, adagrad.to_hash
  end
end


class TestRMSProp < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Optimizers::RMSProp",
      learning_rate: 0.01,
      alpha: 0.8,
      eps: 1e-4,
      clip_norm: 1.0,
    }
    rmsprop = RMSProp.from_hash(hash)
    assert_equal 0.01, rmsprop.learning_rate
    assert_equal 0.8, rmsprop.alpha
    assert_equal 1e-4, rmsprop.eps
    assert_equal 1.0, rmsprop.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    rmsprop = RMSProp.new(0.01, alpha: 0.5)
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    rmsprop.update([dense])
    assert_equal -0.0141, dense.params[:weight].data.mean.round(4)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::RMSProp",
      learning_rate: 0.001,
      alpha: 0.9,
      eps: 1e-7,
      clip_norm: nil,
    }
    rmsprop = RMSProp.new
    assert_equal expected_hash, rmsprop.to_hash
  end
end


class TestAdaDelta < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Optimizers::AdaDelta",
      rho: 0.96,
      eps: 1e-4,
      clip_norm: 1.0,
    }
    adadelta = AdaDelta.from_hash(hash)
    assert_equal 0.96, adadelta.rho
    assert_equal 1e-4, adadelta.eps
    assert_equal 1.0, adadelta.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    adadelta = AdaDelta.new(rho: 0.5)
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    adadelta.update([dense])
    assert_equal -0.0014, dense.params[:weight].data.mean.round(4)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::AdaDelta",
      learning_rate: nil,
      rho: 0.95,
      eps: 1e-6,
      clip_norm: nil,
    }
    adadelta = AdaDelta.new
    assert_equal expected_hash, adadelta.to_hash
  end
end


class TestAdam < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Optimizers::Adam",
      alpha: 0.01,
      beta1: 0.8,
      beta2: 0.9,
      eps: 1e-4,
      clip_norm: 1.0,
    }
    adam = Adam.from_hash(hash)
    assert_equal 0.01, adam.alpha
    assert_equal 0.8, adam.beta1
    assert_equal 0.9, adam.beta2
    assert_equal 1e-4, adam.eps
    assert_equal 1.0, adam.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    adam = Adam.new(alpha: 0.01, beta1: 0.8, beta2: 0.9)
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    adam.update([dense])
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    adam.update([dense])
    assert_equal -0.02, dense.params[:weight].data.mean.round(3)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::Adam",
      learning_rate: nil,
      alpha: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-7,
      clip_norm: nil,
    }
    adam = Adam.new
    assert_equal expected_hash, adam.to_hash
  end
end


class TestRMSPropGraves < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Optimizers::RMSPropGraves",
      learning_rate: 0.001,
      alpha: 0.8,
      eps: 1e-7,
      clip_norm: 1.0,
    }
    rmsprop = RMSPropGraves.from_hash(hash)
    assert_equal 0.001, rmsprop.learning_rate
    assert_equal 0.8, rmsprop.alpha
    assert_equal 1e-7, rmsprop.eps
    assert_equal 1.0, rmsprop.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    rmsprop = RMSPropGraves.new(0.01, alpha: 0.5)
    dense.params[:weight].grad = Numo::SFloat.ones(*dense.params[:weight].data.shape)
    rmsprop.update([dense])
    assert_equal -0.02, dense.params[:weight].data.mean.round(4)
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::RMSPropGraves",
      learning_rate: 0.0001,
      alpha: 0.95,
      eps: 0.0001,
      clip_norm: nil,
    }
    rmsprop = RMSPropGraves.new
    assert_equal expected_hash, rmsprop.to_hash
  end
end
