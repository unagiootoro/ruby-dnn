require "test_helper"

class TestOptimizer < MiniTest::Unit::TestCase
  def test_to_hash
    optimizer = DNN::Optimizers::Optimizer.new
    hash = optimizer.to_hash({momentum: 0.9})
    expected_hash = {
      class: "DNN::Optimizers::Optimizer",
      momentum: 0.9,
      clip_norm: nil,
    }
    assert_equal expected_hash, hash
  end
end


class TestSGD < MiniTest::Unit::TestCase
  Dense = DNN::Layers::Dense
  Zeros = DNN::Initializers::Zeros
  SGD = DNN::Optimizers::SGD

  def test_from_hash
    hash = {
      class: "DNN::Optimizers::SGD",
      lr: 0.1,
      momentum: 0.9,
      clip_norm: 1.0,
    }
    sgd = SGD.from_hash(hash)
    assert_equal 0.1, sgd.lr
    assert_equal 0.9, sgd.momentum
    assert_equal 1.0, sgd.clip_norm
  end

  # It SGD is works as expected.
  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    sgd = SGD.new(lr: 0.1)
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    sgd.update([dense.weight])
    assert_equal(-0.1, dense.weight.data.mean.round(2))
  end

  # It Momentum SGD is works as expected.
  def test_update2
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    sgd = SGD.new(lr: 0.1, momentum: 1)
    sgd.update([dense.weight])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    sgd.update([dense.weight])
    assert_equal(-0.3, dense.weight.data.mean.round(2))
  end

  # It clip norm is works as expected.
  def test_update3
    dense = Dense.new(2, weight_initializer: Zeros.new)
    dense.build([1])
    sgd = SGD.new(lr: 0.1, clip_norm: Math.sqrt(16) / 2)
    dense.weight.grad = Xumo::SFloat.new(*dense.weight.data.shape).fill(2)
    dense.bias.grad = Xumo::SFloat.new(*dense.bias.data.shape).fill(2)
    sgd.update([dense.weight, dense.bias])
    assert_equal(-0.1, dense.weight.data.mean.round(2))
    assert_equal(-0.1, dense.weight.data.mean.round(2))
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::SGD",
      lr: 0.01,
      momentum: 0,
      clip_norm: nil,
    }
    sgd = SGD.new
    assert_equal expected_hash, sgd.to_hash
  end
end


class TestNesterov < MiniTest::Unit::TestCase
  Dense = DNN::Layers::Dense
  Zeros = DNN::Initializers::Zeros
  Nesterov = DNN::Optimizers::Nesterov

  def test_from_hash
    hash = {
      class: "DNN::Optimizers::Nesterov",
      lr: 0.1,
      momentum: 0.8,
      clip_norm: 1.0,
    }
    nesterov = Nesterov.from_hash(hash)
    assert_equal 0.1, nesterov.lr
    assert_equal 0.8, nesterov.momentum
    assert_equal 1.0, nesterov.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    nesterov = Nesterov.new(lr: 0.1, momentum: 0.9)
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    nesterov.update([dense.weight])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    nesterov.update([dense.weight])
    assert_equal(-0.6149, dense.weight.data.mean.round(5))
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::Nesterov",
      lr: 0.01,
      momentum: 0.9,
      clip_norm: nil,
    }
    nesterov = Nesterov.new
    assert_equal expected_hash, nesterov.to_hash
  end
end


class TestAdaGrad < MiniTest::Unit::TestCase
  Dense = DNN::Layers::Dense
  Zeros = DNN::Initializers::Zeros
  AdaGrad = DNN::Optimizers::AdaGrad

  def test_from_hash
    hash = {
      class: "DNN::Optimizers::AdaGrad",
      lr: 0.001,
      eps: 1e-4,
      clip_norm: 1.0,
    }
    adagrad = AdaGrad.from_hash(hash)
    assert_equal 0.001, adagrad.lr
    assert_equal 1e-4, adagrad.eps
    assert_equal 1.0, adagrad.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    adagrad = AdaGrad.new
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adagrad.update([dense.weight])
    assert_equal(-0.01, dense.weight.data.mean.round(3))
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::AdaGrad",
      lr: 0.01,
      eps: 1e-7,
      clip_norm: nil,
    }
    adagrad = AdaGrad.new
    assert_equal expected_hash, adagrad.to_hash
  end
end


class TestRMSProp < MiniTest::Unit::TestCase
  Dense = DNN::Layers::Dense
  Zeros = DNN::Initializers::Zeros
  RMSProp = DNN::Optimizers::RMSProp

  def test_from_hash
    hash = {
      class: "DNN::Optimizers::RMSProp",
      lr: 0.01,
      alpha: 0.8,
      eps: 1e-4,
      clip_norm: 1.0,
    }
    rmsprop = RMSProp.from_hash(hash)
    assert_equal 0.01, rmsprop.lr
    assert_equal 0.8, rmsprop.alpha
    assert_equal 1e-4, rmsprop.eps
    assert_equal 1.0, rmsprop.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    rmsprop = RMSProp.new(lr: 0.01, alpha: 0.5)
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    rmsprop.update([dense.weight])
    assert_equal(-0.0141, dense.weight.data.mean.round(4))
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::RMSProp",
      lr: 0.001,
      alpha: 0.9,
      eps: 1e-7,
      clip_norm: nil,
    }
    rmsprop = RMSProp.new
    assert_equal expected_hash, rmsprop.to_hash
  end
end


class TestAdaDelta < MiniTest::Unit::TestCase
  Dense = DNN::Layers::Dense
  Zeros = DNN::Initializers::Zeros
  AdaDelta = DNN::Optimizers::AdaDelta

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
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adadelta.update([dense.weight])
    assert_equal(-0.0014, dense.weight.data.mean.round(4))
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::AdaDelta",
      rho: 0.95,
      eps: 1e-6,
      clip_norm: nil,
    }
    adadelta = AdaDelta.new
    assert_equal expected_hash, adadelta.to_hash
  end
end


class TestRMSPropGraves < MiniTest::Unit::TestCase
  Dense = DNN::Layers::Dense
  Zeros = DNN::Initializers::Zeros
  RMSPropGraves = DNN::Optimizers::RMSPropGraves

  def test_from_hash
    hash = {
      class: "DNN::Optimizers::RMSPropGraves",
      lr: 0.001,
      alpha: 0.8,
      eps: 1e-7,
      clip_norm: 1.0,
    }
    rmsprop = RMSPropGraves.from_hash(hash)
    assert_equal 0.001, rmsprop.lr
    assert_equal 0.8, rmsprop.alpha
    assert_equal 1e-7, rmsprop.eps
    assert_equal 1.0, rmsprop.clip_norm
  end

  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    rmsprop = RMSPropGraves.new(lr: 0.01, alpha: 0.5)
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    rmsprop.update([dense.weight])
    assert_equal(-0.02, dense.weight.data.mean.round(4))
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::RMSPropGraves",
      lr: 0.0001,
      alpha: 0.95,
      eps: 0.0001,
      clip_norm: nil,
    }
    rmsprop = RMSPropGraves.new
    assert_equal expected_hash, rmsprop.to_hash
  end
end


class TestAdam < MiniTest::Unit::TestCase
  Dense = DNN::Layers::Dense
  Zeros = DNN::Initializers::Zeros
  Adam = DNN::Optimizers::Adam

  def test_from_hash
    hash = {
      class: "DNN::Optimizers::Adam",
      alpha: 0.01,
      beta1: 0.8,
      beta2: 0.9,
      eps: 1e-4,
      amsgrad: true,
      clip_norm: 1.0,
    }
    adam = Adam.from_hash(hash)
    assert_equal 0.01, adam.alpha
    assert_equal 0.8, adam.beta1
    assert_equal 0.9, adam.beta2
    assert_equal 1e-4, adam.eps
    assert_equal true, adam.amsgrad
    assert_equal 1.0, adam.clip_norm
  end

  # It Adam is works as expected.
  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    adam = Adam.new(alpha: 0.01, beta1: 0.8, beta2: 0.9)
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adam.update([dense.weight])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adam.update([dense.weight])
    assert_equal(-0.02, dense.weight.data.mean.round(3))
  end

  # It AMSGrad is works as expected.
  def test_update2
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    adam = Adam.new(alpha: 0.01, beta1: 0.8, beta2: 0.9, amsgrad: true)
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape) * 10
    adam.update([dense.weight])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adam.update([dense.weight])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adam.update([dense.weight])
    assert_equal(-0.022, dense.weight.data.mean.round(3))
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::Adam",
      alpha: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-7,
      amsgrad: false,
      clip_norm: nil,
    }
    adam = Adam.new
    assert_equal expected_hash, adam.to_hash
  end
end


class TestAdaBound < MiniTest::Unit::TestCase
  Dense = DNN::Layers::Dense
  Zeros = DNN::Initializers::Zeros
  AdaBound = DNN::Optimizers::AdaBound

  def test_from_hash
    hash = {
      class: "DNN::Optimizers::AdaBound",
      alpha: 0.01,
      beta1: 0.8,
      beta2: 0.9,
      final_lr: 0.01,
      gamma: 0.01,
      eps: 1e-4,
      amsgrad: true,
      clip_norm: 1.0,
    }
    adabound = AdaBound.from_hash(hash)
    assert_equal 0.01, adabound.alpha
    assert_equal 0.8, adabound.beta1
    assert_equal 0.9, adabound.beta2
    assert_equal 0.01, adabound.final_lr
    assert_equal 0.01, adabound.gamma
    assert_equal 1e-4, adabound.eps
    assert_equal true, adabound.amsgrad
    assert_equal 1.0, adabound.clip_norm
  end

  # It AdaBound is works as expected.
  def test_update
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    adabound = AdaBound.new(alpha: 0.01, beta1: 0.8, beta2: 0.9)
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adabound.update([dense.weight])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adabound.update([dense.weight])
    assert_equal(-0.02, dense.weight.data.mean.round(3))
  end

# It AMSBound is works as expected.
  def test_update2
    dense = Dense.new(10, weight_initializer: Zeros.new)
    dense.build([10])
    adabound = AdaBound.new(alpha: 0.01, beta1: 0.8, beta2: 0.9, amsgrad: true)
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape) * 10
    adabound.update([dense.weight])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adabound.update([dense.weight])
    dense.weight.grad = Xumo::SFloat.ones(*dense.weight.data.shape)
    adabound.update([dense.weight])
    assert_equal(-0.022, dense.weight.data.mean.round(3))
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Optimizers::AdaBound",
      alpha: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      final_lr: 0.1,
      gamma: 0.001,
      eps: 1e-7,
      amsgrad: false,
      clip_norm: nil,
    }
    adabound = AdaBound.new
    assert_equal expected_hash, adabound.to_hash
  end
end
