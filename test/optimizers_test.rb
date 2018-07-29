require "test_helper"

include Numo
include DNN::Activations
include DNN::Layers
include DNN::Initializers
include DNN::Optimizers
Model = DNN::Model
Util = DNN::Util

class TestOptimizer < MiniTest::Unit::TestCase
  def test_initialize
    optimizer = Optimizer.new(0.1)
    assert_equal 0.1, optimizer.learning_rate
  end

  def test_to_hash
    optimizer = Optimizer.new(0.1)
    hash = optimizer.to_hash({momentum: 0.9})
    expected_hash = {
      name: "DNN::Optimizers::Optimizer",
      learning_rate: 0.1,
      momentum: 0.9,
    }
    assert_equal expected_hash, hash
  end
end


class TestSGD < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      name: "DNN::Optimizers::SGD",
      learning_rate: 0.1,
      momentum: 0.9,
    }
    sgd = SGD.load_hash(hash)
    assert_equal 0.1, sgd.learning_rate
    assert_equal 0.9, sgd.momentum
  end

  # f = ->lr, dw { lr * dw }
  # w = 0
  # w -= f.(0.1, 1)  # w => -0.1
  def test_update
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10, weight_initializer: Zeros.new)
    model << dense
    model << IdentityMSE.new
    sgd = SGD.new(0.1)
    model.compile(sgd)
    dense.grads[:weight] = SFloat.ones(*dense.params[:weight].shape)
    dense.grads[:bias] = SFloat.ones(*dense.params[:bias].shape)
    sgd.update(dense)
    assert_equal -0.1, dense.params[:weight].mean.round(2)
  end

  # f = ->lr, dw, v { lr * dw + v }
  # w = 0; v = 0
  # w -= f.(0.1, 1, v)  # w => -0.1
  # v = f.(0.1, 1, v)   # v => -0.1
  # w -= f.(0.1, 1, v)  # w => -0.3
  def test_update2
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10, weight_initializer: Zeros.new)
    model << dense
    model << IdentityMSE.new
    sgd = SGD.new(0.1, momentum: 1)
    model.compile(sgd)
    dense.grads[:weight] = SFloat.ones(*dense.params[:weight].shape)
    dense.grads[:bias] = SFloat.ones(*dense.params[:bias].shape)
    sgd.update(dense)
    sgd.update(dense)
    assert_equal -0.3, dense.params[:weight].mean.round(2)
  end

  def test_to_hash
    expected_hash = {
      name: "DNN::Optimizers::SGD",
      learning_rate: 0.01,
      momentum: 0,
    }
    sgd = SGD.new
    assert_equal expected_hash, sgd.to_hash
  end
end


class TestAdaGrad < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      name: "DNN::Optimizers::AdaGrad",
      learning_rate: 0.001,
    }
    adagrad = AdaGrad.load_hash(hash)
    assert_equal 0.001, adagrad.learning_rate
  end

  # f = ->lr, dw, g { lr / sqrt(g) * dw }
  # f2 = ->dw { dw**2 }
  # w = 0; g = 0
  # g += f2.(1)          # g => 1
  # w -= f.(0.01, 1, g)  # w => -0.01
  def test_update
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10, weight_initializer: Zeros.new)
    model << dense
    model << IdentityMSE.new
    adagrad = AdaGrad.new
    model.compile(adagrad)
    dense.grads[:weight] = SFloat.ones(*dense.params[:weight].shape)
    dense.grads[:bias] = SFloat.ones(*dense.params[:bias].shape)
    adagrad.update(dense)
    assert_equal -0.01, dense.params[:weight].mean.round(3)
  end
end


class TestRMSProp < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      name: "DNN::Optimizers::RMSProp",
      learning_rate: 0.01,
      muse: 0.8,
    }
    rmsprop = RMSProp.load_hash(hash)
    assert_equal 0.01, rmsprop.learning_rate
    assert_equal 0.8, rmsprop.muse
  end

  # f = ->lr, dw, g { lr / sqrt(g) * dw }
  # f2 = ->dw, muse, g { muse * g + (1 - muse) * dw**2 }
  # w = 0; g = 0
  # g = f2.(1, 0.5, 0)   # g => 0.5
  # w -= f.(0.01, 1, g)  # w => -0.0141
  def test_update
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10, weight_initializer: Zeros.new)
    model << dense
    model << IdentityMSE.new
    rmsprop = RMSProp.new(0.01, 0.5)
    model.compile(rmsprop)
    dense.grads[:weight] = SFloat.ones(*dense.params[:weight].shape)
    dense.grads[:bias] = SFloat.ones(*dense.params[:bias].shape)
    rmsprop.update(dense)
    assert_equal -0.0141, dense.params[:weight].mean.round(4)
  end

  def test_to_hash
    expected_hash = {
      name: "DNN::Optimizers::RMSProp",
      learning_rate: 0.001,
      muse: 0.9,
    }
    rmsprop = RMSProp.new
    assert_equal expected_hash, rmsprop.to_hash
  end
end


class TestAdam < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      name: "DNN::Optimizers::Adam",
      learning_rate: 0.01,
      beta1: 0.8,
      beta2: 0.9,
    }
    adam = Adam.load_hash(hash)
    assert_equal 0.8, adam.beta1
    assert_equal 0.9, adam.beta2
  end

  # f = ->lr2, m, v { lr2 * m / sqrt(v) }
  # f2 = ->lr, b1, b2, iter { lr * sqrt(1 - b2**iter) / (1 - b1**iter) }
  # f3 = ->b1, dw, m { (1 - b1) * (dw - m) }
  # f4 = ->b2, dw, v { (1 - b2) * (dw**2 - v) }

  # w = 0; m = 0; v = 0; b1 = 0.8; b2 = 0.9
  # lr2 = f2.(0.01, b1, b2, 1)  # lr2 => 0.0158
  # m += f3.(b1, 1, m)        # m => 0.2
  # v += f4.(b2, 1, v)        # v => 0.1
  # w -= f.(lr2, m, v)        # w => -0.01

  # lr2 = f2.(0.01, b1, b2, 2)  # lr2 => 0.0121
  # m += f3.(b1, 1, m)        # m => 0.36
  # v += f4.(b2, 1, v)        # v => 0.19
  # w -= f.(lr2, m, v)        # w -= -0.02
  def test_update
    model = Model.new
    model << InputLayer.new(10)
    dense = Dense.new(10, weight_initializer: Zeros.new)
    model << dense
    model << IdentityMSE.new
    adam = Adam.new(0.01, 0.8, 0.9)
    model.compile(adam)
    dense.grads[:weight] = SFloat.ones(*dense.params[:weight].shape)
    dense.grads[:bias] = SFloat.ones(*dense.params[:bias].shape)
    adam.update(dense)
    adam.update(dense)
    assert_equal -0.02, dense.params[:weight].mean.round(3)
  end

  def test_to_hash
    expected_hash = {
      name: "DNN::Optimizers::Adam",
      learning_rate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
    }
    adam = Adam.new
    assert_equal expected_hash, adam.to_hash
  end
end
