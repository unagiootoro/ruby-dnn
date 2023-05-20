require "test_helper"

include DNN::Layers
include DNN::Optimizers

class TestBatchNormalization < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::BatchNormalization",
      axis: 1,
      momentum: 0.8,
    }
    batch_norm = BatchNormalization.from_hash(hash)
    assert_equal 1, batch_norm.axis
    assert_equal 0.8, batch_norm.momentum
    assert_equal 1e-7, batch_norm.eps
  end

  def test_call
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    batch_norm.gamma.data = Xumo::SFloat.new(10).fill(3)
    batch_norm.beta.data = Xumo::SFloat.new(10).fill(10)
    x = DNN::Variable.new(Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(10), Xumo::SFloat.new(10).fill(20)]))
    expected = Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(7), Xumo::SFloat.new(10).fill(13)])
    batch_norm.set_learning_phase(true)
    assert_equal expected, batch_norm.(x).data.round(4)
  end

  def test_call2
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    batch_norm.gamma.data = Xumo::SFloat.new(10).fill(3)
    batch_norm.beta.data = Xumo::SFloat.new(10).fill(10)
    batch_norm.running_mean.data = Xumo::SFloat.new(10).fill(15)
    batch_norm.running_var.data = Xumo::SFloat.new(10).fill(25)
    x = DNN::Variable.new(Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(10), Xumo::SFloat.new(10).fill(20)]))
    expected = Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(7), Xumo::SFloat.new(10).fill(13)])
    batch_norm.set_learning_phase(false)
    assert_equal expected, batch_norm.(x).data.round(4)
  end

  def test_call3
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    x = DNN::Variable.new(Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(10), Xumo::SFloat.new(10).fill(20)]))
    batch_norm.set_learning_phase(true)
    y = batch_norm.(x)
    y.backward(Xumo::SFloat.ones(*x.shape))
    assert_equal Xumo::SFloat[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], x.grad.round(4)
    assert_equal Xumo::SFloat[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], batch_norm.gamma.grad
    assert_equal Xumo::SFloat[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], batch_norm.beta.grad
  end

  def test_call4
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    batch_norm.set_trainable(false)
    x = DNN::Variable.new(Xumo::SFloat.cast([Xumo::SFloat.new(10).fill(10), Xumo::SFloat.new(10).fill(20)]))
    batch_norm.set_learning_phase(true)
    y = batch_norm.(x)
    y.backward(Xumo::SFloat.ones(*x.shape))
    assert_equal Xumo::SFloat[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], x.grad.round(4)
    assert_equal Xumo::SFloat[0], batch_norm.gamma.grad
    assert_equal Xumo::SFloat[0], batch_norm.beta.grad
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Layers::BatchNormalization",
      axis: 1,
      momentum: 0.8,
      eps: 1e-4,
    }
    batch_norm = BatchNormalization.new(axis: 1, momentum: 0.8, eps: 1e-4)
    batch_norm.build([10])
    assert_equal expected_hash, batch_norm.to_hash
  end

  def test_get_params
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    expected_hash = {
      gamma: batch_norm.gamma,
      beta: batch_norm.beta,
      running_mean: batch_norm.running_mean,
      running_var: batch_norm.running_var,
    }
    assert_equal expected_hash, batch_norm.get_variables
  end
end
