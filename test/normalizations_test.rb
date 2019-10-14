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

  def test_forward
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    batch_norm.gamma.data = Numo::SFloat.new(10).fill(3)
    batch_norm.beta.data = Numo::SFloat.new(10).fill(10)
    x = Numo::SFloat.cast([Numo::SFloat.new(10).fill(10), Numo::SFloat.new(10).fill(20)])
    expected = Numo::SFloat.cast([Numo::SFloat.new(10).fill(7), Numo::SFloat.new(10).fill(13)])
    DNN.learning_phase = true
    assert_equal expected, batch_norm.forward(x).round(4)
  end

  def test_forward2
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    batch_norm.gamma.data = Numo::SFloat.new(10).fill(3)
    batch_norm.beta.data = Numo::SFloat.new(10).fill(10)
    batch_norm.running_mean.data = Numo::SFloat.new(10).fill(15)
    batch_norm.running_var.data = Numo::SFloat.new(10).fill(25)
    x = Numo::SFloat.cast([Numo::SFloat.new(10).fill(10), Numo::SFloat.new(10).fill(20)])
    expected = Numo::SFloat.cast([Numo::SFloat.new(10).fill(7), Numo::SFloat.new(10).fill(13)])
    DNN.learning_phase = false
    assert_equal expected, batch_norm.forward(x).round(4)
  end

  def test_backward
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    x = Numo::SFloat.cast([Numo::SFloat.new(10).fill(10), Numo::SFloat.new(10).fill(20)])
    DNN.learning_phase = true
    batch_norm.forward(x)
    grad = batch_norm.backward(Numo::SFloat.ones(*x.shape))
    assert_equal Numo::SFloat[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], grad.round(4)
    assert_equal Numo::SFloat[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], batch_norm.gamma.grad
    assert_equal Numo::SFloat[[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], batch_norm.beta.grad
  end

  def test_backward2
    batch_norm = BatchNormalization.new
    batch_norm.build([10])
    batch_norm.trainable = false
    x = Numo::SFloat.cast([Numo::SFloat.new(10).fill(10), Numo::SFloat.new(10).fill(20)])
    DNN.learning_phase = true
    batch_norm.forward(x)
    grad = batch_norm.backward(Numo::SFloat.ones(*x.shape))
    assert_equal Numo::SFloat[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], grad.round(4)
    assert_equal Numo::SFloat[0], batch_norm.gamma.grad
    assert_equal Numo::SFloat[0], batch_norm.beta.grad
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Layers::BatchNormalization",
      name: nil,
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
    assert_equal expected_hash, batch_norm.get_params
  end
end
