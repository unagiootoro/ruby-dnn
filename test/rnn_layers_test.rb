require "test_helper"

include DNN
include Layers
include Activations
include Optimizers
include Losses

class TestRNN < MiniTest::Unit::TestCase
  def test_initialize
    rnn = RNN.new(64, stateful: true, return_sequences: false,
                  weight_initializer: RandomUniform.new,
                  bias_initializer: RandomUniform.new,
                  l1_lambda: 0.1, l2_lambda: 0.2)
    assert_equal 64, rnn.num_nodes
    assert_equal true, rnn.stateful
    assert_equal false, rnn.return_sequences
  end

  def test_build
    rnn = RNN.new(64, return_sequences: false)
    rnn.build([16, 64])
    assert_equal 16, rnn.instance_variable_get(:@time_length)
  end

  def test_output_shape
    rnn = RNN.new(64)
    rnn.build([16, 64])
    assert_equal [16, 64], rnn.output_shape
  end

  def test_output_shape2
    rnn = RNN.new(64, return_sequences: false)
    rnn.build([16, 64])
    assert_equal [64], rnn.output_shape
  end

  def test_reset_state
    rnn = RNN.new(64)
    rnn.build([16, 64])
    rnn.params[:hidden].data = Numo::SFloat.ones(16, 64)
    rnn.reset_state
    assert_equal Numo::SFloat.zeros(16, 64), rnn.params[:hidden].data
  end

  def test_to_hash
    rnn = RNN.new(64, stateful: true, return_sequences: false,
                  l1_lambda: 0.1, l2_lambda: 0.2, use_bias: false)
    expected_hash = {
      class: "DNN::Layers::RNN",
      num_nodes: 64,
      weight_initializer: rnn.weight_initializer.to_hash,
      bias_initializer: rnn.bias_initializer.to_hash,
      l1_lambda: 0.1,
      l2_lambda: 0.2,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    assert_equal expected_hash, rnn.to_hash
  end

  def test_regularizers
    dense = RNN.new(1, l1_lambda: 1, l2_lambda: 1)
    dense.build([10])
    assert_kind_of Lasso, dense.regularizers[0]
    assert_kind_of Lasso, dense.regularizers[1]
    assert_kind_of Ridge, dense.regularizers[2]
    assert_kind_of Ridge, dense.regularizers[3]
  end

  def test_regularizers2
    dense = RNN.new(1)
    dense.build([10])
    assert_equal [], dense.regularizers
  end
end



class TestSimpleRNN_Dense < MiniTest::Unit::TestCase
  def test_forward
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    w = Param.new
    w.data = Numo::SFloat.new(64, 16).fill(1)
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16).fill(1)
    b = Param.new
    b.data = Numo::SFloat.new(16).fill(0)

    dense = SimpleRNN_Dense.new(w, w2, b, Tanh.new)
    assert_equal [1, 16], dense.forward(x, h).shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(2, 16).seq[1, false].reshape(1, 16)
    w = Param.new
    w.data = Numo::SFloat.new(64, 16).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16).fill(1)
    w2.grad = 0
    b = Param.new
    b.data = Numo::SFloat.new(16).fill(0)
    b.grad = 0

    dense = SimpleRNN_Dense.new(w, w2, b, Tanh.new)
    dense.forward(x, h)
    dx, dh = dense.backward(dh2)
    assert_equal [1, 64], dx.shape
    assert_equal [1, 16], dh.shape
  end

  def test_backward2
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(2, 16).seq[1, false].reshape(1, 16)
    w = Param.new
    w.data = Numo::SFloat.new(64, 16).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16).fill(1)
    w2.grad = 0

    dense = SimpleRNN_Dense.new(w, w2, nil, Tanh.new)
    dense.forward(x, h)
    dense.backward(dh2)
    assert_nil dense.instance_variable_get(:@bias)
  end

  def test_backward3
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(2, 16).seq[1, false].reshape(1, 16)
    w = Param.new
    w.data = Numo::SFloat.new(64, 16).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16).fill(1)
    w2.grad = 0
    b = Param.new
    b.data = Numo::SFloat.new(16).fill(0)
    b.grad = 0

    dense = SimpleRNN_Dense.new(w, w2, b, Tanh.new)
    dense.trainable = false
    dense.forward(x, h)
    dx, dh = dense.backward(dh2)
    assert_equal 0, dense.instance_variable_get(:@weight).grad
    assert_equal 0, dense.instance_variable_get(:@recurrent_weight).grad
    assert_equal 0, dense.instance_variable_get(:@bias).grad
  end
end


class TestSimpleRNN < MiniTest::Unit::TestCase
  
  def test_from_hash
    hash = {
      class: "DNN::Layers::SimpleRNN",
      num_nodes: 64,
      weight_initializer: RandomUniform.new.to_hash,
      bias_initializer: RandomUniform.new.to_hash,
      l1_lambda: 0.1,
      l2_lambda: 0.2,
      use_bias: false,
      stateful: true,
      return_sequences: false,
      activation: ReLU.new.to_hash,
    }
    rnn = SimpleRNN.from_hash(hash)
    assert_equal 64, rnn.num_nodes
    assert_kind_of RandomUniform, rnn.weight_initializer
    assert_kind_of RandomUniform, rnn.bias_initializer
    assert_equal 0.1, rnn.l1_lambda
    assert_equal 0.2, rnn.l2_lambda
    assert_equal false, rnn.use_bias
    assert_equal true, rnn.stateful
    assert_equal false, rnn.return_sequences
    assert_kind_of ReLU, rnn.activation
  end

  def test_forward
    x = Numo::SFloat.new(1, 16, 64).seq
    rnn = SimpleRNN.new(64)
    rnn.build([16, 64])
    assert_equal [1, 16, 64], rnn.forward(x).shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 16, 64).seq
    y = Numo::SFloat.new(1, 16, 64).seq
    rnn = SimpleRNN.new(64)
    rnn.build([16, 64])
    rnn.forward(x)
    assert_equal [1, 16, 64], rnn.backward(y).shape
  end

  def test_backward2
    x = Numo::SFloat.new(1, 16, 64).seq
    y = Numo::SFloat.new(1, 16, 64).seq
    rnn = SimpleRNN.new(64, use_bias: false)
    rnn.build([16, 64])
    rnn.forward(x)
    rnn.backward(y)
    assert_nil rnn.params[:bias]
  end

  def test_backward3
    x = Numo::SFloat.new(1, 16, 64).seq
    y = Numo::SFloat.new(1, 16, 64).seq
    rnn = SimpleRNN.new(64)
    rnn.trainable = false
    rnn.build([16, 64])
    rnn.forward(x)
    rnn.backward(y)
    assert_equal 0, rnn.params[:weight].grad
    assert_equal 0, rnn.params[:recurrent_weight].grad
    assert_equal 0, rnn.params[:bias].grad
  end

  def test_to_hash
    rnn = SimpleRNN.new(64, stateful: true, return_sequences: false,
                        l1_lambda: 0.1, l2_lambda: 0.2, use_bias: false, activation: ReLU.new)
    expected_hash = {
      class: "DNN::Layers::SimpleRNN",
      num_nodes: 64,
      weight_initializer: rnn.weight_initializer.to_hash,
      bias_initializer: rnn.bias_initializer.to_hash,
      l1_lambda: 0.1,
      l2_lambda: 0.2,
      use_bias: false,
      stateful: true,
      return_sequences: false,
      activation: rnn.activation.to_hash,
    }
    assert_equal expected_hash, rnn.to_hash
  end

  def test_build
    rnn = SimpleRNN.new(64, weight_initializer: Const.new(2), bias_initializer: Const.new(2))
    rnn.build([16, 32])
    assert_equal Numo::SFloat.new(32, 64).fill(2), rnn.params[:weight].data
    assert_equal Numo::SFloat.new(64, 64).fill(2), rnn.params[:recurrent_weight].data
    assert_equal Numo::SFloat.new(64).fill(2), rnn.params[:bias].data
    assert_kind_of SimpleRNN_Dense, rnn.instance_variable_get(:@layers)[15]
  end
end


class TestLSTM_Dense < MiniTest::Unit::TestCase
  def test_forward
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    c = Numo::SFloat.new(1, 16).seq
    w = Param.new
    w.data = Numo::SFloat.new(64, 16 * 4).fill(1)
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16 * 4).fill(1)
    b = Param.new
    b.data = Numo::SFloat.new(16 * 4).fill(0)

    dense = LSTM_Dense.new(w, w2, b)
    h2, c2 = dense.forward(x, h, c)
    assert_equal [1, 16], h2.shape
    assert_equal [1, 16], c2.shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(1, 16).seq
    c = Numo::SFloat.new(1, 16).seq
    dc2 = Numo::SFloat.new(1, 16).seq
    w = Param.new
    w.data = Numo::SFloat.new(64, 16 * 4).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16 * 4).fill(1)
    w2.grad = 0
    b = Param.new
    b.data = Numo::SFloat.new(16 * 4).fill(0)
    b.grad = 0

    dense = LSTM_Dense.new(w, w2, b)
    dense.forward(x, h, c)
    dx, dh, dc = dense.backward(dh2, dc2)
    assert_equal [1, 64], dx.shape
    assert_equal [1, 16], dh.shape
    assert_equal [1, 16], dc.shape
  end

  def test_backward2
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(1, 16).seq
    c = Numo::SFloat.new(1, 16).seq
    dc2 = Numo::SFloat.new(1, 16).seq
    w = Param.new
    w.data = Numo::SFloat.new(64, 16 * 4).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16 * 4).fill(1)
    w2.grad = 0

    dense = LSTM_Dense.new(w, w2, nil)
    dense.forward(x, h, c)
    dense.backward(dh2, dc2)
    assert_nil dense.instance_variable_get(:@bias)
  end

  def test_backward3
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(1, 16).seq
    c = Numo::SFloat.new(1, 16).seq
    dc2 = Numo::SFloat.new(1, 16).seq
    w = Param.new
    w.data = Numo::SFloat.new(64, 16 * 4).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16 * 4).fill(1)
    w2.grad = 0
    b = Param.new
    b.data = Numo::SFloat.new(16 * 4).fill(0)
    b.grad = 0

    dense = LSTM_Dense.new(w, w2, b)
    dense.trainable = false
    dense.forward(x, h, c)
    dense.backward(dh2, dc2)
    assert_equal 0, dense.instance_variable_get(:@weight).grad
    assert_equal 0, dense.instance_variable_get(:@recurrent_weight).grad
    assert_equal 0, dense.instance_variable_get(:@bias).grad
  end
end


class TestLSTM < MiniTest::Unit::TestCase
  
  def test_from_hash
    hash = {
      class: "DNN::Layers::LSTM",
      num_nodes: 64,
      weight_initializer: RandomUniform.new.to_hash,
      bias_initializer: RandomUniform.new.to_hash,
      l1_lambda: 0.1,
      l2_lambda: 0.2,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    lstm = LSTM.from_hash(hash)
    assert_equal 64, lstm.num_nodes
    assert_kind_of RandomUniform, lstm.weight_initializer
    assert_kind_of RandomUniform, lstm.bias_initializer
    assert_equal 0.1, lstm.l1_lambda
    assert_equal 0.2, lstm.l2_lambda
    assert_equal false, lstm.use_bias
    assert_equal true, lstm.stateful
    assert_equal false, lstm.return_sequences
  end

  def test_forward
    x = Numo::SFloat.new(1, 16, 64).seq
    lstm = LSTM.new(64)
    lstm.build([16, 64])
    assert_equal [1, 16, 64], lstm.forward(x).shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 16, 64).seq
    y = Numo::SFloat.new(1, 16, 64).seq
    lstm = LSTM.new(64)
    lstm.build([16, 64])
    lstm.forward(x)
    assert_equal [1, 16, 64], lstm.backward(y).shape
  end

  def test_backward2
    x = Numo::SFloat.new(1, 16, 64).seq
    y = Numo::SFloat.new(1, 16, 64).seq
    lstm = LSTM.new(64, use_bias: false)
    lstm.build([16, 64])
    lstm.forward(x)
    lstm.backward(y)
    assert_nil lstm.params[:bias]
  end

  def test_backward3
    x = Numo::SFloat.new(1, 16, 64).seq
    y = Numo::SFloat.new(1, 16, 64).seq
    lstm = LSTM.new(64)
    lstm.trainable = false
    lstm.build([16, 64])
    lstm.forward(x)
    lstm.backward(y)
    assert_equal 0, lstm.params[:weight].grad
    assert_equal 0, lstm.params[:recurrent_weight].grad
    assert_equal 0, lstm.params[:bias].grad
  end

  def test_to_hash
    lstm = LSTM.new(64, stateful: true, return_sequences: false,
                        l1_lambda: 0.1, l2_lambda: 0.2, use_bias: false)
    expected_hash = {
      class: "DNN::Layers::LSTM",
      num_nodes: 64,
      weight_initializer: lstm.weight_initializer.to_hash,
      bias_initializer: lstm.bias_initializer.to_hash,
      l1_lambda: 0.1,
      l2_lambda: 0.2,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    assert_equal expected_hash, lstm.to_hash
  end

  def test_reset_state
    lstm = LSTM.new(64)
    lstm.build([16, 64])
    lstm.params[:hidden].data = Numo::SFloat.ones(16, 64)
    lstm.params[:cell].data = Numo::SFloat.ones(16, 64)
    lstm.reset_state
    assert_equal Numo::SFloat.zeros(16, 64), lstm.params[:hidden].data
    assert_equal Numo::SFloat.zeros(16, 64), lstm.params[:cell].data
  end

  def test_build
    lstm = LSTM.new(64, weight_initializer: Const.new(2), bias_initializer: Const.new(2))
    lstm.build([16, 32])
    assert_equal Numo::SFloat.new(32, 256).fill(2), lstm.params[:weight].data
    assert_equal Numo::SFloat.new(64, 256).fill(2), lstm.params[:recurrent_weight].data
    assert_equal Numo::SFloat.new(256).fill(2), lstm.params[:bias].data
    assert_kind_of LSTM_Dense, lstm.instance_variable_get(:@layers)[15]
  end
end


class TestGRU_Dense < MiniTest::Unit::TestCase
  def test_forward
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    w = Param.new
    w.data = Numo::SFloat.new(64, 16 * 3).fill(1)
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16 * 3).fill(1)
    b = Param.new
    b.data = Numo::SFloat.new(16 * 3).fill(0)

    dense = GRU_Dense.new(w, w2, b)
    assert_equal [1, 16], dense.forward(x, h).shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(2, 16).seq[1, false].reshape(1, 16)
    w = Param.new
    w.data = Numo::SFloat.new(64, 16 * 3).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16 * 3).fill(1)
    w2.grad = 0
    b = Param.new
    b.data = Numo::SFloat.new(16 * 3).fill(0)
    b.grad = 0
    dense = GRU_Dense.new(w, w2, b)
    dense.forward(x, h)
    dx, dh = dense.backward(dh2)
    assert_equal [1, 64], dx.shape
    assert_equal [1, 16], dh.shape
  end

  def test_backward2
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(2, 16).seq[1, false].reshape(1, 16)
    w = Param.new
    w.data = Numo::SFloat.new(64, 16 * 3).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16 * 3).fill(1)
    w2.grad = 0
    b = Param.new
    b.data = Numo::SFloat.new(16 * 3).fill(0)
    b.grad = 0

    dense = GRU_Dense.new(w, w2, nil)
    dense.forward(x, h)
    dense.backward(dh2)
    assert_nil dense.instance_variable_get(:@bias)
  end

  def test_backward3
    x = Numo::SFloat.new(1, 64).seq
    h = Numo::SFloat.new(1, 16).seq
    dh2 = Numo::SFloat.new(2, 16).seq[1, false].reshape(1, 16)
    w = Param.new
    w.data = Numo::SFloat.new(64, 16 * 3).fill(1)
    w.grad = 0
    w2 = Param.new
    w2.data = Numo::SFloat.new(16, 16 * 3).fill(1)
    w2.grad = 0
    b = Param.new
    b.data = Numo::SFloat.new(16 * 3).fill(0)
    b.grad = 0
    dense = GRU_Dense.new(w, w2, b)
    dense.trainable = false
    dense.forward(x, h)
    dx, dh = dense.backward(dh2)
    assert_equal 0, dense.instance_variable_get(:@weight).grad
    assert_equal 0, dense.instance_variable_get(:@recurrent_weight).grad
    assert_equal 0, dense.instance_variable_get(:@bias).grad
  end
end


class TestGRU < MiniTest::Unit::TestCase
  
  def test_from_hash
    hash = {
      class: "DNN::Layers::GRU",
      num_nodes: 64,
      weight_initializer: RandomUniform.new.to_hash,
      bias_initializer: RandomUniform.new.to_hash,
      l1_lambda: 0.1,
      l2_lambda: 0.2,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    gru = GRU.from_hash(hash)
    assert_equal 64, gru.num_nodes
    assert_kind_of RandomUniform, gru.weight_initializer
    assert_kind_of RandomUniform, gru.bias_initializer
    assert_equal 0.1, gru.l1_lambda
    assert_equal 0.2, gru.l2_lambda
    assert_equal false, gru.use_bias
    assert_equal true, gru.stateful
    assert_equal false, gru.return_sequences
  end

  def test_to_hash
    gru = GRU.new(64, stateful: true, return_sequences: false,
                        l1_lambda: 0.1, l2_lambda: 0.2, use_bias: false)
    expected_hash = {
      class: "DNN::Layers::GRU",
      num_nodes: 64,
      weight_initializer: gru.weight_initializer.to_hash,
      bias_initializer: gru.bias_initializer.to_hash,
      l1_lambda: 0.1,
      l2_lambda: 0.2,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    assert_equal expected_hash, gru.to_hash
  end

  def test_build
    gru = GRU.new(64, weight_initializer: Const.new(2), bias_initializer: Const.new(2))
    gru.build([16, 32])
    assert_equal Numo::SFloat.new(32, 192).fill(2), gru.params[:weight].data
    assert_equal Numo::SFloat.new(64, 192).fill(2), gru.params[:recurrent_weight].data
    assert_equal Numo::SFloat.new(192).fill(2), gru.params[:bias].data
    assert_kind_of GRU_Dense, gru.instance_variable_get(:@layers)[15]
  end
end
