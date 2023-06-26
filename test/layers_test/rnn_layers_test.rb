require "test_helper"

include DNN::Layers

class TestRNN < MiniTest::Unit::TestCase
  def test_initialize
    rnn = RNN.new(64, stateful: true, return_sequences: false,
                  weight_initializer: DNN::Initializers::RandomUniform.new,
                  recurrent_weight_initializer: DNN::Initializers::RandomUniform.new,
                  bias_initializer: DNN::Initializers::RandomUniform.new,
                  weight_regularizer: DNN::Regularizers::L1.new,
                  recurrent_weight_regularizer: DNN::Regularizers::L2.new,
                  bias_regularizer: DNN::Regularizers::L1L2.new)
    assert_equal 64, rnn.num_units
    assert_equal true, rnn.stateful
    assert_equal false, rnn.return_sequences
  end

  def test_to_hash
    rnn = RNN.new(64, stateful: true, return_sequences: false, use_bias: false,
                  weight_regularizer: DNN::Regularizers::L1.new,
                  recurrent_weight_regularizer: DNN::Regularizers::L2.new,
                  bias_regularizer: DNN::Regularizers::L1L2.new)
    expected_hash = {
      class: "DNN::Layers::RNN",
      num_units: 64,
      weight_initializer: rnn.weight_initializer.to_hash,
      recurrent_weight_initializer: rnn.recurrent_weight_initializer.to_hash,
      bias_initializer: rnn.bias_initializer.to_hash,
      weight_regularizer: rnn.weight_regularizer.to_hash,
      recurrent_weight_regularizer: rnn.recurrent_weight_regularizer.to_hash,
      bias_regularizer: rnn.bias_regularizer.to_hash,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    assert_equal expected_hash, rnn.to_hash
  end

  def test_regularizers
    rnn = RNN.new(1, weight_regularizer: DNN::Regularizers::L1.new,
                       recurrent_weight_regularizer: DNN::Regularizers::L2.new,
                       bias_regularizer: DNN::Regularizers::L1L2.new)
    rnn.build([1, 10])
    assert_kind_of DNN::Regularizers::L1, rnn.regularizers[0]
    assert_kind_of DNN::Regularizers::L2, rnn.regularizers[1]
    assert_kind_of DNN::Regularizers::L1L2, rnn.regularizers[2]
  end

  def test_regularizers2
    rnn = RNN.new(1)
    rnn.build([1, 10])
    assert_equal [], rnn.regularizers
  end
end


class TestSimpleRNNCell < MiniTest::Unit::TestCase
  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 64).seq)
    h = DNN::Tensor.new(Xumo::SFloat.new(1, 16).seq)
    w = DNN::Variable.new
    w.data = Xumo::SFloat.new(64, 16).fill(1)
    w2 = DNN::Variable.new
    w2.data = Xumo::SFloat.new(16, 16).fill(1)
    b = DNN::Variable.new
    b.data = Xumo::SFloat.new(16).fill(0)

    cell = SimpleRNNCell.new
    assert_equal [1, 16], cell.forward(x, h, w, w2, b).shape
  end
end


class TestSimpleRNN < MiniTest::Unit::TestCase
  def test_reset_state
    rnn = SimpleRNN.new(64)
    rnn.build([16, 64])
    h = DNN::Variable.new(Xumo::SFloat.ones(16, 64))
    rnn.instance_variable_set(:@h, h)
    rnn.reset_state
    assert_equal Xumo::SFloat.zeros(16, 64), rnn.instance_variable_get(:@h).data
  end

  def test_from_hash
    hash = {
      class: "DNN::Layers::SimpleRNN",
      num_units: 64,
      weight_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      recurrent_weight_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      bias_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      weight_regularizer: DNN::Regularizers::L1.new.to_hash,
      recurrent_weight_regularizer: DNN::Regularizers::L2.new.to_hash,
      bias_regularizer: DNN::Regularizers::L1L2.new.to_hash,
      use_bias: false,
      stateful: true,
      return_sequences: false,
      activation: :ReLU,
    }
    rnn = SimpleRNN.from_hash(hash)
    assert_equal 64, rnn.num_units
    assert_kind_of DNN::Initializers::RandomUniform, rnn.weight_initializer
    assert_kind_of DNN::Initializers::RandomUniform, rnn.recurrent_weight_initializer
    assert_kind_of DNN::Initializers::RandomUniform, rnn.bias_initializer
    assert_kind_of DNN::Regularizers::L1, rnn.weight_regularizer
    assert_kind_of DNN::Regularizers::L2, rnn.recurrent_weight_regularizer
    assert_kind_of DNN::Regularizers::L1L2, rnn.bias_regularizer
    assert_equal false, rnn.use_bias
    assert_equal true, rnn.stateful
    assert_equal false, rnn.return_sequences
    assert_equal :ReLU, rnn.activation
  end

  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 16, 64).seq)
    rnn = SimpleRNN.new(64)
    rnn.build([16, 64])
    assert_equal [1, 16, 64], rnn.forward(x).shape
  end

  def test_forward2
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 16, 64).seq)
    rnn = SimpleRNN.new(64, stateful: true)
    rnn.build([16, 64])
    rnn.forward(x)
    assert_equal [1, 16, 64], rnn.forward(x).shape
  end

  def test_to_hash
    rnn = SimpleRNN.new(64, stateful: true, return_sequences: false, use_bias: false, activation: :ReLU,
                        weight_regularizer: DNN::Regularizers::L1.new,
                        recurrent_weight_regularizer: DNN::Regularizers::L2.new,
                        bias_regularizer: DNN::Regularizers::L1L2.new)
    expected_hash = {
      class: "DNN::Layers::SimpleRNN",
      num_units: 64,
      weight_initializer: rnn.weight_initializer.to_hash,
      recurrent_weight_initializer: rnn.recurrent_weight_initializer.to_hash,
      bias_initializer: rnn.bias_initializer.to_hash,
      weight_regularizer: rnn.weight_regularizer.to_hash,
      recurrent_weight_regularizer: rnn.recurrent_weight_regularizer.to_hash,
      bias_regularizer: rnn.bias_regularizer.to_hash,
      use_bias: false,
      stateful: true,
      return_sequences: false,
      activation: :ReLU,
    }
    assert_equal expected_hash, rnn.to_hash
  end

  def test_build
    rnn = SimpleRNN.new(64, weight_initializer: DNN::Initializers::Const.new(2),
                            recurrent_weight_initializer: DNN::Initializers::Const.new(2),
                            bias_initializer: DNN::Initializers::Const.new(2))
    rnn.build([16, 32])
    assert_equal Xumo::SFloat.new(32, 64).fill(2), rnn.weight.data
    assert_equal Xumo::SFloat.new(64, 64).fill(2), rnn.recurrent_weight.data
    assert_equal Xumo::SFloat.new(64).fill(2), rnn.bias.data
  end

  def test_get_variables
    rnn = SimpleRNN.new(1)
    rnn.build([1, 10])
    expected_hash = {
      weight: rnn.weight,
      recurrent_weight: rnn.recurrent_weight,
      bias: rnn.bias,
      h: rnn.instance_variable_get(:@h),
    }
    assert_equal expected_hash, rnn.get_variables
  end
end


class TestLSTMCell < MiniTest::Unit::TestCase
  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 64).seq)
    h = DNN::Tensor.new(Xumo::SFloat.new(1, 16).seq)
    c = DNN::Tensor.new(Xumo::SFloat.new(1, 16).seq)
    w = DNN::Variable.new
    w.data = Xumo::SFloat.new(64, 16 * 4).fill(1)
    w2 = DNN::Variable.new
    w2.data = Xumo::SFloat.new(16, 16 * 4).fill(1)
    b = DNN::Variable.new
    b.data = Xumo::SFloat.new(16 * 4).fill(0)

    cell = LSTMCell.new
    h2, c2 = cell.forward(x, h, c, w, w2, b)
    assert_equal [1, 16], h2.shape
    assert_equal [1, 16], c2.shape
  end
end


class TestLSTM < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::LSTM",
      num_units: 64,
      weight_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      recurrent_weight_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      bias_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      weight_regularizer: DNN::Regularizers::L1.new.to_hash,
      recurrent_weight_regularizer: DNN::Regularizers::L2.new.to_hash,
      bias_regularizer: DNN::Regularizers::L1L2.new.to_hash,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    lstm = LSTM.from_hash(hash)
    assert_equal 64, lstm.num_units
    assert_kind_of DNN::Initializers::RandomUniform, lstm.weight_initializer
    assert_kind_of DNN::Initializers::RandomUniform, lstm.recurrent_weight_initializer
    assert_kind_of DNN::Initializers::RandomUniform, lstm.bias_initializer
    assert_kind_of DNN::Regularizers::L1, lstm.weight_regularizer
    assert_kind_of DNN::Regularizers::L2, lstm.recurrent_weight_regularizer
    assert_kind_of DNN::Regularizers::L1L2, lstm.bias_regularizer
    assert_equal false, lstm.use_bias
    assert_equal true, lstm.stateful
    assert_equal false, lstm.return_sequences
  end

  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 16, 64).seq)
    lstm = LSTM.new(64)
    lstm.build([16, 64])
    assert_equal [1, 16, 64], lstm.forward(x).shape
  end

  def test_forward2
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 16, 64).seq)
    lstm = LSTM.new(64, stateful: true)
    lstm.build([16, 64])
    lstm.forward(x)
    assert_equal [1, 16, 64], lstm.forward(x).shape
  end

  def test_reset_state
    lstm = LSTM.new(64)
    lstm.build([16, 64])
    h = DNN::Variable.new(Xumo::SFloat.ones(16, 64))
    lstm.instance_variable_set(:@h, h)
    c = DNN::Variable.new(Xumo::SFloat.ones(16, 64))
    lstm.instance_variable_set(:@c, c)
    lstm.reset_state
    assert_equal Xumo::SFloat.zeros(16, 64), lstm.instance_variable_get(:@h).data
    assert_equal Xumo::SFloat.zeros(16, 64), lstm.instance_variable_get(:@c).data
  end

  def test_build
    lstm = LSTM.new(64, weight_initializer: DNN::Initializers::Const.new(2),
                    recurrent_weight_initializer: DNN::Initializers::Const.new(2),
                    bias_initializer: DNN::Initializers::Const.new(2))
    lstm.build([16, 32])
    assert_equal Xumo::SFloat.new(32, 256).fill(2), lstm.weight.data
    assert_equal Xumo::SFloat.new(64, 256).fill(2), lstm.recurrent_weight.data
    assert_equal Xumo::SFloat.new(256).fill(2), lstm.bias.data
  end

  def test_to_hash
    lstm = LSTM.new(64, stateful: true, return_sequences: false, use_bias: false,
                    weight_regularizer: DNN::Regularizers::L1.new,
                    recurrent_weight_regularizer: DNN::Regularizers::L2.new,
                    bias_regularizer: DNN::Regularizers::L1L2.new)
    expected_hash = {
      class: "DNN::Layers::LSTM",
      num_units: 64,
      weight_initializer: lstm.weight_initializer.to_hash,
      recurrent_weight_initializer: lstm.recurrent_weight_initializer.to_hash,
      bias_initializer: lstm.bias_initializer.to_hash,
      weight_regularizer: lstm.weight_regularizer.to_hash,
      recurrent_weight_regularizer: lstm.recurrent_weight_regularizer.to_hash,
      bias_regularizer: lstm.bias_regularizer.to_hash,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    assert_equal expected_hash, lstm.to_hash
  end

  def test_get_params
    lstm = LSTM.new(1)
    lstm.build([1, 10])
    expected_hash = {
      weight: lstm.weight,
      recurrent_weight: lstm.recurrent_weight,
      bias: lstm.bias,
      h: lstm.instance_variable_get(:@h),
      c: lstm.instance_variable_get(:@c),
    }
    assert_equal expected_hash, lstm.get_variables
  end
end


class TestGRUCell < MiniTest::Unit::TestCase
  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 64).seq)
    h = DNN::Tensor.new(Xumo::SFloat.new(1, 16).seq)
    w = DNN::Variable.new
    w.data = Xumo::SFloat.new(64, 16 * 3).fill(1)
    w2 = DNN::Variable.new
    w2.data = Xumo::SFloat.new(16, 16 * 3).fill(1)
    b = DNN::Variable.new
    b.data = Xumo::SFloat.new(16 * 3).fill(0)

    cell = GRUCell.new
    assert_equal [1, 16], cell.forward(x, h, w, w2, b).shape
  end
end


class TestGRU < MiniTest::Unit::TestCase
  def test_reset_state
    gru = GRU.new(64)
    gru.build([16, 64])
    h = DNN::Variable.new(Xumo::SFloat.ones(16, 64))
    gru.instance_variable_set(:@h, h)
    gru.reset_state
    assert_equal Xumo::SFloat.zeros(16, 64), gru.instance_variable_get(:@h).data
  end

  def test_from_hash
    hash = {
      class: "DNN::Layers::GRU",
      num_units: 64,
      weight_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      recurrent_weight_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      bias_initializer: DNN::Initializers::RandomUniform.new.to_hash,
      weight_regularizer: DNN::Regularizers::L1.new.to_hash,
      recurrent_weight_regularizer: DNN::Regularizers::L2.new.to_hash,
      bias_regularizer: DNN::Regularizers::L1L2.new.to_hash,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    gru = GRU.from_hash(hash)
    assert_equal 64, gru.num_units
    assert_kind_of DNN::Initializers::RandomUniform, gru.weight_initializer
    assert_kind_of DNN::Initializers::RandomUniform, gru.recurrent_weight_initializer
    assert_kind_of DNN::Initializers::RandomUniform, gru.bias_initializer
    assert_kind_of DNN::Regularizers::L1, gru.weight_regularizer
    assert_kind_of DNN::Regularizers::L2, gru.recurrent_weight_regularizer
    assert_kind_of DNN::Regularizers::L1L2, gru.bias_regularizer
    assert_equal false, gru.use_bias
    assert_equal true, gru.stateful
    assert_equal false, gru.return_sequences
  end

  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 16, 64).seq)
    gru = GRU.new(64)
    gru.build([16, 64])
    assert_equal [1, 16, 64], gru.forward(x).shape
  end

  def test_forward2
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 16, 64).seq)
    gru = GRU.new(64, stateful: true)
    gru.build([16, 64])
    gru.forward(x)
    assert_equal [1, 16, 64], gru.forward(x).shape
  end

  def test_to_hash
    gru = GRU.new(64, stateful: true, return_sequences: false, use_bias: false,
                  weight_regularizer: DNN::Regularizers::L1.new,
                  recurrent_weight_regularizer: DNN::Regularizers::L2.new,
                  bias_regularizer: DNN::Regularizers::L1L2.new)
    expected_hash = {
      class: "DNN::Layers::GRU",
      num_units: 64,
      weight_initializer: gru.weight_initializer.to_hash,
      recurrent_weight_initializer: gru.recurrent_weight_initializer.to_hash,
      bias_initializer: gru.bias_initializer.to_hash,
      weight_regularizer: gru.weight_regularizer.to_hash,
      recurrent_weight_regularizer: gru.recurrent_weight_regularizer.to_hash,
      bias_regularizer: gru.bias_regularizer.to_hash,
      use_bias: false,
      stateful: true,
      return_sequences: false,
    }
    assert_equal expected_hash, gru.to_hash
  end

  def test_build
    gru = GRU.new(64, weight_initializer: DNN::Initializers::Const.new(2),
                  recurrent_weight_initializer: DNN::Initializers::Const.new(2),
                  bias_initializer: DNN::Initializers::Const.new(2))
    gru.build([16, 32])
    assert_equal Xumo::SFloat.new(32, 192).fill(2), gru.weight.data
    assert_equal Xumo::SFloat.new(64, 192).fill(2), gru.recurrent_weight.data
    assert_equal Xumo::SFloat.new(192).fill(2), gru.bias.data
  end

  def test_get_variables
    gru = GRU.new(1)
    gru.build([1, 10])
    expected_hash = {
      weight: gru.weight,
      recurrent_weight: gru.recurrent_weight,
      bias: gru.bias,
      h: gru.instance_variable_get(:@h),
    }
    assert_equal expected_hash, gru.get_variables
  end
end
