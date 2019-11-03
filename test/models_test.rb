require "test_helper"
require "json"

include DNN::Layers
include DNN::Optimizers
include DNN::Initializers
include DNN::Losses
include DNN::Models


class TestSequential < MiniTest::Unit::TestCase
  def test_initialize
    model = Sequential.new([InputLayer.new([10]), Dense.new(10)])
    assert_kind_of Dense, model.instance_variable_get(:@stack)[1]
  end

  def test_setup
    model = Sequential.new
    model << InputLayer.new(2)
    dense = Dense.new(1)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    assert_kind_of SGD, model.optimizer
  end

  def test_setup_ng
    model = Sequential.new
    model << InputLayer.new(10)
    model << Dense.new(10)
    assert_raises TypeError do
      model.setup(false, MeanSquaredError.new)
    end
    assert_raises TypeError do
      model.setup(SGD.new, false)
    end
  end

  def test_train
    call_cnt = 0
    call_flg = [0, 0, 0, 0, 0, 0]

    before_epoch_f = -> do
      call_cnt += 1
      call_flg[0] = call_cnt
    end
    after_epoch_f = -> do
      call_cnt += 1
      call_flg[1] = call_cnt
    end
    before_train_on_batch_f = -> do
      call_cnt += 1
      call_flg[2] = call_cnt
    end
    after_train_on_batch_f= -> do
      call_cnt += 1
      call_flg[3] = call_cnt
    end
    before_test_on_batch_f = -> do
      call_cnt += 1
      call_flg[4] = call_cnt
    end
    after_test_on_batch_f = -> do
      call_cnt += 1
      call_flg[5] = call_cnt
    end

    before_epoch_cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch, before_epoch_f)
    after_epoch_cbk = DNN::Callbacks::LambdaCallback.new(:after_epoch, after_epoch_f)
    before_train_on_batch_cbk = DNN::Callbacks::LambdaCallback.new(:before_train_on_batch, before_train_on_batch_f)
    after_train_on_batch_cbk = DNN::Callbacks::LambdaCallback.new(:after_train_on_batch, after_train_on_batch_f)
    before_test_on_batch_cbk = DNN::Callbacks::LambdaCallback.new(:before_test_on_batch, before_test_on_batch_f)
    after_test_on_batch_cbk = DNN::Callbacks::LambdaCallback.new(:after_test_on_batch, after_test_on_batch_f)

    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Numo::SFloat[[65, 130], [155, 310]]
    model = Sequential.new
    model << InputLayer.new(3)
    model << Dense.new(2)
    model.setup(SGD.new, MeanSquaredError.new)
    model.add_callback(before_epoch_cbk)
    model.add_callback(after_epoch_cbk)
    model.add_callback(before_train_on_batch_cbk)
    model.add_callback(after_train_on_batch_cbk)
    model.add_callback(before_test_on_batch_cbk)
    model.add_callback(after_test_on_batch_cbk)
    model.train(x, y, 1, batch_size: 2, verbose: false, test: [x, y])

    assert_equal [1, 6, 2, 3, 4, 5], call_flg
  end

  def test_train_on_batch
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Numo::SFloat[[65, 130], [155, 310]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Numo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    loss = model.train_on_batch(x, y)

    assert_equal 0, loss
  end

  # It is accuracy is 1.
  def test_accuracy
    model = Sequential.new
    model << InputLayer.new(3)
    model.setup(SGD.new, MeanSquaredError.new)
    x = Numo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    y = Numo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    assert_equal 1, model.accuracy(x, y, batch_size: 1).first
  end

  # It is accuracy is 0.5.
  def test_accuracy2
    model = Sequential.new
    model << InputLayer.new(3)
    model.setup(SGD.new, MeanSquaredError.new)
    x = Numo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    y = Numo::SFloat[[0, 1, 0.5], [0, 1, 0.5]]
    assert_equal 0.5, model.accuracy(x, y).first
  end

  def test_test_on_batch
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Numo::SFloat[[65, 130], [155, 310]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Numo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    correct, loss = model.test_on_batch(x, y)

    assert_equal 2, correct
    assert_equal 0, loss
  end

  # It is matching dense forward result and unuse loss activation.
  def test_predict
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Numo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, SigmoidCrossEntropy.new)

    assert_equal Numo::SFloat[[65, 130], [155, 310]], model.predict(x, use_loss_activation: false)
  end

  # It is matching dense forward result and use loss activation.
  def test_predict2
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Numo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, SigmoidCrossEntropy.new)

    assert_equal Numo::SFloat[[1, 1], [1, 1]], model.predict(x, use_loss_activation: true)
  end

  # It is matching dense forward result.
  def test_predict1
    x = Numo::SFloat[1, 2, 3]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Numo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)

    assert_equal Numo::SFloat[65, 130], model.predict1(x)
  end

  # It is including callback function in @callback.
  def test_add_callback
    model = Sequential.new
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch, proc {})
    model.add_callback(cbk)
    assert_equal [cbk], model.instance_variable_get(:@callbacks)
  end

  # It is not including callback function in @callback.
  def test_clear_callbacks
    model = Sequential.new
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch, proc {})
    model.add_callback(cbk)
    model.clear_callbacks
    assert_equal [], model.instance_variable_get(:@callbacks)
  end

  # It is running all callback function.
  def test_call_callbacks
    call_cnt = 0
    call_flg = [0, 0]
    prc1 = proc do
      call_cnt += 1
      call_flg[0] = call_cnt
    end
    prc2 = proc do
      call_cnt += 1
      call_flg[1] = call_cnt
    end
    cbk1 = DNN::Callbacks::LambdaCallback.new(:before_epoch, prc1)
    cbk2 = DNN::Callbacks::LambdaCallback.new(:before_epoch, prc2)
    model = Sequential.new
    model.add_callback(cbk1)
    model.add_callback(cbk2)
    model.send(:call_callbacks, :before_epoch)
    assert_equal [1, 2], call_flg
  end

  def test_copy
    x = Numo::SFloat[[0, 0]]
    model = Sequential.new
    model << InputLayer.new(2)
    model << Dense.new(1)
    model.setup(SGD.new, MeanSquaredError.new)
    model2 = model.copy
    assert_equal model.predict(x), model2.predict(x)
  end

  def test_layers
    sequential = Sequential.new
    sequential << Dense.new(8)
    sequential << Dense.new(1)
    model = Sequential.new
    model << InputLayer.new(2)
    model << sequential
    model.predict1(Numo::SFloat.zeros(2))
    assert_kind_of InputLayer, model.layers.first
    assert_kind_of Dense, model.layers.last
  end

  def test_trainable_layers
    sequential = Sequential.new
    sequential << Dense.new(8)
    sequential << Dense.new(1)
    model = Sequential.new
    model << InputLayer.new(2)
    model << sequential
    model.predict1(Numo::SFloat.zeros(2))
    assert_equal 1, model.trainable_layers[1].num_nodes
  end

  def test_get_layer
    model = Sequential.new
    model << InputLayer.new(2)
    model << Dense.new(1)
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(2))
    assert_kind_of Dense, model.get_layer(:Dense_0)
  end

  def test_naming
    dense1 = Dense.new(1)
    model = Sequential.new
    model << InputLayer.new(10)
    model << Dense.new(5)
    model << dense1
    model.predict1(Numo::SFloat.zeros(10))
    
    assert_equal :Dense_1, dense1.name
    assert_equal :Dense_1__bias, dense1.bias.name
  end

  def test_lshift
    model = Sequential.new
    input_layer = InputLayer.new(10)
    model << input_layer
    model2 = Sequential.new
    model2 << Dense.new(10)
    model << model2
    model.predict1(Numo::SFloat.zeros(10))
    assert_kind_of InputLayer, model.layers[0]
    assert_kind_of Dense, model.layers[1]
  end

  def test_lshift_ng
    model = Sequential.new
    assert_raises TypeError do
      model << SGD.new
    end
  end

  # It is matching [].
  def test_remove
    model = Sequential.new
    input_layer = InputLayer.new(10)
    model << input_layer
    model2 = Sequential.new
    model2 << Dense.new(10)
    model << model2
    model.remove(input_layer)
    model.remove(model2)
    assert_equal [], model.stack
  end
end
