require "test_helper"
require "json"

include DNN::Layers
include DNN::Optimizers
include DNN::Initializers
include DNN::Losses
include DNN::Models

class StubMultiOutputModel < DNN::Models::Model
  def initialize(dense)
    super()
    @dense = dense
  end

  def forward(x)
    out = @dense.(x)
    [out, out]
  end
end

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
    call_flg = [0, 0, 0, 0, 0, 0, 0, 0]

    before_train_f = -> do
      call_cnt += 1
      call_flg[0] = call_cnt
    end
    after_train_f = -> do
      call_cnt += 1
      call_flg[1] = call_cnt
    end
    before_epoch_f = -> do
      call_cnt += 1
      call_flg[2] = call_cnt
    end
    after_epoch_f = -> do
      call_cnt += 1
      call_flg[3] = call_cnt
    end
    before_train_on_batch_f = -> do
      call_cnt += 1
      call_flg[4] = call_cnt
    end
    after_train_on_batch_f= -> do
      call_cnt += 1
      call_flg[5] = call_cnt
    end
    before_test_on_batch_f = -> do
      call_cnt += 1
      call_flg[6] = call_cnt
    end
    after_test_on_batch_f = -> do
      call_cnt += 1
      call_flg[7] = call_cnt
    end

    before_train_cbk = DNN::Callbacks::LambdaCallback.new(:before_train, &before_train_f)
    after_train_cbk = DNN::Callbacks::LambdaCallback.new(:after_train, &after_train_f)
    before_epoch_cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch, &before_epoch_f)
    after_epoch_cbk = DNN::Callbacks::LambdaCallback.new(:after_epoch, &after_epoch_f)
    before_train_on_batch_cbk = DNN::Callbacks::LambdaCallback.new(:before_train_on_batch, &before_train_on_batch_f)
    after_train_on_batch_cbk = DNN::Callbacks::LambdaCallback.new(:after_train_on_batch, &after_train_on_batch_f)
    before_test_on_batch_cbk = DNN::Callbacks::LambdaCallback.new(:before_test_on_batch, &before_test_on_batch_f)
    after_test_on_batch_cbk = DNN::Callbacks::LambdaCallback.new(:after_test_on_batch, &after_test_on_batch_f)

    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Xumo::SFloat[[65, 130], [155, 310]]
    model = Sequential.new
    model << InputLayer.new(3)
    model << Dense.new(2)
    model.setup(SGD.new, MeanSquaredError.new)
    model.add_callback(before_train_cbk)
    model.add_callback(after_train_cbk)
    model.add_callback(before_epoch_cbk)
    model.add_callback(after_epoch_cbk)
    model.add_callback(before_train_on_batch_cbk)
    model.add_callback(after_train_on_batch_cbk)
    model.add_callback(before_test_on_batch_cbk)
    model.add_callback(after_test_on_batch_cbk)
    model.train(x, y, 1, batch_size: 2, verbose: false, test: [x, y])

    assert_equal [1, 8, 2, 7, 3, 4, 5, 6], call_flg
  end

  def test_train_on_batch
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Xumo::SFloat[[65, 130], [155, 310]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    loss = model.train_on_batch(x, y)

    assert_equal 0, loss
  end

  # Test multiple outputs.
  def test_train_on_batch2
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Xumo::SFloat[[65, 130], [155, 310]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = StubMultiOutputModel.new(dense)
    model.setup(SGD.new, [MeanSquaredError.new, MeanSquaredError.new])
    loss = model.train_on_batch(x, [y, y])

    assert_equal [0, 0], loss
  end

  # It is accuracy is 1.
  def test_evaluate
    model = Sequential.new
    model << InputLayer.new(3)
    model.setup(SGD.new, MeanSquaredError.new)
    x = Xumo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    y = Xumo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    assert_equal 1, model.evaluate(x, y, batch_size: 1).first
  end

  # It is accuracy is 0.5.
  def test_evaluate2
    model = Sequential.new
    model << InputLayer.new(3)
    model.setup(SGD.new, MeanSquaredError.new)
    x = Xumo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    y = Xumo::SFloat[[0, 1, 0.5], [0, 1, 0.5]]
    assert_equal 0.5, model.evaluate(x, y).first
  end

  def test_test_on_batch
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Xumo::SFloat[[65, 130], [155, 310]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    correct, loss = model.test_on_batch(x, y)

    assert_equal 2, correct
    assert_equal 0, loss
  end

  # Test multiple outputs.
  def test_test_on_batch2
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Xumo::SFloat[[65, 130], [155, 310]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = StubMultiOutputModel.new(dense)
    model.setup(SGD.new, [MeanSquaredError.new, MeanSquaredError.new])
    corrects, losss = model.test_on_batch(x, [y, y])

    assert_equal [2, 2], corrects
    assert_equal [0, 0], losss
  end

  # It is matching dense forward result and unuse loss activation.
  def test_predict
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, SigmoidCrossEntropy.new)

    assert_equal Xumo::SFloat[[65, 130], [155, 310]], model.predict(x, use_loss_activation: false)
  end

  # It is matching dense forward result and use loss activation.
  def test_predict2
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, SigmoidCrossEntropy.new)

    assert_equal Xumo::SFloat[[1, 1], [1, 1]], model.predict(x, use_loss_activation: true)
  end

  # Test multiple outputs.
  def test_predict3
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    expected_y = [Xumo::SFloat[[1, 1], [1, 1]], Xumo::SFloat[[1, 1], [1, 1]]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = StubMultiOutputModel.new(dense)
    model.setup(SGD.new, [SigmoidCrossEntropy.new, SigmoidCrossEntropy.new])

    assert_equal expected_y, model.predict(x, use_loss_activation: true)
  end

  # It is matching dense forward result.
  def test_predict1
    x = Xumo::SFloat[1, 2, 3]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)

    assert_equal Xumo::SFloat[65, 130], model.predict1(x)
  end

  # It is including callback function in @callback.
  def test_add_callback
    model = Sequential.new
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch) {}
    model.add_callback(cbk)
    assert_equal [cbk], model.instance_variable_get(:@callbacks)
  end

  # It is including callback function in @callback.
  def test_add_lambda_callback
    model = Sequential.new
    model.add_lambda_callback(:before_epoch) {}
    cbk = model.instance_variable_get(:@callbacks)[0]
    assert_kind_of DNN::Callbacks::LambdaCallback, cbk
  end

  # It is not including callback function in @callback.
  def test_clear_callbacks
    model = Sequential.new
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch) {}
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
    cbk1 = DNN::Callbacks::LambdaCallback.new(:before_epoch, &prc1)
    cbk2 = DNN::Callbacks::LambdaCallback.new(:before_epoch, &prc2)
    model = Sequential.new
    model.add_callback(cbk1)
    model.add_callback(cbk2)
    model.send(:call_callbacks, :before_epoch)
    assert_equal [1, 2], call_flg
  end

  def test_metrics_to_str
    met = { accuracy: 0.00011, test_loss: 0.00011 }
    str_met = "accuracy: 0.0001, test_loss: 0.0001"
    model = DNN::Models::Model.new
    assert_equal str_met, model.send(:metrics_to_str, met)
  end

  def test_metrics_to_str2
    met = { accuracy: [0.00011, 0.00011], test_loss: [0.00011, 0.00011] }
    str_met = "accuracy: [0.0001, 0.0001], test_loss: [0.0001, 0.0001]"
    model = DNN::Models::Model.new
    assert_equal str_met, model.send(:metrics_to_str, met)
  end

  def test_copy
    x = Xumo::SFloat[[0, 0]]
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
    model.predict1(Xumo::SFloat.zeros(2))
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
    model.predict1(Xumo::SFloat.zeros(2))
    assert_equal 1, model.trainable_layers[1].num_units
  end

  def test_get_layer
    model = Sequential.new
    model << InputLayer.new(2)
    model << Dense.new(1)
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Xumo::SFloat.zeros(2))
    assert_kind_of Dense, model.get_layer(:stack)[1]
  end

  # # It is result of load marshal is as expected.
  def test_set_all_params_data
    dense0 = DNN::Layers::Dense.new(5)
    dense1 = DNN::Layers::Dense.new(1)
    model = DNN::Models::Sequential.new([InputLayer.new(10), dense0, dense1])
    model.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model.predict1(Xumo::SFloat.zeros(10))
    model2 = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(5), Dense.new(1)])
    model2.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model2.predict1(Xumo::SFloat.zeros(10))

    dense_params_data = [
      { weight: dense0.weight.data, bias:  dense0.bias.data},
      { weight: dense1.weight.data, bias:  dense1.bias.data},
    ]
    model2.set_all_params_data(dense_params_data)

    x = Xumo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end

  # # It is result of load marshal is as expected.
  def test_get_all_params_data
    model = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(5), Dense.new(1)])
    model.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model.predict1(Xumo::SFloat.zeros(10))
    model2 = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(5), Dense.new(1)])
    model2.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model2.predict1(Xumo::SFloat.zeros(10))

    params_data = model.get_all_params_data
    model2.set_all_params_data(params_data)

    x = Xumo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end

  def test_to_cpu
    x = Xumo::SFloat[[0, 0]]
    model = Sequential.new
    model << InputLayer.new(2)
    model << Dense.new(1)
    model.setup(SGD.new, MeanSquaredError.new)
    y = model.predict(x)
    model.to_cpu
    assert_equal y, model.predict(x)
  end

  def test_add
    model = Sequential.new
    input_layer = InputLayer.new(10)
    model.add(input_layer)
    model2 = Sequential.new
    model2.add(Dense.new(10))
    model.add(model2)
    model.predict1(Xumo::SFloat.zeros(10))
    assert_kind_of InputLayer, model.layers[0]
    assert_kind_of Dense, model.layers[1]
  end

  def test_add_ng
    model = Sequential.new
    assert_raises TypeError do
      model.add(SGD.new)
    end
  end

  def test_insert
    model = Sequential.new
    input_layer = InputLayer.new(10)
    model.add(input_layer)
    model.add(Dense.new(10))
    model.insert(1, Dense.new(20))
    model.predict1(Xumo::SFloat.zeros(10))
    assert_equal 20, model.layers[1].num_units
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
