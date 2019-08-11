require "test_helper"
require "json"

include DNN::Activations
include DNN::Layers
include DNN::Optimizers
include DNN::Initializers
include DNN::Losses
include DNN::Models


class TestSequential < MiniTest::Unit::TestCase
  def test_load_hash_params
    model = Sequential.new
    model << InputLayer.new([10])
    dense = Dense.new(10)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(10))
    hash = model.params_to_hash
    model.load_hash_params(hash)
    model.predict1(Numo::SFloat.zeros(10))
    assert_equal dense.get_params[:weight].data, model.layers[1].get_params[:weight].data
  end

  def test_load_json_params
    model = Sequential.new
    model << InputLayer.new([10])
    dense = Dense.new(10)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(10))
    json = model.params_to_json
    model.load_json_params(json)
    model.predict1(Numo::SFloat.zeros(10))
    assert_equal dense.get_params[:weight].data, model.layers[1].get_params[:weight].data
  end

  def test_params_to_json
    model = Sequential.new
    model << InputLayer.new([10])
    dense = Dense.new(10)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(10))
    json = model.params_to_json
    param = JSON.parse(json)["params"][0]["weight"]
    bin = Base64.decode64(param[1])
    data = Xumo::SFloat.from_binary(bin).reshape(*param[0])

    assert_equal dense.get_params[:weight].data, data
  end

  def test_params_to_hash
    model = Sequential.new
    model << InputLayer.new([10])
    dense = Dense.new(10)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(10))
    hash = model.params_to_hash
    param = hash[:params][0][:weight]
    bin = param[1]
    data = Xumo::SFloat.from_binary(bin).reshape(*param[0])

    assert_equal dense.get_params[:weight].data, data
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

  def test_setup_completed?
    model = Sequential.new
    model << InputLayer.new(10)
    model << Dense.new(10)
    model.setup(SGD.new, MeanSquaredError.new)
    assert_equal true, model.setup_completed?
  end

  def setup_completed2?
    model = Sequential.new
    model << InputLayer.new(10)
    model << Dense.new(10)
    assert_equal false, model.setup_completed?
  end

  def test_train
    call_cnt = 0
    call_flg = [0, 0, 0, 0, 0, 0]
    before_epoch_cbk = -> epoch do
      call_cnt += 1
      call_flg[0] = call_cnt
    end
    after_epoch_cbk = -> epoch do
      call_cnt += 1
      call_flg[1] = call_cnt
    end
    before_train_on_batch_cbk = -> do
      call_cnt += 1
      call_flg[2] = call_cnt
    end
    after_train_on_batch_cbk= -> loss do
      call_cnt += 1
      call_flg[3] = call_cnt
    end
    before_test_on_batch_cbk = -> do
      call_cnt += 1
      call_flg[4] = call_cnt
    end
    after_test_on_batch_cbk= -> loss do
      call_cnt += 1
      call_flg[5] = call_cnt
    end

    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Numo::SFloat[[65, 130], [155, 310]]
    model = Sequential.new
    model << InputLayer.new(3)
    model << Dense.new(2)
    model.setup(SGD.new, MeanSquaredError.new)
    model.train(x, y, 1, batch_size: 2, verbose: false, test: [x, y],
                before_epoch_cbk: before_epoch_cbk, after_epoch_cbk: after_epoch_cbk,
                before_train_on_batch_cbk: before_train_on_batch_cbk, after_train_on_batch_cbk: after_train_on_batch_cbk,
                before_test_on_batch_cbk: before_test_on_batch_cbk, after_test_on_batch_cbk: after_test_on_batch_cbk)

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

  def test_accurate
    model = Sequential.new
    model << InputLayer.new(3)
    model.setup(SGD.new, MeanSquaredError.new)
    x = Numo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    y = Numo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    assert_equal 1, model.accurate(x, y, batch_size: 1).first
  end

  def test_accurate2
    model = Sequential.new
    model << InputLayer.new(3)
    model.setup(SGD.new, MeanSquaredError.new)
    x = Numo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    y = Numo::SFloat[[0, 1, 0.5], [0, 1, 0.5]]
    assert_equal 0.5, model.accurate(x, y).first
  end

  def test_test_on_batch
    call_cnt = 0
    call_flg = [0, 0]
    before = -> do
      call_cnt += 1
      call_flg[0] = call_cnt
    end
    after = -> loss do
      call_cnt += 1
      call_flg[1] = call_cnt
    end

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
    correct, loss = model.test_on_batch(x, y, before_test_on_batch_cbk: before, after_test_on_batch_cbk: after)

    assert_equal 2, correct
    assert_equal 0, loss
    assert_equal [1, 2], call_flg
  end

  def test_predict
    x = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Numo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Numo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)

    assert_equal Numo::SFloat[[65, 130], [155, 310]], model.predict(x)
  end

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

  def test_layers_ng
    model = Sequential.new
    model << InputLayer.new(2)
    model << Dense.new(8)
    assert_raises DNN::DNN_Error do
      model.layers
    end
  end

  def test_has_param_layers
    sequential = Sequential.new
    sequential << Dense.new(8)
    sequential << Dense.new(1)
    model = Sequential.new
    model << InputLayer.new(2)
    model << sequential
    model.predict1(Numo::SFloat.zeros(2))
    assert_equal 1, model.has_param_layers[1].num_nodes
  end

  def test_get_layer
    model = Sequential.new
    model << InputLayer.new(2)
    model << Dense.new(1)
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(2))
    assert_kind_of Dense, model.get_layer(1)
  end

  def test_get_layer2
    model = Sequential.new
    model << InputLayer.new(2)
    model << Dense.new(8)
    model << Dense.new(1)
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(2))
    assert_equal 1, model.get_layer(Dense, 1).num_nodes
  end

  def test_optimizer
    model = Sequential.new
    model << InputLayer.new(10)
    model.setup(SGD.new, MeanSquaredError.new)
    assert_kind_of SGD, model.optimizer
  end

  def test_optimizer_ng
    model = Sequential.new
    model << InputLayer.new(10)
    assert_raises DNN::DNN_Error do
      model.optimizer
    end
  end

  def test_loss_func
    model = Sequential.new
    model << InputLayer.new(10)
    model.setup(SGD.new, MeanSquaredError.new)
    assert_kind_of MeanSquaredError, model.loss_func
  end

  def test_loss_func_ng
    model = Sequential.new
    model << InputLayer.new(10)
    assert_raises DNN::DNN_Error do
      model.loss_func
    end
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
end
