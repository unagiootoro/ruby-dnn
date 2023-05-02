require "test_helper"
require "json"

include DNN::Layers
include DNN::Optimizers
include DNN::Initializers
include DNN::Losses
include DNN::Models

class StubMultiInputModel < DNN::Models::Model
  def initialize(dense1, dense2)
    super()
    @dense1 = dense1
    @dense2 = dense2
  end

  def forward(x1, x2)
    y1 = @dense1.(x1)
    y2 = @dense2.(x2)
    y1 + y2
  end
end

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

  def test_optimize
    y = DNN::Tensor.new(Xumo::SFloat[[65, 130], [155, 310]])
    t = y
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = Sequential.new
    model << InputLayer.new(3)
    model << dense
    model.setup(SGD.new, MeanSquaredError.new)
    loss = model.optimize(y, t)

    assert_equal 0, Utils.to_f(loss.data)
  end

  # Test multiple outputs.
  def test_optimize2
    y = DNN::Tensor.new(Xumo::SFloat[[65, 130], [155, 310]])
    t = y
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = StubMultiOutputModel.new(dense)
    model.setup(SGD.new, [MeanSquaredError.new, MeanSquaredError.new])
    losses = model.optimize([y, y], [t, t])

    assert_equal [0, 0], losses.map { |loss| Utils.to_f(loss.data)}
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

  # Test multiple inputs.
  def test_train_on_batch2
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Xumo::SFloat[[130, 260], [310, 620]]

    dense1 = Dense.new(2)
    dense1.build([3])
    dense1.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense1.bias.data = Xumo::SFloat[5, 10]

    dense2 = Dense.new(2)
    dense2.build([3])
    dense2.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense2.bias.data = Xumo::SFloat[5, 10]

    model = StubMultiInputModel.new(dense1, dense2)
    model.setup(SGD.new, MeanSquaredError.new)
    loss = model.train_on_batch([x, x], y)

    assert_equal 0, loss
  end

  # Test multiple outputs.
  def test_train_on_batch3
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
    loss = model.test_on_batch(x, y)
    assert_equal 0, loss
  end

  # Test multiple inputs.
  def test_test_on_batch2
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Xumo::SFloat[[130, 260], [310, 620]]

    dense1 = Dense.new(2)
    dense1.build([3])
    dense1.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense1.bias.data = Xumo::SFloat[5, 10]

    dense2 = Dense.new(2)
    dense2.build([3])
    dense2.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense2.bias.data = Xumo::SFloat[5, 10]

    model = StubMultiInputModel.new(dense1, dense2)
    model.setup(SGD.new, MeanSquaredError.new)
    loss = model.test_on_batch([x, x], y)

    assert_equal 0, loss
  end

  # Test multiple outputs.
  def test_test_on_batch3
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    y = Xumo::SFloat[[65, 130], [155, 310]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = StubMultiOutputModel.new(dense)
    model.setup(SGD.new, [MeanSquaredError.new, MeanSquaredError.new])
    losss = model.test_on_batch(x, [y, y])
    assert_equal [0, 0], losss
  end

  # It is matching dense forward result.
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

    assert_equal Xumo::SFloat[[65, 130], [155, 310]], model.predict(x)
  end

  # Test multiple outputs.
  def test_predict2
    x = Xumo::SFloat[[1, 2, 3], [4, 5, 6]]
    expected_y = [Xumo::SFloat[[65, 130], [155, 310]], Xumo::SFloat[[65, 130], [155, 310]]]
    dense = Dense.new(2)
    dense.build([3])
    dense.weight.data = Xumo::SFloat[[10, 20], [10, 20], [10, 20]]
    dense.bias.data = Xumo::SFloat[5, 10]
    model = StubMultiOutputModel.new(dense)
    model.setup(SGD.new, [SigmoidCrossEntropy.new, SigmoidCrossEntropy.new])

    assert_equal expected_y, model.predict(x)
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

  def test_get_layer
    model = Sequential.new
    model << InputLayer.new(2)
    model << Dense.new(1)
    model.setup(SGD.new, MeanSquaredError.new)
    model.predict1(Xumo::SFloat.zeros(2))
    assert_kind_of Dense, model.get_layer(:stack)[1]
  end

  # It is result of load marshal is as expected.
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
      {},
      { weight: dense0.weight.data, bias:  dense0.bias.data},
      { weight: dense1.weight.data, bias:  dense1.bias.data},
    ]
    model2.set_all_params_data(dense_params_data)

    x = Xumo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end

  # It is result of load marshal is as expected.
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
