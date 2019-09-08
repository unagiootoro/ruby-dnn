require "test_helper"

class TestLoader < MiniTest::Unit::TestCase
  # It is result of load marshal is as expected.
  def test_hash_to_params
    dense = DNN::Layers::Dense.new(1)
    model = DNN::Models::Sequential.new([InputLayer.new(10), dense])
    model.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(10))
    model2 = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])
    model2.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model2.predict1(Numo::SFloat.zeros(10))

    loader = DNN::Loaders::Loader.new(model2)
    dense_params = { weight: [[10, 1], dense.weight.data.to_binary], bias: [[1], dense.bias.data.to_binary] }
    hash = { params: [dense_params] }
    loader.send(:hash_to_params, hash)

    x = Numo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end
end


class TestSaver < MiniTest::Unit::TestCase
  # It is result of load marshal is as expected.
  def test_hash_to_params
    model = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])
    model.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(10))
    model2 = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])
    model2.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model2.predict1(Numo::SFloat.zeros(10))

    saver = DNN::Savers::Saver.new(model)
    hash = saver.send(:params_to_hash)
    loader = DNN::Loaders::Loader.new(model2)
    loader.send(:hash_to_params, hash)

    x = Numo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end
end


class TestMarshalSaver < MiniTest::Unit::TestCase
  # It is result of load marshal is as expected.
  def test_dump_bin
    model = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])
    model.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(10))
    model2 = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])

    saver = DNN::Savers::MarshalSaver.new(model, include_optimizer: false)
    bin = saver.send(:dump_bin)
    loader = DNN::Loaders::MarshalLoader.new(model2)
    loader.send(:load_bin, bin)

    x = Numo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end

  # It is can continue training.
  def test_dump_bin2
    x = Numo::SFloat.new(1, 10).rand
    y = Numo::SFloat.new(1, 1).rand
    model = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])
    model.setup(DNN::Optimizers::SGD.new(momentum: 0.9), DNN::Losses::MeanSquaredError.new)
    model.train_on_batch(x, y)
    model2 = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])

    saver = DNN::Savers::MarshalSaver.new(model, include_optimizer: true)
    bin = saver.send(:dump_bin)
    loader = DNN::Loaders::MarshalLoader.new(model2)
    loader.send(:load_bin, bin)
    model.train_on_batch(x, y)
    model2.train_on_batch(x, y)

    assert_equal model.predict(x), model2.predict(x)
  end
end


class TestJSONSaver < MiniTest::Unit::TestCase
  # It is result of load marshal is as expected.
  def test_dump_bin
    model = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])
    model.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model.predict1(Numo::SFloat.zeros(10))
    model2 = DNN::Models::Sequential.new([InputLayer.new(10), Dense.new(1)])
    model2.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model2.predict1(Numo::SFloat.zeros(10))

    saver = DNN::Savers::JSONSaver.new(model)
    bin = saver.send(:dump_bin)
    loader = DNN::Loaders::JSONLoader.new(model2)
    loader.send(:load_bin, bin)

    x = Numo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end
end
