require "test_helper"

class TestMarshalSaver < MiniTest::Unit::TestCase
  # It is result of load marshal is as expected.
  def test_dump_bin
    model = DNN::Models::Sequential.new([DNN::Layers::InputLayer.new(10), DNN::Layers::Dense.new(1)])
    model.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model.predict1(Xumo::SFloat.zeros(10))
    model2 = DNN::Models::Sequential.new([DNN::Layers::InputLayer.new(10), DNN::Layers::Dense.new(1)])
    model2.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model2.predict1(Xumo::SFloat.zeros(10))

    saver = DNN::Savers::MarshalSaver.new(model, include_model: false)
    bin = saver.send(:dump_bin)
    loader = DNN::Loaders::MarshalLoader.new(model2)
    loader.send(:load_bin, bin)

    x = Xumo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end

  # It is can continue training.
  def test_dump_bin2
    x = Xumo::SFloat.new(1, 10).rand
    y = Xumo::SFloat.new(1, 1).rand
    model = DNN::Models::Sequential.new([DNN::Layers::InputLayer.new(10), DNN::Layers::Dense.new(1)])
    model.setup(DNN::Optimizers::SGD.new(momentum: 0.9), DNN::Losses::MeanSquaredError.new)
    model.train_on_batch(x, y)
    model2 = DNN::Models::Sequential.new

    saver = DNN::Savers::MarshalSaver.new(model, include_model: true)
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
    model = DNN::Models::Sequential.new([DNN::Layers::InputLayer.new(10), DNN::Layers::Dense.new(1)])
    model.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model.predict1(Xumo::SFloat.zeros(10))
    model2 = DNN::Models::Sequential.new([DNN::Layers::InputLayer.new(10), DNN::Layers::Dense.new(1)])
    model2.setup(DNN::Optimizers::SGD.new, DNN::Losses::MeanSquaredError.new)
    model2.predict1(Xumo::SFloat.zeros(10))

    saver = DNN::Savers::JSONSaver.new(model)
    bin = saver.send(:dump_bin)
    loader = DNN::Loaders::JSONLoader.new(model2)
    loader.send(:load_bin, bin)

    x = Xumo::SFloat.new(10).rand
    assert_equal model.predict1(x), model2.predict1(x)
  end
end
