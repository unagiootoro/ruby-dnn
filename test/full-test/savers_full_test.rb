require "test_helper"

class StubSaveChain < DNN::Models::Chain
  def initialize
    @conv2d = DNN::Layers::Conv2D.new(3, 3, padding: true,
                                      weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
  end

  def forward(x)
    x = @conv2d.(x)
    x = DNN::Layers::MaxPool2D.(x, 2)
    x
  end
end

class StubSaveModel < DNN::Models::Model
  def initialize
    super
    @chain = StubSaveChain.new
    @conv2d = DNN::Layers::Conv2D.new(3, 3, padding: true,
                                      weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
  end

  def forward(x)
    x = InputLayer.new([32, 32, 3]).(x)
    x = @chain.(x)
    x = @conv2d.(x)
    x
  end
end

class TestMarshalSaver < MiniTest::Unit::TestCase
  def test_load
    Numo::SFloat.srand(0)
    x = Numo::SFloat.new(100, 32, 32, 3).rand(0, 1)
    y = Numo::SFloat.new(100, 16, 16, 3).rand(0, 1)

    model = StubSaveModel.new
    model.setup(DNN::Optimizers::Adam.new, DNN::Losses::MeanSquaredError.new)
    model.train(x, y, 2, batch_size: 10, verbose: false)

    model2 = StubSaveModel.new

    saver = DNN::Savers::MarshalSaver.new(model, include_model: true)
    saver.save("test/full-test/tmp_model.marshal")
    loader = DNN::Loaders::MarshalLoader.new(model2)
    loader.load("test/full-test/tmp_model.marshal")
    File.unlink("test/full-test/tmp_model.marshal")

    model.train(x, y, 2, batch_size: 10, verbose: false)
    model2.train(x, y, 2, batch_size: 10, verbose: false)

    assert_in_delta model.predict(x)[0, 15, 15, 0], model2.predict(x)[0, 15, 15, 0], 0.01
  end

  def test_load2
    Numo::SFloat.srand(0)
    x = Numo::SFloat.new(100, 32, 32, 3).rand(0, 1)
    y = Numo::SFloat.new(100, 16, 16, 3).rand(0, 1)

    sequential = DNN::Models::Sequential.new
    sequential << DNN::Layers::Conv2D.new(8, 3, padding: true,
                                          weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
    sequential << DNN::Layers::MaxPool2D.new(2)

    model = DNN::Models::Sequential.new
    model << DNN::Layers::InputLayer.new([32, 32, 3])
    model << sequential
    model << DNN::Layers::Conv2D.new(3, 3, padding: true,
                                     weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
    model.setup(DNN::Optimizers::Adam.new, DNN::Losses::MeanSquaredError.new)
    model.train(x, y, 2, batch_size: 10, verbose: false)

    model2 = DNN::Models::Sequential.new

    saver = DNN::Savers::MarshalSaver.new(model, include_model: true)
    saver.save("test/full-test/tmp_model.marshal")
    loader = DNN::Loaders::MarshalLoader.new(model2)
    loader.load("test/full-test/tmp_model.marshal")
    File.unlink("test/full-test/tmp_model.marshal")

    model.train(x, y, 2, batch_size: 10, verbose: false)
    model2.train(x, y, 2, batch_size: 10, verbose: false)

    assert_in_delta model.predict(x)[0, 15, 15, 0], model2.predict(x)[0, 15, 15, 0], 0.01
  end
end
