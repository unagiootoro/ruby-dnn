require "test_helper"

class TestFullSequential < MiniTest::Unit::TestCase
  def test_train
    Numo::SFloat.srand(0)
    x = Numo::SFloat.new(100, 32, 32, 3).rand(0, 1)
    y = Numo::SFloat.new(100, 16, 16, 3).rand(0, 1)

    model = DNN::Models::Sequential.new
    model << DNN::Layers::InputLayer.new([32, 32, 3])
    model << DNN::Layers::Conv2D.new(8, 3, padding: true,
                                     weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
    model << DNN::Layers::MaxPool2D.new(2)
    model << DNN::Layers::Conv2D.new(3, 3, padding: true,
                                     weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
    model.setup(DNN::Optimizers::Adam.new, DNN::Losses::MeanSquaredError.new)
    model.train(x, y, 2, batch_size: 10, verbose: true, test: [x, y])

    assert_in_delta 0.1490977704524994, model.predict(x)[0, 15, 15, 0], 0.01
  end

  def test_train2
    Numo::SFloat.srand(0)
    x = Numo::SFloat.new(100, 32, 32, 3).rand(0, 1)
    y = Numo::SFloat.new(100, 16, 16, 3).rand(0, 1)

    model = DNN::Models::Sequential.new
    model << DNN::Layers::InputLayer.new([32, 32, 3])
    model << DNN::Layers::Conv2D.new(8, 3, padding: true,
                                     weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
    model << DNN::Layers::MaxPool2D.new(2)
    model << DNN::Layers::Conv2D.new(3, 3, padding: true,
                                     weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
    model.setup(DNN::Optimizers::SGD.new(momentum: 0.9, clip_norm: 0.5), DNN::Losses::MeanSquaredError.new)
    model.train(x, y, 2, batch_size: 10, verbose: false, test: [x, y])

    assert_in_delta 0.42336219549179077, model.predict(x)[0, 15, 15, 0], 0.01
  end
end
