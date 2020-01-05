class TestCheckPointe < MiniTest::Unit::TestCase
  def test_after_epoch
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
    model.add_callback(DNN::Callbacks::CheckPoint.new("test/full-test/tmp_model"))
    model.train(x, y, 1, batch_size: 10, verbose: false)
    model.clear_callbacks

    model2 = DNN::Models::Sequential.new

    loader = DNN::Loaders::MarshalLoader.new(model2)
    loader.load("test/full-test/tmp_model_epoch1.marshal")
    File.unlink("test/full-test/tmp_model_epoch1.marshal")

    model.train(x, y, 1, batch_size: 10, verbose: false)
    model2.train(x, y, 1, batch_size: 10, verbose: false)

    assert_in_delta model.predict(x)[0, 15, 15, 0], model2.predict(x)[0, 15, 15, 0], 0.01
  end
end
