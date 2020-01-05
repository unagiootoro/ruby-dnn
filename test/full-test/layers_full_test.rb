require "test_helper"

class StubMergeNet < DNN::Models::Model
  def initialize
    super
    @l1 = DNN::Layers::Dense.new(16, weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
    @l2 = DNN::Layers::Dense.new(16, weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
    @l3 = DNN::Layers::Dense.new(16, weight_initializer: DNN::Initializers::RandomNormal.new(seed: 0))
  end

  def forward(x)
    x = DNN::Layers::InputLayer.(x)
    h1 = @l1.(x)
    h2 = @l2.(x)
    x = DNN::Layers::Concatenate.(h1, h2)
    y = @l3.(x)
    y
  end
end

class TestFullLayers < MiniTest::Unit::TestCase
  def test_train
    Numo::SFloat.srand(0)
    x = Numo::SFloat.new(100, 16).rand(0, 1)
    y = Numo::SFloat.new(100, 16).rand(0, 1)

    model = StubMergeNet.new
    model.setup(DNN::Optimizers::Adam.new, DNN::Losses::MeanSquaredError.new)
    model.train(x, y, 2, batch_size: 10, verbose: false)

    assert_in_delta 0.10624708235263824, model.predict(x)[0, 8], 0.01
  end
end
