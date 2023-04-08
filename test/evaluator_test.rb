require "test_helper"

include DNN::Models
include DNN::Layers
include DNN::Losses

class TestEvaluator < MiniTest::Unit::TestCase
  # It is accuracy is 1.
  def test_evaluate
    model = Sequential.new
    model << InputLayer.new(3)
    model.setup(SGD.new, MeanSquaredError.new)
    x = Xumo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    y = Xumo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    evaluator = DNN::Evaluator.new(model)
    assert_equal 1, evaluator.evaluate(x, y, batch_size: 1).first
  end

  # It is accuracy is 0.5.
  def test_evaluate2
    model = Sequential.new
    model << InputLayer.new(3)
    model.setup(SGD.new, MeanSquaredError.new)
    x = Xumo::SFloat[[0, 0.5, 1], [0.5, 1, 0]]
    y = Xumo::SFloat[[0, 1, 0.5], [0, 1, 0.5]]
    evaluator = DNN::Evaluator.new(model)
    assert_equal 0.5, evaluator.evaluate(x, y).first
  end
end
