require "test_helper"

class TestTrainer < MiniTest::Unit::TestCase
  def test_metrics_to_str
    met = { accuracy: 0.00011, test_loss: 0.00011 }
    str_met = "accuracy: 0.0001, test_loss: 0.0001"
    trainer = DNN::Trainer.new(DNN::Models::Model.new)
    assert_equal str_met, trainer.send(:metrics_to_str, met)
  end

  def test_metrics_to_str2
    met = { accuracy: [0.00011, 0.00011], test_loss: [0.00011, 0.00011] }
    str_met = "accuracy: [0.0001, 0.0001], test_loss: [0.0001, 0.0001]"
    trainer = DNN::Trainer.new(DNN::Models::Model.new)
    assert_equal str_met, trainer.send(:metrics_to_str, met)
  end
end
