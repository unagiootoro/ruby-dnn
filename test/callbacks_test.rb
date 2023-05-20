require "test_helper"

class TestLambdaCallback < MiniTest::Unit::TestCase
  def test_initialize
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch) { return true }
    assert_equal true, cbk.before_epoch
  end
end

class TestEarlyStopping < MiniTest::Unit::TestCase
  def test_after_train_on_batch
    cbk = DNN::Callbacks::EarlyStopping.new(:loss, 0.1)
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:loss, 0.09)
    cbk.after_train_on_batch
    assert_equal cbk.runner.send(:check_stop_requested), "Early stopped."
  end

  def test_after_train_on_batch2
    cbk = DNN::Callbacks::EarlyStopping.new(:loss, 0.1)
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:loss, 0.11)
    cbk.after_train_on_batch
    assert_nil cbk.runner.send(:check_stop_requested)
  end

  def test_after_train_on_batch3
    cbk = DNN::Callbacks::EarlyStopping.new(:accuracy, 0.1)
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:accuracy, 0.11)
    cbk.after_train_on_batch
    assert_equal cbk.runner.send(:check_stop_requested), "Early stopped."
  end

  def test_after_train_on_batch4
    cbk = DNN::Callbacks::EarlyStopping.new(:accuracy, 0.1)
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:accuracy, 0.09)
    cbk.after_train_on_batch
    assert_nil cbk.runner.send(:check_stop_requested)
  end

  def test_after_epoch
    cbk = DNN::Callbacks::EarlyStopping.new(:test_loss, 0.1)
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:test_loss, 0.09)
    cbk.after_epoch
    assert_equal cbk.runner.send(:check_stop_requested), "Early stopped."
  end

  def test_after_epoch2
    cbk = DNN::Callbacks::EarlyStopping.new(:test_loss, 0.1)
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:test_loss, 0.11)
    cbk.after_epoch
    assert_nil cbk.runner.send(:check_stop_requested)
  end

  def test_after_epoch3
    cbk = DNN::Callbacks::EarlyStopping.new(:test_accuracy, 0.1)
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:test_accuracy, 0.11)
    cbk.after_epoch
    assert_equal cbk.runner.send(:check_stop_requested), "Early stopped."
  end

  def test_after_epoch4
    cbk = DNN::Callbacks::EarlyStopping.new(:test_accuracy, 0.1)
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:test_accuracy, 0.09)
    cbk.after_epoch
    assert_nil cbk.runner.send(:check_stop_requested)
  end
end

class TestNaNStopping < MiniTest::Unit::TestCase
  def test_after_train_on_batch
    cbk = DNN::Callbacks::NaNStopping.new
    cbk.runner = DNN::Trainer.new(DNN::Models::Sequential.new)
    cbk.runner.add_log(:loss, Float::NAN)
    cbk.after_train_on_batch
    assert_equal "loss is NaN.", cbk.runner.send(:check_stop_requested)
  end
end

class TestLogger < MiniTest::Unit::TestCase
  def test_after_epoch
    cbk = DNN::Callbacks::Logger.new
    trainer = DNN::Trainer.new(DNN::Models::Sequential.new)
    trainer.add_callback(cbk)
    trainer.add_log(:epoch, 1)
    assert_equal [1], cbk.get_log(:epoch)
  end
end
