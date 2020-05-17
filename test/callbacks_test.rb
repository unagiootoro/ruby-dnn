require "test_helper"

class TestLambdaCallback < MiniTest::Unit::TestCase
  def test_initialize
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch) { return true }
    assert_equal true, cbk.before_epoch
  end
end

class StubCallbacksTestModel < DNN::Models::Model
end

class TestEarlyStopping < MiniTest::Unit::TestCase
  def test_after_train_on_batch
    cbk = DNN::Callbacks::EarlyStopping.new(:train_loss, 0.1)
    stub_model = StubCallbacksTestModel.new
    cbk.model = stub_model
    stub_model.last_log[:train_loss] = 0.09
    assert_throws :stop do
      cbk.after_train_on_batch
    end
  end

  def test_after_epoch
    cbk = DNN::Callbacks::EarlyStopping.new(:test_accuracy, 0.1)
    stub_model = StubCallbacksTestModel.new
    cbk.model = stub_model
    stub_model.last_log[:test_accuracy] = 0.11
    assert_throws :stop do
      cbk.after_epoch
    end
  end

  def test_after_epoch2
    cbk = DNN::Callbacks::EarlyStopping.new(:test_accuracy, 0.1)
    stub_model = StubCallbacksTestModel.new
    cbk.model = stub_model
    stub_model.last_log[:test_accuracy] = 0.11
    assert_throws :stop do
      cbk.after_epoch
    end
  end
end

class TestNaNStopping < MiniTest::Unit::TestCase
  def test_after_train_on_batch
    cbk = DNN::Callbacks::NaNStopping.new
    stub_model = StubCallbacksTestModel.new
    cbk.model = stub_model
    stub_model.last_log[:train_loss] = Float::NAN
    assert_throws :stop do
      cbk.after_train_on_batch
    end
  end
end

class TestLogger < MiniTest::Unit::TestCase
  def test_after_epoch
    cbk = DNN::Callbacks::Logger.new
    stub_model = StubCallbacksTestModel.new
    cbk.model = stub_model
    stub_model.last_log[:epoch] = 1
    stub_model.last_log[:test_loss] = 2
    stub_model.last_log[:test_accuracy] = 3
    cbk.after_epoch

    assert_equal Xumo::UInt32[1], cbk.get_log(:epoch)
    assert_equal Xumo::SFloat[2], cbk.get_log(:test_loss)
    assert_equal Xumo::SFloat[3], cbk.get_log(:test_accuracy)
  end

  def test_after_train_on_batch
    cbk = DNN::Callbacks::Logger.new
    stub_model = StubCallbacksTestModel.new
    cbk.model = stub_model
    stub_model.last_log[:train_loss] = 1
    stub_model.last_log[:step] = 2
    cbk.after_train_on_batch

    assert_equal Xumo::SFloat[1], cbk.get_log(:train_loss)
    assert_equal Xumo::UInt32[2], cbk.get_log(:step)
  end
end
