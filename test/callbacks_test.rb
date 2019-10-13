require "test_helper"

class TestLambdaCallback < MiniTest::Unit::TestCase
  def test_initialize
    func = -> { return true }
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch, func)
    assert_equal true, cbk.before_epoch
  end
end


class StubCallbacksTestModel < DNN::Models::Model
  attr_accessor :file_name

  def save(file_name)
    @file_name = file_name
  end
end


class TestCheckPoint < MiniTest::Unit::TestCase
  def test_after_epoch
    cbk = DNN::Callbacks::CheckPoint.new("save")
    stub_model = StubCallbacksTestModel.new
    cbk.model = stub_model
    stub_model.last_log[:epoch] = 1
    cbk.after_epoch
    assert_equal "save_epoch1", stub_model.file_name
  end
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

  def test_after_test_on_batch
    cbk = DNN::Callbacks::EarlyStopping.new(:test_accuracy, 0.1)
    stub_model = StubCallbacksTestModel.new
    cbk.model = stub_model
    stub_model.last_log[:test_accuracy] = 0.11
    assert_throws :stop do
      cbk.after_test_on_batch
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