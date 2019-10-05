require "test_helper"

class TestCallback < MiniTest::Unit::TestCase
  def test_initialize
    func = -> { return true }
    cbk = DNN::Callbacks::Callback.new(:before_epoch, func)
    assert_equal true, cbk.call
  end
end


class StubModel < DNN::Models::Model
  attr_accessor :epoch
  attr_accessor :last_loss
  attr_accessor :last_accuracy
  attr_accessor :file_name

  def save(file_name)
    @file_name = file_name
  end
end


class TestCheckPoint < MiniTest::Unit::TestCase
  def test_call
    cbk = DNN::Callbacks::CheckPoint.new("save")
    stub_model = StubModel.new
    cbk.model = stub_model
    stub_model.epoch = 1
    cbk.call
    assert_equal "save_epoch1", stub_model.file_name
  end
end


class TestEarlyStopping < MiniTest::Unit::TestCase
  def test_call
    cbk = DNN::Callbacks::EarlyStopping.new(loss: 0.1)
    stub_model = StubModel.new
    cbk.model = stub_model
    stub_model.last_loss = 0.09
    assert_throws :stop do
      cbk.call
    end
  end

  def test_call2
    cbk = DNN::Callbacks::EarlyStopping.new(event: :after_test_on_batch, accuracy: 0.1)
    stub_model = StubModel.new
    cbk.model = stub_model
    stub_model.last_accuracy = 0.11
    assert_throws :stop do
      cbk.call
    end
  end
end


class TestNaNStopping < MiniTest::Unit::TestCase
  def test_call
    cbk = DNN::Callbacks::NaNStopping.new
    stub_model = StubModel.new
    cbk.model = stub_model
    stub_model.last_loss = Float::NAN
    assert_throws :stop do
      cbk.call
    end
  end
end
