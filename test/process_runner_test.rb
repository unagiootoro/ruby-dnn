require "test_helper"
require "json"

class TestProcessRunner < MiniTest::Unit::TestCase
  # It is including callback function in @callback.
  def test_add_callback
    runner = DNN::ProcessRunner.new
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch) {}
    runner.add_callback(cbk)
    assert_equal [cbk], runner.instance_variable_get(:@callbacks)
  end

  # It is including callback function in @callback.
  def test_add_lambda_callback
    runner = DNN::ProcessRunner.new
    runner.add_lambda_callback(:before_epoch) {}
    cbk = runner.instance_variable_get(:@callbacks)[0]
    assert_kind_of DNN::Callbacks::LambdaCallback, cbk
  end

  # It is not including callback function in @callback.
  def test_clear_callbacks
    runner = DNN::ProcessRunner.new
    cbk = DNN::Callbacks::LambdaCallback.new(:before_epoch) {}
    runner.add_callback(cbk)
    runner.clear_callbacks
    assert_equal [], runner.instance_variable_get(:@callbacks)
  end

  # It is running all callback function.
  def test_call_callbacks
    call_cnt = 0
    call_flg = [0, 0]
    prc1 = proc do
      call_cnt += 1
      call_flg[0] = call_cnt
    end
    prc2 = proc do
      call_cnt += 1
      call_flg[1] = call_cnt
    end
    cbk1 = DNN::Callbacks::LambdaCallback.new(:before_epoch, &prc1)
    cbk2 = DNN::Callbacks::LambdaCallback.new(:before_epoch, &prc2)
    runner = DNN::ProcessRunner.new
    runner.add_callback(cbk1)
    runner.add_callback(cbk2)
    runner.send(:call_callbacks, :before_epoch)
    assert_equal [1, 2], call_flg
  end
end
