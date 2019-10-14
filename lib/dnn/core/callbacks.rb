module DNN
  module Callbacks

    class Callback
      attr_accessor :model

      # Please implement the method used for callback event.

      # Process performed before one training.
      # def before_epoch; end

      # Process performed after one training.
      # def after_epoch; end

      # Set the proc to be performed before train on batch processing.
      # def before_train_on_batch; end

      # Set the proc to be performed after train on batch processing.
      # def after_train_on_batch; end

      # Set the proc to be performed before test on batch processing.
      # def before_test_on_batch; end

      # Set the proc to be performed after test on batch processing.
      # def after_test_on_batch; end
    end

    # This callback wrap the lambda function.
    class LambdaCallback < Callback
      def initialize(event, lambda)
        instance_eval do
          define_singleton_method(event) { lambda.call }
        end
      end
    end

    # A callback that save the model at the after of the epoch.
    class CheckPoint < Callback
      def initialize(base_file_name)
        @base_file_name = base_file_name
      end

      def after_epoch
        model.save(@base_file_name + "_epoch#{model.last_log[:epoch]}.marshal")
      end
    end

    # A callback to stop training the model early after test on batch.
    class EarlyStopping < Callback
      def initialize(trigger, tolerance)
        @trigger = trigger
        @tolerance = tolerance
      end

      def after_train_on_batch
        throw :stop, "Early stopped." if judge_early_stopping_train
      end

      def after_epoch
        throw :stop, "Early stopped." if judge_early_stopping_test
      end

      private

      def judge_early_stopping_train
        case @trigger
        when :train_loss
          return true if model.last_log[@trigger].mean <= @tolerance
        end
        false
      end

      def judge_early_stopping_test
        case @trigger
        when :test_loss
          return true if model.last_log[@trigger].mean <= @tolerance
        when :test_accuracy
          return true if model.last_log[@trigger] >= @tolerance
        end
        false
      end
    end

    # A callback to stop training the model if loss is NaN by after train on batch.
    class NaNStopping < Callback
      def after_train_on_batch
        throw :stop, "loss is NaN." if model.last_log[:train_loss].nan?
      end
    end

    # A callback that save the log.
    # The following logs will be recorded.
    # epoch:          Current epoch.
    # train_loss:     Batch training loss.
    # test_loss:      Mean test loss.
    # test_accuracy:  Test accuracy.
    class Logger < Callback
      def initialize
        @log = {
          epoch: [],
          train_loss: [],
          test_loss: [],
          test_accuracy: [],
        }
      end

      def after_epoch
        @log[:epoch] << model.last_log[:epoch]
        @log[:test_loss] << model.last_log[:test_loss]
        @log[:test_accuracy] << model.last_log[:test_accuracy]
      end

      def after_train_on_batch
        @log[:train_loss] << model.last_log[:train_loss]
      end

      # Get a log.
      # @param [Symbol] tag Tag indicating the type of Log.
      # @return [Numo::NArray] Return the recorded log.
      def get_log(tag)
        case tag
        when :epoch
          Numo::UInt32.cast(@log[tag])
        else
          Numo::SFloat.cast(@log[tag])
        end
      end
    end

  end
end
