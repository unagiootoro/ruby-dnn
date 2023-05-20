module DNN
  module Callbacks

    class Callback
      attr_accessor :runner

      # Please implement the method used for callback event.

      # Process performed before all training.
      # def before_train; end

      # Process performed after all training.
      # def after_train; end

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

      # Set the proc to run when adding a log.
      # The following logs will be recorded.
      # epoch:          Current epoch.
      # step:           Current step in epoch.
      # loss:           Batch training loss.
      # accuracy:       Batch training accuracy.
      # test_loss:      Mean test loss.
      # test_accuracy:  Test accuracy.
      # def on_log_added(tag, valud); end
    end

    # This callback wrap the lambda function.
    class LambdaCallback < Callback
      # @param [Symbol] event Event to execute callback.
      # @yield Register the contents of the callback.
      def initialize(event, &block)
        instance_eval do
          define_singleton_method(event) { block.call }
        end
      end
    end

    # A callback that save the model at the after of the epoch.
    # @param [DNN::Model] model Target model.
    # @param [String] base_file_name Base file name for saving.
    # @param [Boolean] include_model When set a true, save data included model structure.
    # @param [Integer] interval Save interval.
    class CheckPoint < Callback
      def initialize(model, base_file_name, include_model: true, interval: 1)
        @model = model
        @base_file_name = base_file_name
        @include_model = include_model
        @interval = interval
      end

      def after_epoch
        saver = Savers::MarshalSaver.new(@model, include_model: @include_model)
        if @runner.last_log(:epoch) % @interval == 0
          saver.save(@base_file_name + "_epoch#{model.last_log(:epoch)}.marshal")
        end
      end
    end

    # A callback to stop training the model early after test on batch.
    # @param [Symbol] trigger A log that triggers early stopping.
    #                         Specify one of :loss, :test_loss, :test_accuracy
    # @param [Float] tolerance Tolerance value for early stopping.
    class EarlyStopping < Callback
      def initialize(trigger, tolerance)
        @trigger = trigger
        @tolerance = tolerance
      end

      def after_train_on_batch
        request_early_stop if judge_early_stopping_train
      end

      def after_epoch
        request_early_stop if judge_early_stopping_test
      end

      private

      def request_early_stop
        @runner.request_stop("Early stopped.")
      end

      def judge_early_stopping_train
        case @trigger
        when :loss
          return true if @runner.last_log(@trigger) <= @tolerance
        when :accuracy
          return true if @runner.last_log(@trigger) >= @tolerance
        end
        false
      end

      def judge_early_stopping_test
        case @trigger
        when :test_loss
          return true if @runner.last_log(@trigger) <= @tolerance
        when :test_accuracy
          return true if @runner.last_log(@trigger) >= @tolerance
        end
        false
      end
    end

    # A callback to stop training the model if loss is NaN by after train on batch.
    class NaNStopping < Callback
      def after_train_on_batch
        @runner.request_stop("loss is NaN.") if @runner.last_log(:loss).nan?
      end
    end

    # A callback that save the log.
    # The following logs will be recorded.
    # epoch:          Current epoch.
    # step:           Current step in epoch.
    # loss:           Batch training loss.
    # accuracy:       Batch training accuracy.
    # test_loss:      Mean test loss.
    # test_accuracy:  Test accuracy.
    class Logger < Callback
      def initialize
        @log = {}
      end

      def on_log_added(tag, value)
        if @log[tag]
          @log[tag] << value
        else
          @log[tag] = [value]
        end
      end

      # Get a log.
      # @param [Symbol] tag Tag indicating the type of Log.
      # @return [Array] Return the recorded log.
      def get_log(tag)
        @log[tag] ||= []
        @log[tag]
      end
    end

  end
end
