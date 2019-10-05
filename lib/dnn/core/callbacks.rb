module DNN
  module Callbacks

    class Callback
      attr_reader :event
      attr_accessor :model

      def initialize(event, proc = nil)
        @event = event
        if proc
          instance_eval do
            define_singleton_method(:call) { proc.call }
          end
        end
      end

      def call
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'call'")
      end
    end

    # A callback that save the model at the after of the epoch.
    class CheckPoint < Callback
      def initialize(base_file_name)
        super(:after_epoch)
        @base_file_name = base_file_name
      end

      def call
        model.save(@base_file_name + "_epoch#{model.epoch}")
      end
    end


    # A callback to stop training the model early after test on batch.
    class EarlyStopping < Callback
      def initialize(event: :after_train_on_batch, loss: nil, accuracy: nil)
        super(event)
        if event == :after_train_on_batch && !loss
          raise DNN_Error.new("Loss must be set when event is after_train_on_batch.")
        elsif event == :after_test_on_batch && !loss && !accuracy
          raise DNN_Error.new("Either loss or accuracy must be set when event is after_test_on_batch.")
        end
        @loss = loss
        @accuracy = accuracy
      end

      def call
        throw :stop, "Early stopped." if judge_early_stopping
      end

      private def judge_early_stopping
        return true if @loss && model.last_loss <= @loss
        if @event == :after_test_on_batch
          return true if @accuracy && model.last_accuracy >= @accuracy
        end
        false
      end
    end

    # A callback to stop training the model if loss is NaN by after train on batch.
    class NaNStopping < Callback
      def initialize
        super(:after_train_on_batch)
      end

      def call
        throw :stop, "loss is NaN." if model.last_loss.nan?
      end
    end

  end
end
