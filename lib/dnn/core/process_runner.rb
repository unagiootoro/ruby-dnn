module DNN
  class ProcessRunner
    def initialize
      @callbacks = []
      @last_logs = {}
    end

    # Add callback function.
    # @param [Callback] callback Callback object.
    def add_callback(callback)
      callback.runner = self
      @callbacks << callback
    end

    # Add lambda callback.
    # @param [Symbol] event Event to execute callback.
    # @yield Register the contents of the callback.
    def add_lambda_callback(event, &block)
      callback = Callbacks::LambdaCallback.new(event, &block)
      callback.runner = self
      @callbacks << callback
    end

    # Clear the callback function registered for each event.
    def clear_callbacks
      @callbacks = []
    end

    def call_callbacks(event, *args)
      @callbacks.each do |callback|
        callback.send(event, *args) if callback.respond_to?(event)
      end
    end

    def add_log(tag, value)
      @last_logs[tag] = value
      call_callbacks(:on_log_added, tag, value)
    end

    def last_log(tag)
      @last_logs[tag]
    end
  end
end
