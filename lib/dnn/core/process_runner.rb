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

    def call_callbacks(event)
      @callbacks.each do |callback|
        callback.send(event) if callback.respond_to?(event)
      end
    end

    def set_last_log(tag, data)
      @last_logs[tag] = data
    end

    def last_log(tag)
      @last_logs[tag]
    end
  end
end
