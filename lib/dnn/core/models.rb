module DNN
  module Models

    # This class deals with the model of the network.
    class Model
      attr_accessor :optimizer
      attr_accessor :loss_func

      # Load marshal model.
      # @param [String] file_name File name of marshal model to load.
      # @return [DNN::Models::Model] Return the loaded model.
      def self.load(file_name)
        model = self.new
        loader = Loaders::MarshalLoader.new(model)
        loader.load(file_name)
        model
      end

      def initialize
        @optimizer = nil
        @loss_func = nil
        @last_link = nil
        @built = false
        @callbacks = {
          before_epoch: [],
          after_epoch: [],
          before_train_on_batch: [],
          after_train_on_batch: [],
          before_test_on_batch: [],
          after_test_on_batch: [],
        }
        @layers_cache = nil
      end

      # This method is provided for compatibility with v0.12.4.
      # Load hash model parameters.
      # @param [Hash] hash Hash to load model parameters.
      def load_hash_params(hash)
        has_param_layers_params = hash[:params]
        has_param_layers_index = 0
        has_param_layers.uniq.each do |layer|
          hash_params = has_param_layers_params[has_param_layers_index]
          hash_params.each do |key, (shape, bin)|
            data = Xumo::SFloat.from_binary(bin).reshape(*shape)
            layer.get_params[key].data = data
          end
          has_param_layers_index += 1
        end
      end

      # This method is provided for compatibility with v0.12.4.
      # Load json model parameters.
      # @param [String] json_str JSON string to load model parameters.
      def load_json_params(json_str)
        hash = JSON.parse(json_str, symbolize_names: true)
        has_param_layers_params = hash[:params]
        has_param_layers_index = 0
        has_param_layers.uniq.each do |layer|
          hash_params = has_param_layers_params[has_param_layers_index]
          hash_params.each do |key, (shape, base64_param)|
            bin = Base64.decode64(base64_param)
            data = Xumo::SFloat.from_binary(bin).reshape(*shape)
            layer.get_params[key].data = data
          end
          has_param_layers_index += 1
        end
      end

      # Set optimizer and loss_func to model.
      # @param [DNN::Optimizers::Optimizer] optimizer Optimizer to use for learning.
      # @param [DNN::Losses::Loss] loss_func Loss function to use for learning.
      def setup(optimizer, loss_func)
        unless optimizer.is_a?(Optimizers::Optimizer)
          raise TypeError.new("optimizer:#{optimizer.class} is not an instance of DNN::Optimizers::Optimizer class.")
        end
        unless loss_func.is_a?(Losses::Loss)
          raise TypeError.new("loss_func:#{loss_func.class} is not an instance of DNN::Losses::Loss class.")
        end
        @optimizer = optimizer
        @loss_func = loss_func
      end

      # Start training.
      # Setup the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @param [Integer] epochs Number of training.
      # @param [Integer] initial_epoch Initial epoch.
      # @param [Integer] batch_size Batch size used for one training.
      # @param [Array | NilClass] test If you to test the model for every 1 epoch,
      #                                specify [x_test, y_test]. Don't test to the model, specify nil.
      # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
      def train(x, y, epochs,
                batch_size: 1,
                initial_epoch: 1,
                test: nil,
                verbose: true)
        raise DNN_Error.new("The model is not optimizer setup complete.") unless @optimizer
        raise DNN_Error.new("The model is not loss_func setup complete.") unless @loss_func
        check_xy_type(x, y)
        iter = Iterator.new(x, y)
        num_train_datas = x.is_a?(Array) ? x[0].shape[0] : x.shape[0]
        (initial_epoch..epochs).each do |epoch|
          call_callbacks(:before_epoch, epoch)
          puts "【 epoch #{epoch}/#{epochs} 】" if verbose
          iter.foreach(batch_size) do |x_batch, y_batch, index|
            loss_value = train_on_batch(x_batch, y_batch)
            if loss_value.is_a?(Xumo::SFloat)
              loss_value = loss_value.mean
            elsif loss_value.nan?
              puts "\nloss is nan" if verbose
              return
            end
            num_trained_datas = (index + 1) * batch_size
            num_trained_datas = num_trained_datas > num_train_datas ? num_train_datas : num_trained_datas
            log = "\r"
            40.times do |i|
              if i < num_trained_datas * 40 / num_train_datas
                log << "="
              elsif i == num_trained_datas * 40 / num_train_datas
                log << ">"
              else
                log << "_"
              end
            end
            log << "  #{num_trained_datas}/#{num_train_datas} loss: #{sprintf('%.8f', loss_value)}"
            print log if verbose
          end
          if test
            acc, test_loss = accuracy(test[0], test[1], batch_size: batch_size)
            print "  accuracy: #{acc}, test loss: #{sprintf('%.8f', test_loss)}" if verbose
          end
          puts "" if verbose
          call_callbacks(:after_epoch, epoch)
        end
      end

      alias fit train

      # Training once.
      # Setup the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @param [Integer] batch_size Batch size used for one test.
      # @return [Float | Numo::SFloat] Return loss value in the form of Float or Numo::SFloat.
      def train_on_batch(x, y)
        raise DNN_Error.new("The model is not optimizer setup complete.") unless @optimizer
        raise DNN_Error.new("The model is not loss_func setup complete.") unless @loss_func
        check_xy_type(x, y)
        call_callbacks(:before_train_on_batch)
        x = forward(x, true)
        loss_value = @loss_func.loss(x, y, layers)
        dy = @loss_func.backward(x, y)
        backward(dy)
        @optimizer.update(layers)
        @loss_func.regularizers_backward(layers)
        call_callbacks(:after_train_on_batch, loss_value)
        loss_value
      end

      # Evaluate model and get accuracy of test data.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @param [Integer] batch_size Batch size used for one test.
      # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
      def accuracy(x, y, batch_size: 100)
        check_xy_type(x, y)
        num_test_datas = x.is_a?(Array) ? x[0].shape[0] : x.shape[0]
        batch_size = batch_size >= num_test_datas[0] ? num_test_datas : batch_size
        iter = Iterator.new(x, y, random: false)
        total_correct = 0
        sum_loss = 0
        max_steps = (num_test_datas.to_f / batch_size).ceil
        iter.foreach(batch_size) do |x_batch, y_batch|
          correct, loss_value = test_on_batch(x_batch, y_batch)
          total_correct += correct
          sum_loss += loss_value.is_a?(Xumo::SFloat) ? loss_value.mean : loss_value
        end
        mean_loss = sum_loss / max_steps
        [total_correct.to_f / num_test_datas, mean_loss]
      end

      # Evaluate once.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
      def test_on_batch(x, y)
        call_callbacks(:before_test_on_batch)
        x = forward(x, false)
        correct = evaluate(x, y)
        loss_value = @loss_func.loss(x, y, layers)
        call_callbacks(:after_test_on_batch, loss_value)
        [correct, loss_value]
      end

      # Implement the process to evaluate this model.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      private def evaluate(x, y)
        if x.shape[1..-1] == [1]
          correct = 0
          x.shape[0].times do |i|
            if @loss_func.is_a?(Losses::SigmoidCrossEntropy)
              correct += 1 if (x[i, 0] < 0 && y[i, 0] < 0.5) || (x[i, 0] >= 0 && y[i, 0] >= 0.5)
            else
              correct += 1 if (x[i, 0] < 0 && y[i, 0] < 0) || (x[i, 0] >= 0 && y[i, 0] >= 0)
            end
          end
        else
          correct = x.max_index(axis: 1).eq(y.max_index(axis: 1)).count
        end
        correct
      end

      # Predict data.
      # @param [Numo::SFloat] x Input data.
      def predict(x)
        check_xy_type(x)
        forward(x, false)
      end

      # Predict one data.
      # @param [Numo::SFloat] x Input data. However, x is single data.
      def predict1(x)
        check_xy_type(x)
        predict(x.reshape(1, *x.shape))[0, false]
      end

      # Add callback function.
      # @param [Symbol] event Callback event. The following can be used for event.
      #                       before_epoch: Process: performed before one training.
      #                       after_epoch: Process:  performed after one training.
      #                       before_train_on_batch: Set the proc to be performed before train on batch processing.
      #                       after_train_on_batch:  Set the proc to be performed after train on batch processing.
      #                       before_test_on_batch:  Set the proc to be performed before test on batch processing.
      #                       after_test_on_batch:   Set the proc to be performed after test on batch processing.
      def add_callback(event, callback)
        raise DNN_UnknownEventError.new("Unknown event #{event}.") unless @callbacks.has_key?(event)
        @callbacks[event] << callback
      end

      # Clear the callback function registered for each event.
      # @param [Symbol] event Callback event. The following can be used for event.
      #                       before_epoch: Process: performed before one training.
      #                       after_epoch: Process:  performed after one training.
      #                       before_train_on_batch: Set the proc to be performed before train on batch processing.
      #                       after_train_on_batch:  Set the proc to be performed after train on batch processing.
      #                       before_test_on_batch:  Set the proc to be performed before test on batch processing.
      #                       after_test_on_batch:   Set the proc to be performed after test on batch processing.
      def clear_callbacks(event)
        raise DNN_UnknownEventError.new("Unknown event #{event}.") unless @callbacks.has_key?(event)
        @callbacks[event] = []
      end

      # Save the model in marshal format.
      # @param [String] file_name Name to save model.
      def save(file_name)
        saver = Savers::MarshalSaver.new(self)
        saver.save(file_name)
      end

      # @return [DNN::Models::Model] Return the copy this model.
      def copy
        Marshal.load(Marshal.dump(self))
      end

      # Get the all layers.
      # @return [Array] All layers array.
      def layers
        raise DNN_Error.new("This model is not built. You need build this model using predict or train.") unless built?
        return @layers_cache if @layers_cache
        layers = []
        get_layers = -> link do
          return unless link
          layers.unshift(link.layer)
          if link.is_a?(TwoInputLink)
            get_layers.(link.prev1)
            get_layers.(link.prev2)
          else
            get_layers.(link.prev)
          end
        end
        get_layers.(@last_link)
        @layers_cache = layers.uniq
      end

      # Get the all has param layers.
      # @return [Array] All has param layers array.
      def has_param_layers
        layers.select { |layer| layer.is_a?(Layers::HasParamLayer) }
      end

      # Get the layer that the model has.
      # @param [Symbol] The name of the layer to get.
      # @return [DNN::Layers::Layer] Return the layer.
      def get_layer(name)
        layers.find { |layer| layer.name == name }
      end

      # @return [Boolean] If model have already been built then return true.
      def built?
        @built
      end

      private

      def forward(x, learning_phase)
        DNN.learning_phase = learning_phase
        @layers_cache = nil
        y, @last_link = call(x)
        unless @built
          @built = true
          naming
        end
        y
      end

      def backward(dy)
        @last_link.backward(dy)
      end

      def call_callbacks(event, *args)
        @callbacks[event].each do |callback|
          callback.call(*args)
        end
      end

      def naming
        layers.each do |layer|
          id = layers.select { |l| l.is_a?(layer.class) }.index(layer)
          class_name = layer.class.name.split("::").last
          layer.name = "#{class_name}_#{id}".to_sym unless layer.name
          if layer.is_a?(Layers::HasParamLayer)
            layer.get_params.each do |param_key, param|
              param.name = "#{layer.name}__#{param_key}".to_sym unless param.name
            end
          end
        end
      end

      def check_xy_type(x, y = nil)
        if !x.is_a?(Xumo::SFloat) && !x.is_a?(Array)
          raise TypeError.new("x:#{x.class.name} is not an instance of #{Xumo::SFloat.name} class or Array class.")
        end
        if y && !y.is_a?(Xumo::SFloat) && !x.is_a?(Array)
          raise TypeError.new("y:#{y.class.name} is not an instance of #{Xumo::SFloat.name} class or Array class.")
        end
      end
    end


    class Sequential < Model
      attr_reader :stack

      # @param [Array] stack All layers possessed by the model.
      def initialize(stack = [])
        super()
        @stack = stack.clone
      end

      # Add layer to the model.
      # @param [DNN::Layers::Layer] layer Layer to add to the model.
      # @return [DNN::Models::Model] Return self.
      def add(layer)
        unless layer.is_a?(Layers::Layer) || layer.is_a?(Model)
          raise TypeError.new("layer: #{layer.class.name} is not an instance of the DNN::Layers::Layer class or DNN::Models::Model class.")
        end
        @stack << layer
        self
      end

      alias << add

      # Remove layer to the model.
      # @param [DNN::Layers::Layer] layer Layer to remove to the model.
      # @return [Boolean] Return true if success for remove layer.
      def remove(layer)
        unless layer.is_a?(Layers::Layer) || layer.is_a?(Model)
          raise TypeError.new("layer: #{layer.class.name} is not an instance of the DNN::Layers::Layer class or DNN::Models::Model class.")
        end
        @stack.delete(layer) ? true : false
      end

      def call(x)
        @stack.each do |layer|
          x = layer.(x)
        end
        x
      end
    end

  end
end
