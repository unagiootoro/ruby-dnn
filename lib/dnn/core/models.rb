module DNN
  module Models

    # This class deals with the model of the network.
    class Model
      # Load marshal model.
      # @param [String] file_name File name of marshal model to load.
      def self.load(file_name)
        loader = Loaders::MarshalLoader.new(self.new)
        loader.load(file_name)
      end

      def initialize
        @optimizer = nil
        @last_link = nil
        @setup_completed = false
        @built = false
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
        @setup_completed = true
        @optimizer = optimizer
        @loss_func = loss_func
      end

      # Start training.
      # Setup the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @param [Integer] epochs Number of training.
      # @param [Integer] batch_size Batch size used for one training.
      # @param [Array | NilClass] test If you to test the model for every 1 epoch,
      #                                specify [x_test, y_test]. Don't test to the model, specify nil.
      # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
      # @param [Lambda] before_epoch_cbk Process performed before one training.
      # @param [Lambda] after_epoch_cbk Process performed after one training.
      # @param [Lambda] before_train_on_batch_cbk Set the proc to be performed before train on batch processing.
      # @param [Lambda] after_train_on_batch_cbk Set the proc to be performed after train on batch processing.
      # @param [Lambda] before_test_on_batch_cbk Set the proc to be performed before test on batch processing.
      # @param [Lambda] after_test_on_batch_cbk Set the proc to be performed after test on batch processing.
      def train(x, y, epochs,
                batch_size: 1,
                test: nil,
                verbose: true,
                before_epoch_cbk: nil,
                after_epoch_cbk: nil,
                before_train_on_batch_cbk: nil,
                after_train_on_batch_cbk: nil,
                before_test_on_batch_cbk: nil,
                after_test_on_batch_cbk: nil)
        raise DNN_Error.new("The model is not setup complete.") unless setup_completed?
        check_xy_type(x, y)
        iter = Iterator.new(x, y)
        num_train_datas = x.shape[0]
        (1..epochs).each do |epoch|
          before_epoch_cbk&.call(epoch)
          puts "【 epoch #{epoch}/#{epochs} 】" if verbose
          iter.foreach(batch_size) do |x_batch, y_batch, index|
            loss_value = train_on_batch(x_batch, y_batch, before_train_on_batch_cbk: before_train_on_batch_cbk,
                                        after_train_on_batch_cbk: after_train_on_batch_cbk)
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
            acc, test_loss = accurate(test[0], test[1], batch_size: batch_size, before_test_on_batch_cbk: before_test_on_batch_cbk,
                                      after_test_on_batch_cbk: after_test_on_batch_cbk)
            print "  accurate: #{acc}, test loss: #{sprintf('%.8f', test_loss)}" if verbose
          end
          puts "" if verbose
          after_epoch_cbk&.call(epoch)
        end
      end

      alias fit train

      # Training once.
      # Setup the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @param [Lambda] before_train_on_batch_cbk Set the proc to be performed before train on batch processing.
      # @param [Lambda] after_train_on_batch_cbk Set the proc to be performed after train on batch processing.
      # @return [Float | Numo::SFloat] Return loss value in the form of Float or Numo::SFloat.
      def train_on_batch(x, y, before_train_on_batch_cbk: nil, after_train_on_batch_cbk: nil)
        raise DNN_Error.new("The model is not setup complete.") unless setup_completed?
        check_xy_type(x, y)
        before_train_on_batch_cbk&.call
        x = forward(x, true)
        loss_value = @loss_func.forward(x, y, layers)
        dy = @loss_func.backward(y, layers)
        backward(dy)
        @optimizer.update(layers.uniq)
        after_train_on_batch_cbk&.call(loss_value)
        loss_value
      end

      # Evaluate model and get accurate of test data.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @param [Lambda] before_test_on_batch_cbk Set the proc to be performed before test on batch processing.
      # @param [Lambda] after_test_on_batch_cbk Set the proc to be performed after test on batch processing.
      # @return [Array] Returns the test data accurate and mean loss in the form [accurate, mean_loss].
      def accurate(x, y, batch_size: 100, before_test_on_batch_cbk: nil, after_test_on_batch_cbk: nil)
        check_xy_type(x, y)
        batch_size = batch_size >= x.shape[0] ? x.shape[0] : batch_size
        iter = Iterator.new(x, y, random: false)
        total_correct = 0
        sum_loss = 0
        max_steps = (x.shape[0].to_f / batch_size).ceil
        iter.foreach(batch_size) do |x_batch, y_batch|
          correct, loss_value = test_on_batch(x_batch, y_batch, before_test_on_batch_cbk: before_test_on_batch_cbk,
                                              after_test_on_batch_cbk: after_test_on_batch_cbk)
          total_correct += correct
          sum_loss += loss_value.is_a?(Xumo::SFloat) ? loss_value.mean : loss_value
        end
        mean_loss = sum_loss / max_steps
        [total_correct.to_f / x.shape[0], mean_loss]
      end

      # Evaluate once.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @param [Lambda] before_test_on_batch_cbk Set the proc to be performed before test on batch processing.
      # @param [Lambda] after_test_on_batch_cbk Set the proc to be performed after test on batch processing.
      # @return [Array] Returns the test data accurate and mean loss in the form [accurate, mean_loss].
      def test_on_batch(x, y, before_test_on_batch_cbk: nil, after_test_on_batch_cbk: nil)
        before_test_on_batch_cbk&.call
        x = forward(x, false)
        correct = evaluate(x, y)
        loss_value = @loss_func.forward(x, y, layers)
        after_test_on_batch_cbk&.call(loss_value)
        [correct, loss_value]
      end

      private def evaluate(y, t)
        correct = 0
        y.shape[0].times do |i|
          if y.shape[1..-1] == [1]
            if @loss_func.is_a?(Losses::SigmoidCrossEntropy)
              correct += 1 if (y[i, 0] < 0 && t[i, 0] < 0.5) || (y[i, 0] >= 0 && t[i, 0] >= 0.5)
            else
              correct += 1 if (y[i, 0] < 0 && t[i, 0] < 0) || (y[i, 0] >= 0 && t[i, 0] >= 0)
            end
          else
            correct += 1 if y[i, true].max_index == t[i, true].max_index
          end
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
        layers
      end

      # Get the all has param layers.
      # @return [Array] All has param layers array.
      def has_param_layers
        layers.select { |layer| layer.is_a?(Layers::HasParamLayer) }
      end

      # Get the layer that the model has.
      # @overload get_layer(index)
      #   @param [Integer] The index of the layer to get.
      #   @return [DNN::Layers::Layer] Return the layer.
      # @overload get_layer(layer_class, index)
      #   @param [Integer] The index of the layer to get.
      #   @param [Class] The class of the layer to get.
      #   @return [DNN::Layers::Layer] Return the layer.
      def get_layer(*args)
        if args.length == 1
          index = args[0]
          layers[index]
        else
          layer_class, index = args
          layers.select { |layer| layer.is_a?(layer_class) }[index]
        end
      end

      # @return [DNN::Optimizers::Optimizer] optimizer Return the optimizer to use for learning.
      def optimizer
        raise DNN_Error.new("The model is not setup complete.") unless setup_completed?
        @optimizer
      end

      # @return [DNN::Losses::Loss] loss_func Return the loss function to use for learning.
      def loss_func
        raise DNN_Error.new("The model is not setup complete.") unless setup_completed?
        @loss_func
      end

      # @return [Boolean] If model have already been setup completed then return true.
      def setup_completed?
        @setup_completed
      end

      # @return [Boolean] If model have already been built then return true.
      def built?
        @built
      end

      private

      def forward(x, learning_phase)
        DNN.learning_phase = learning_phase
        y, @last_link = call([x, nil])
        unless @built
          @built = true
          tagging
        end
        y
      end

      def backward(dy)
        bwd = -> link, dy do
          return dy unless link
          if link.is_a?(TwoInputLink)
            dy1, dy2 = link.layer.backward(dy)
            [bwd.(link.prev1, dy1), bwd.(link.prev2, dy2)]
          else
            dy = link.layer.backward(dy)
            bwd.(link.prev, dy)
          end
        end
        bwd.(@last_link, dy)
      end

      def tagging
        target_layers = layers.uniq
        target_layers.each do |layer|
          id = target_layers.select { |l| l.is_a?(layer.class) }.index(layer)
          class_name = layer.class.name.split("::").last
          layer.tag = "#{class_name}_#{id}".to_sym
          if layer.is_a?(Layers::HasParamLayer)
            layer.get_params.each do |param_key, param|
              param.tag = "#{layer.tag}__#{param_key}".to_sym
            end
          end
        end
      end

      def check_xy_type(x, y = nil)
        unless x.is_a?(Xumo::SFloat)
          raise TypeError.new("x:#{x.class.name} is not an instance of #{Xumo::SFloat.name} class.")
        end
        if y && !y.is_a?(Xumo::SFloat)
          raise TypeError.new("y:#{y.class.name} is not an instance of #{Xumo::SFloat.name} class.")
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

      def call(x)
        @stack.each do |layer|
          x = layer.(x)
        end
        x
      end
    end

  end
end
