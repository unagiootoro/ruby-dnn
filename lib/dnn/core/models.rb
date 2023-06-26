module DNN
  module Models

    # This class is used to hold multiple layers in an array.
    class LayerList < Array
      def self.from_hash_list(hash_list)
        layers_list = new
        hash_list.each do |hash|
          obj_class = DNN.const_get(hash[:class])
          obj = obj_class.allocate
          if obj.is_a?(Chain)
            obj = obj_class.new
            obj.load_hash(hash)
          else
            obj = Layers::Layer.from_hash(hash)
          end
          layers_list << obj
        end
        layers_list
      end

      def to_hash_list
        map(&:to_hash)
      end

      # Get the all layers.
      # @return [Array] All layers array.
      def layers
        layers_array = []
        each do |layer|
          if layer.is_a?(Layers::Layer)
            layers_array << layer
          elsif layer.is_a?(Chain) || layer.is_a?(LayerList)
            layers_array.concat(layer.layers)
          end
        end
        layers_array
      end
    end

    class Chain
      def initialize
        @layers_cache = nil
      end

      # Forward propagation.
      # @param [Array] input_tensors Input tensors.
      # @return [Tensor] Output tensor.
      def forward(*input_tensors)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward'"
      end

      # Forward propagation and create a link.
      # @param [Array] input_tensors Input tensors.
      # @return [Tensor] Output tensor.
      def call(*input_tensors)
        forward(*input_tensors)
      end

      # @return [Boolean] learning_phase Specifies whether it is in the learning phase.
      def set_learning_phase(learning_phase)
        layers.each do |layer|
          layer.set_learning_phase(learning_phase)
        end
      end

      # @return [Boolean] Setting false prevents learning of parameters.
      def trainable?
        layers.each do |layer|
          return true if layer.trainable?
        end
        false
      end

      # @param [Boolean] trainable Specifies whether to allow learning.
      def set_trainable(trainable)
        layers.each do |layer|
          layer.set_trainable(trainable)
        end
      end

      # Get the all layers.
      # @return [Array] All layers array.
      def layers
        return @layers_cache if @layers_cache
        layers_array = []
        instance_variables.sort.each do |ivar|
          obj = instance_variable_get(ivar)
          if obj.is_a?(Layers::Layer)
            layers_array << obj
          elsif obj.is_a?(Chain) || obj.is_a?(LayerList)
            layers_array.concat(obj.layers)
          end
        end
        @layers_cache = layers_array
      end

      def to_hash
        layers_hash = { class: self.class.name }
        instance_variables.sort.each do |ivar|
          obj = instance_variable_get(ivar)
          if obj.is_a?(Layers::Layer) || obj.is_a?(Chain)
            layers_hash[ivar] = obj.to_hash
          elsif obj.is_a?(LayerList)
            layers_hash[ivar] = obj.to_hash_list
          end
        end
        layers_hash
      end

      def load_hash(layers_hash)
        instance_variables.sort.each do |ivar|
          hash_or_array = layers_hash[ivar]
          if hash_or_array.is_a?(Array)
            instance_variable_set(ivar, LayerList.from_hash_list(hash_or_array))
          elsif hash_or_array.is_a?(Hash)
            obj_class = DNN.const_get(hash_or_array[:class])
            obj = obj_class.allocate
            if obj.is_a?(Chain)
              obj = obj_class.new
              obj.load_hash(hash_or_array)
              instance_variable_set(ivar, obj)
            else
              instance_variable_set(ivar, Layers::Layer.from_hash(hash_or_array))
            end
          end
        end
      end
    end

    # This class deals with the model of the network.
    class Model < Chain
      attr_accessor :optimizer
      attr_accessor :loss_weights

      # Load marshal model.
      # @param [String] file_name File name of marshal model to load.
      # @return [DNN::Models::Model] Return the loaded model.
      def self.load(file_name)
        model = self.allocate
        loader = Loaders::MarshalLoader.new(model)
        loader.load(file_name)
        model
      end

      def initialize
        super
        @optimizer = nil
        @loss_func = nil
        @built = false
        @loss_weights = nil
      end

      def call(*input_tensors)
        output_tensors = forward(*input_tensors)
        @built = true unless @built
        output_tensors
      end

      # Set optimizer and loss_func to model.
      # @param [DNN::Optimizers::Optimizer] optimizer Optimizer to use for learning.
      # @param [DNN::Losses::Loss] loss_func Loss function to use for learning.
      # @param [Array | NilClass] loss_weights Setting loss weights contribution.
      def setup(optimizer, loss_func, loss_weights: nil)
        unless optimizer.is_a?(Optimizers::Optimizer)
          raise TypeError, "optimizer:#{optimizer.class} is not an instance of DNN::Optimizers::Optimizer class."
        end
        unless loss_func.is_a?(Losses::Loss) || loss_func.is_a?(Array)
          raise TypeError, "loss_func:#{loss_func.class} is not an instance of DNN::Losses::Loss or Array class."
        end
        @optimizer = optimizer
        self.loss_func = loss_func
        @loss_weights = loss_weights
      end

      def loss_func
        @loss_func
      end

      def loss_func=(lfs)
        if lfs.is_a?(Array)
          @loss_func = []
          lfs.each.with_index do |lf, i|
            unless lf.is_a?(Losses::Loss)
              raise TypeError, "loss_func[#{i}]:#{lf.class} is not an instance of DNN::Losses::Loss class."
            end
            @loss_func << lf
          end
        else
          @loss_func = lfs
        end
      end

      # Training once.
      # Setup the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @return [Float | Array] Return loss value in the form of Float or Array.
      def train_on_batch(x, y)
        raise DNNError, "The model is not optimizer setup complete." unless @optimizer
        raise DNNError, "The model is not loss_func setup complete." unless @loss_func
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        Utils.check_input_data_type("y", y, Xumo::SFloat)
        set_learning_phase(true)
        inputs = Tensor.convert(x)
        outputs = call(*inputs)
        losses = optimize(outputs, Tensor.convert(y))
        if losses.is_a?(Array)
          losses.map { |loss| Utils.to_f(loss.data) }
        else
          Utils.to_f(losses.data)
        end
      end

      # Update model parameters using output data and teacher data.
      # Setup the model before use this method.
      # @param [DNN::Tensor] y Output data or it array.
      # @param [DNN::Tensor] t Teacher data or it array.
      # @return [DNN::Tensor | Array] Return loss tensor or it array.
      def optimize(y, t)
        raise DNNError, "The model is not optimizer setup complete." unless @optimizer
        raise DNNError, "The model is not loss_func setup complete." unless @loss_func
        Utils.check_input_data_type("y", y, DNN::Tensor)
        Utils.check_input_data_type("t", t, DNN::Tensor)
        if y.is_a?(Array)
          result = []
          y.each_index do |i|
            loss_weight = @loss_weights ? @loss_weights[i] : nil
            loss = compute_train_loss(y[i], t[i], @loss_func[i], loss_weight)
            result << loss
            loss.backward(Xumo::SFloat.ones(y[i].data[0...1, false].shape[0], 1))
          end
        else
          loss = compute_train_loss(y, t, @loss_func)
          result = loss
          loss.backward(Xumo::SFloat.ones(y.data[0...1, false].shape[0], 1))
        end
        @optimizer.update(get_all_trainable_variables)
        result
      end

      private def compute_train_loss(y, t, loss_func, loss_weight = nil)
        unless y.shape == t.shape
          raise DNNShapeError, "The shape of y does not match the t shape. y shape is #{y.shape}, but t shape is #{t.shape}."
        end
        loss = loss_func.(y, t)
        loss *= loss_weight if loss_weight
        regularizers = layers.select { |layer| layer.respond_to?(:regularizers) }
                            .map(&:regularizers).flatten
        regularizers.each do |regularizer|
          loss = regularizer.(loss)
        end
        loss
      end

      # Test once.
      # @param [Numo::SFloat | Array] x Input test data.
      # @param [Numo::SFloat | Array] y Output test data.
      # @return [Float | Array] Return loss value in the form of Float or Array.
      def test_on_batch(x, y)
        raise DNNError, "The model is not loss_func setup complete." unless @loss_func
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        Utils.check_input_data_type("y", y, Xumo::SFloat)
        set_learning_phase(false)
        if x.is_a?(Array)
          input = x.map { |v| Tensor.new(v) }
        else
          input = Tensor.new(x)
        end
        output_tensors = call(*input)
        if output_tensors.is_a?(Array)
          loss_data = []
          output_tensors.each.with_index do |out, i|
            loss = @loss_func[i].(out, Tensor.new(y[i]))
            loss_data << loss.data
          end
          loss_data.map { |v| Utils.to_f(v) }
        else
          out = output_tensors
          loss = @loss_func.(out, Tensor.new(y))
          Utils.to_f(loss.data)
        end
      end

      # Implement the process to accuracy this model.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @return [Integer] Returns the test data accuracy.
      def accuracy(x, y)
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
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        set_learning_phase(false)
        out = call(Tensor.new(x))
        if out.is_a?(Array)
          out.map { |tensor| tensor.data }
        else
          out.data
        end
      end

      # Predict one data.
      # @param [Numo::SFloat] x Input data. However, x is single data.
      def predict1(x)
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        input = if x.is_a?(Array)
                  x.map { |v| v.reshape(1, *v.shape) }
                else
                  x.reshape(1, *x.shape)
                end
        y = predict(input)
        if y.is_a?(Array)
          y.map { |v| v[0, false] }
        else
          y[0, false]
        end
      end

      # Load marshal params.
      # @param [String] file_name File name of marshal model to load.
      def load_params(file_name)
        loader = Loaders::MarshalLoader.new(self)
        loader.load(file_name)
      end

      # Save the model in marshal format.
      # @param [String] file_name Name to save model.
      def save(file_name)
        saver = Savers::MarshalSaver.new(self, include_model: true)
        saver.save(file_name)
      end

      # Save the params in marshal format.
      # @param [String] file_name Name to save model.
      def save_params(file_name)
        saver = Savers::MarshalSaver.new(self, include_model: false)
        saver.save(file_name)
      end

      # @return [DNN::Models::Model] Return the copy this model.
      def copy
        Marshal.load(Marshal.dump(self))
      end

      # Get the layer that the model has.
      # @param [Symbol] name The name of the layer to get.
      # @return [DNN::Layers::Layer] Return the layer.
      def get_layer(name)
        layer = instance_variable_get("@#{name}")
        return layer if layer.is_a?(Layers::Layer) || layer.is_a?(Chain) || layer.is_a?(LayerList)
        nil
      end

      # @return [Boolean] If model have already been built then return true.
      def built?
        @built
      end

      # Clean all layers.
      def clean_layers
        layers.each(&:clean)
        if @loss_func.is_a?(Array)
          @loss_func.each do |lf|
            lf.clean
          end
        elsif @loss_func.is_a?(Losses::Loss)
          @loss_func.clean
        end
        @layers_cache = nil
      end

      # Get parameter data of all layers.
      # @return [Array] Parameter data.
      def get_all_params_data
        layers.map do |layer|
          layer.get_variables.to_h do |key, param|
            [key, param.data]
          end
        end
      end

      # Set parameter data of all layers.
      # @param [Array] params_data Parameter data obtained by get_all_params_data.
      def set_all_params_data(params_data)
        layers.each.with_index do |layer, i|
          params_data[i].each do |(key, data)|
            layer.get_variables[key].data = data
          end
        end
      end

      # Convert the parameters of model and optimizer for cpu.
      # @return [DNN::Models::Model] Return self.
      def to_cpu
        params_data = get_all_params_data
        clean_layers
        set_all_params_data(params_data)
        layers.each do |layer|
          layer.get_variables.each do |key, param|
            data = param.data
            if DNN.use_cumo? && data.is_a?(Cumo::NArray)
              param.data = Utils.cumo2numo(data)
            end
          end
        end
        @optimizer.status.each do |key, state|
          next unless state
          state.each do |param, data|
            if DNN.use_cumo? && data.is_a?(Cumo::NArray)
              state[param] = Utils.cumo2numo(data)
            end
          end
        end
        self
      end

      # Convert the parameters of model and optimizer for gpu.
      # @return [DNN::Models::Model] Return self.
      def to_gpu
        params_data = get_all_params_data
        clean_layers
        set_all_params_data(params_data)
        layers.each do |layer|
          layer.get_variables.each do |(key, param)|
            data = param.data
            if DNN.use_cumo? && data.is_a?(Numo::NArray)
              param.data = Utils.numo2cumo(data)
            end
          end
        end
        @optimizer.status.each do |(key, state)|
          next unless state
          state.each do |(param, data)|
            if DNN.use_cumo? && data.is_a?(Numo::NArray)
              state[param] = Utils.numo2cumo(data)
            end
          end
        end
        self
      end

      def get_all_trainable_variables
        layers.flat_map { |layer| layer.get_trainable_variables.values }.compact.select(&:grad)
      end
    end

    class Sequential < Model
      attr_reader :stack

      # @param [Array] stack All layers possessed by the model.
      def initialize(stack = [])
        super()
        @stack = LayerList.new
        stack.each do |layer|
          add(layer)
        end
      end

      # Add layer to the model.
      # @param [DNN::Layers::Layer | DNN::Models::Chain] layer Layer or Chain to add to the model.
      # @return [DNN::Models::Model] Return self.
      def add(layer)
        unless layer.is_a?(Layers::Layer) || layer.is_a?(Chain)
          raise TypeError, "layer: #{layer.class.name} is not an instance of the DNN::Layers::Layer class or DNN::Models::Chain class."
        end
        @stack << layer
        self
      end

      alias << add

      # Insert layer to the model by index position.
      # @param [DNN::Layers::Layer | DNN::Models::Chain] layer Layer or Chain to add to the model.
      # @return [DNN::Models::Model] Return self.
      def insert(index, layer)
        unless layer.is_a?(Layers::Layer) || layer.is_a?(Chain)
          raise TypeError, "layer: #{layer.class.name} is not an instance of the DNN::Layers::Layer class or DNN::Models::Chain class."
        end
        @stack.insert(index, layer)
        self
      end

      # Remove layer to the model.
      # @param [DNN::Layers::Layer | DNN::Models::Chain] layer Layer to remove to the model.
      # @return [Boolean] Return true if success for remove layer.
      def remove(layer)
        @stack.delete(layer) ? true : false
      end

      def forward(x)
        @stack.each do |layer|
          x = layer.(x)
        end
        x
      end
    end

  end
end
