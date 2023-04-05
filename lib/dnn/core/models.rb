module DNN
  module Models

    # This class is used to hold multiple layers in an array.
    class LayersList < Array
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
          elsif layer.is_a?(Chain) || layer.is_a?(LayersList)
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
      # @param [Tensor] input_tensors Input tensors.
      # @return [Tensor] Output tensor.
      def forward(input_tensors)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward'"
      end

      # Forward propagation and create a link.
      # @param [Tensor | Array] input_tensors Input tensors.
      # @return [Tensor] Output tensor.
      def call(input_tensors)
        forward(input_tensors)
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
          elsif obj.is_a?(Chain) || obj.is_a?(LayersList)
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
          elsif obj.is_a?(LayersList)
            layers_hash[ivar] = obj.to_hash_list
          end
        end
        layers_hash
      end

      def load_hash(layers_hash)
        instance_variables.sort.each do |ivar|
          hash_or_array = layers_hash[ivar]
          if hash_or_array.is_a?(Array)
            instance_variable_set(ivar, LayersList.from_hash_list(hash_or_array))
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

      def call(input_tensors)
        output_tensors = forward(input_tensors)
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
        DNN.learning_phase = true
        output_tensors = call(Tensor.new(x))
        if output_tensors.is_a?(Array)
          loss_data = []
          output_tensors.each.with_index do |out, i|
            loss_opt = {}
            loss_opt[:layers] = layers if i == 0
            loss_opt[:loss_weight] = @loss_weights[i] if @loss_weights
            loss = @loss_func[i].loss(out, Tensor.new(y[i]), **loss_opt)
            loss_data << loss.data
            loss.backward(Xumo::SFloat.ones(y[i][0...1, false].shape[0], 1))
          end
        else
          out = output_tensors
          loss = @loss_func.loss(out, Tensor.new(y), layers: layers)
          loss_data = loss.data
          loss.backward(Xumo::SFloat.ones(y[0...1, false].shape[0], 1))
        end
        @optimizer.update(get_all_trainable_params)
        if loss_data.is_a?(Array)
          loss_data.map { |v| Utils.to_f(v) }
        else
          Utils.to_f(loss_data)
        end
      end

      # Test once.
      # @param [Numo::SFloat | Array] x Input test data.
      # @param [Numo::SFloat | Array] y Output test data.
      # @return [Float | Array] Return loss value in the form of Float or Array.
      def test_on_batch(x, y)
        raise DNNError, "The model is not loss_func setup complete." unless @loss_func
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        Utils.check_input_data_type("y", y, Xumo::SFloat)
        DNN.learning_phase = false
        output_tensors = call(Tensor.new(x))
        if output_tensors.is_a?(Array)
          loss_data = []
          output_tensors.each.with_index do |out, i|
            output_data << out.data
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
      # @param [Boolean] use_loss_activation Use loss activation when loss has an activation.
      def predict(x, use_loss_activation: true)
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        DNN.learning_phase = false
        output_tensors = call(Tensor.new(x))
        if output_tensors.is_a?(Array)
          lfs = @loss_func
          ary_output_tensors = output_tensors
        else
          lfs = [@loss_func]
          ary_output_tensors = [output_tensors]
        end
        ys = []
        ary_output_tensors.each.with_index do |out, i|
          y = out.data
          lf = lfs[i]
          if use_loss_activation && lf && lf.class.respond_to?(:activation)
            y = lf.class.activation(y)
          end
          ys << y
        end
        output_tensors.is_a?(Array) ? ys : ys.first
      end

      # Predict one data.
      # @param [Numo::SFloat] x Input data. However, x is single data.
      def predict1(x, use_loss_activation: true)
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        input = if x.is_a?(Array)
                  x.map { |v| v.reshape(1, *v.shape) }
                else
                  x.reshape(1, *x.shape)
                end
        y = predict(input, use_loss_activation: use_loss_activation)
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

      # Get the all trainable layers.
      # @return [Array] All has param layers array.
      def trainable_layers
        layers.select { |layer| layer.is_a?(Layers::TrainableLayer) }
      end

      # Get the layer that the model has.
      # @param [Symbol] name The name of the layer to get.
      # @return [DNN::Layers::Layer] Return the layer.
      def get_layer(name)
        layer = instance_variable_get("@#{name}")
        return layer if layer.is_a?(Layers::Layer) || layer.is_a?(Chain) || layer.is_a?(LayersList)
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
        trainable_layers.map do |layer|
          layer.get_params.to_h do |key, param|
            [key, param.data]
          end
        end
      end

      # Set parameter data of all layers.
      # @param [Array] params_data Parameter data obtained by get_all_params_data.
      def set_all_params_data(params_data)
        trainable_layers.each.with_index do |layer, i|
          params_data[i].each do |(key, data)|
            layer.get_params[key].assign(data)
          end
        end
      end

      # Convert the parameters of model and optimizer for cpu.
      # @return [DNN::Models::Model] Return self.
      def to_cpu
        params_data = get_all_params_data
        clean_layers
        set_all_params_data(params_data)
        trainable_layers.each do |layer|
          layer.get_params.each do |key, param|
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
        trainable_layers.each do |layer|
          layer.get_params.each do |(key, param)|
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

      def get_all_trainable_params
        layers.select { |layer| layer.is_a?(Layers::TrainableLayer) && layer.trainable }
              .map { |layer| layer.get_params.values }.flatten.compact
              .select(&:grad)
      end
    end

    class Sequential < Model
      attr_reader :stack

      # @param [Array] stack All layers possessed by the model.
      def initialize(stack = [])
        super()
        @stack = LayersList.new
        stack.each do |layer|
          add(layer)
        end
      end

      # Add layer to the model.
      # @param [DNN::Layers::Layer | DNN::Models::Chain] layer Layer or Chain to add to the model.
      # @return [DNN::Models::Model] Return self.
      def add(layer)
        if layer.is_a?(Layers::MergeLayer)
          raise TypeError, "layer: #{layer.class.name} should not be a DNN::Layers::MergeLayer class."
        end
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
        if layer.is_a?(Layers::MergeLayer)
          raise TypeError, "layer: #{layer.class.name} should not be a DNN::Layers::MergeLayer class."
        end
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

    class FixedModel < Model
      attr_reader :layers

      def initialize(output_tensor, layers)
        super()
        @input_link = get_input_link(output_tensor.link)
        @layers = layers
      end

      def forward(input_tensors)
        if input_tensors.is_a?(Array)
          input_tensors.each do |tensor|
            @input_link.forward(tensor)
          end
        else
          @input_link.forward(input_tensors)
        end
      end

      private

      def get_input_link(last_link)
        get_input_link = -> link do
          if link.is_a?(Link)
            return link unless link.prev
            get_input_link.(link.prev)
          else
            return link unless link.prev1
            get_input_link.(link.prev1)
          end
        end
        get_input_link.(last_link)
      end
    end

  end
end
