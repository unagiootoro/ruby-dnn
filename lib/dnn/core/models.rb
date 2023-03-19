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
      attr_reader :last_log

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
        @callbacks = []
        @last_log = {}
        @early_stop_requested = false
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

      # Start training.
      # Setup the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @param [Integer] epochs Number of training.
      # @param [Integer] batch_size Batch size used for one training.
      # @param [Integer] initial_epoch Initial epoch.
      # @param [Array | NilClass] test If you to test the model for every 1 epoch,
      #                                specify [x_test, y_test]. Don't test to the model, specify nil.
      # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
      # @param [Boolean] accuracy Set true to compute the accuracy.
      # @param [IO] io Specifies the IO object to use for logging.
      def train(x, y, epochs,
                batch_size: 1,
                initial_epoch: 1,
                test: nil,
                verbose: true,
                accuracy: true,
                io: $stdout)
        trainer = ModelTrainer.new(self)
        trainer.start_train(x, y, epochs,
                            batch_size: batch_size,
                            initial_epoch: initial_epoch,
                            test: test,
                            verbose: verbose,
                            accuracy: accuracy,
                            io: io)
        trainer.update while trainer.training?
      end

      alias fit train

      # Start training by iterator.
      # Setup the model before use this method.
      # @param [DNN::Iterator] train_iterator Iterator used for training.
      # @param [Integer] epochs Number of training.
      # @param [Integer] batch_size Batch size used for one training.
      # @param [Integer] initial_epoch Initial epoch.
      # @param [Array | NilClass] test If you to test the model for every 1 epoch,
      #                                specify [x_test, y_test]. Don't test to the model, specify nil.
      # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
      # @param [Boolean] accuracy Set true to compute the accuracy.
      # @param [IO] io Specifies the IO object to use for logging.
      def train_by_iterator(train_iterator, epochs,
                            batch_size: 1,
                            initial_epoch: 1,
                            test: nil,
                            verbose: true,
                            accuracy: true,
                            io: $stdout)
        trainer = ModelTrainer.new(self)
        trainer.start_train_by_iterator(train_iterator, epochs,
                                        batch_size: batch_size,
                                        initial_epoch: initial_epoch,
                                        test: test,
                                        verbose: verbose,
                                        accuracy: accuracy,
                                        io: io)
        trainer.update while trainer.training?
      end

      alias fit_by_iterator train_by_iterator

      # Implement the training process to be performed in one step.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @return [Hash] Hash of contents to be output to log.
      def train_step(x, y, accuracy: false)
        output_data, loss_data = train_on_batch_internal(x, y)
        if loss_data.is_a?(Array)
          loss_value = []
          accuracy = []
          loss_data.each_index do |i|
            loss_value << Utils.to_f(loss_data)
            accuracy << accuracy(output_data[i], y[i]).to_f / y[i].shape[0]
          end
        else
          loss_value = Utils.to_f(loss_data)
          accuracy = accuracy(output_data, y).to_f / y.shape[0]
        end
        { loss: loss_value, accuracy: accuracy }
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
        *, loss_data = train_on_batch_internal(x, y)
        if loss_data.is_a?(Array)
          loss_data.map { |v| Utils.to_f(v) }
        else
          Utils.to_f(loss_data)
        end
      end

      private def train_on_batch_internal(x, y)
        DNN.learning_phase = true
        output_tensors = call(Tensor.convert(x))
        if output_tensors.is_a?(Array)
          output_data = []
          loss_data = []
          output_tensors.each.with_index do |out, i|
            output_data << out.data
            loss_opt = {}
            loss_opt[:layers] = layers if i == 0
            loss_opt[:loss_weight] = @loss_weights[i] if @loss_weights
            loss = @loss_func[i].loss(out, Tensor.convert(y[i]), **loss_opt)
            loss_data << loss.data
            loss.link.backward(Xumo::SFloat.ones(y[i][0...1, false].shape[0], 1))
          end
        else
          out = output_tensors
          output_data = out.data
          loss = @loss_func.loss(out, Tensor.convert(y), layers: layers)
          loss_data = loss.data
          loss.link.backward(Xumo::SFloat.ones(y[0...1, false].shape[0], 1))
        end
        @optimizer.update(get_all_trainable_params)
        [output_data, loss_data]
      end

      # Evaluate model and get accuracy and loss of test data.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @param [Integer] batch_size Batch size used for one test.
      # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
      #                 If accuracy is not needed returns in the form [nil, mean_loss].
      def evaluate(x, y, batch_size: 100, accuracy: true)
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        Utils.check_input_data_type("y", y, Xumo::SFloat)
        evaluator = ModelEvaluator.new(self)
        evaluator.start_evaluate(x, y, batch_size: batch_size, accuracy: accuracy)
        evaluator.update while evaluator.evaluating?
        [@last_log[:test_accuracy], @last_log[:test_loss]]
      end

      # Evaluate model by iterator.
      # @param [DNN::Iterator] test_iterator Iterator used for testing.
      # @param [Integer] batch_size Batch size used for one test.
      # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
      #                 If accuracy is not needed returns in the form [nil, mean_loss].
      def evaluate_by_iterator(test_iterator, batch_size: 100, accuracy: true)
        evaluator = ModelEvaluator.new(self)
        evaluator.start_evaluate_by_iterator(test_iterator, batch_size: batch_size, accuracy: accuracy)
        evaluator.update while evaluator.evaluating?
        [@last_log[:test_accuracy], @last_log[:test_loss]]
      end

      # Testing process to be performed in one step.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @return [Hash] Hash of contents to be output to log.
      def test_step(x, y, accuracy: false)
        output_data, loss_data = test_on_batch_internal(x, y)
        if loss_data.is_a?(Array)
          loss_value = []
          accuracy = []
          loss_data.each_index do |i|
            loss_value << Utils.to_f(loss_data)
            accuracy << accuracy(output_data[i], y[i]).to_f / y[i].shape[0]
          end
        else
          loss_value = Utils.to_f(loss_data)
        end
        { test_loss: loss_value, test_accuracy: accuracy(output_data, y) }
      end

      # Test once.
      # @param [Numo::SFloat | Array] x Input test data.
      # @param [Numo::SFloat | Array] y Output test data.
      # @return [Float | Array] Return loss value in the form of Float or Array.
      def test_on_batch(x, y)
        raise DNNError, "The model is not loss_func setup complete." unless @loss_func
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        Utils.check_input_data_type("y", y, Xumo::SFloat)
        *, loss_data = test_on_batch_internal(x, y)
        if loss_data.is_a?(Array)
          loss_data.map { |v| Utils.to_f(v) }
        else
          Utils.to_f(loss_data)
        end
      end

      private def test_on_batch_internal(x, y)
        DNN.learning_phase = false
        output_tensors = call(Tensor.convert(x))
        if output_tensors.is_a?(Array)
          output_data = []
          loss_data = []
          output_tensors.each.with_index do |out, i|
            output_data << out.data
            loss = @loss_func[i].(out, Tensor.convert(y[i]))
            loss_data << loss.data
          end
        else
          out = output_tensors
          output_data = out.data
          loss = @loss_func.(out, Tensor.convert(y))
          loss_data = loss.data
        end
        [output_data, loss_data]
      end

      # Implement the process to accuracy this model.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @return [Integer] Returns the test data accuracy.
      private def accuracy(x, y)
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
        output_tensors = call(Tensor.convert(x))
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

      # Add callback function.
      # @param [Callback] callback Callback object.
      def add_callback(callback)
        callback.model = self
        @callbacks << callback
      end

      # Add lambda callback.
      # @param [Symbol] event Event to execute callback.
      # @yield Register the contents of the callback.
      def add_lambda_callback(event, &block)
        callback = Callbacks::LambdaCallback.new(event, &block)
        callback.model = self
        @callbacks << callback
      end

      # Clear the callback function registered for each event.
      def clear_callbacks
        @callbacks = []
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
            layer.get_params[key].data = data
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

      # Request training early stop.
      def request_early_stop
        @early_stop_requested = true
      end

      def check_early_stop_requested
        if @early_stop_requested
          @early_stop_requested = false
          return true
        end
        false
      end

      def get_all_trainable_params
        layers.select { |layer| layer.is_a?(Layers::TrainableLayer) && layer.trainable }
              .map { |layer| layer.get_params.values }.flatten.compact
              .select(&:grad)
      end

      def call_callbacks(event)
        @callbacks.each do |callback|
          callback.send(event) if callback.respond_to?(event)
        end
      end
    end

    class ModelTrainer
      def initialize(model)
        @model = model
        @state = :none
        @initial_epoch = 1
        @step = 1
        @max_steps = 1
        @train_iterator = nil
        @max_epochs = 1
        @batch_size = 1
        @epoch = 1
        @test = nil
        @verbose = false
        @accuracy = false
        @io = nil
        @num_train_datas = 0
      end

      # Start training.
      # Setup the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @param [Integer] epochs Number of training.
      # @param [Integer] batch_size Batch size used for one training.
      # @param [Integer] initial_epoch Initial epoch.
      # @param [Array | NilClass] test If you to test the model for every 1 epoch,
      #                                specify [x_test, y_test]. Don't test to the model, specify nil.
      # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
      # @param [Boolean] accuracy Set true to compute the accuracy.
      # @param [IO] io Specifies the IO object to use for logging.
      def start_train(x, y, epochs,
                      batch_size: 1,
                      initial_epoch: 1,
                      test: nil,
                      verbose: true,
                      accuracy: true,
                      io: $stdout)
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        Utils.check_input_data_type("y", y, Xumo::SFloat)
        train_iterator = Iterator.new(x, y)
        start_train_by_iterator(train_iterator, epochs,
                                batch_size: batch_size,
                                initial_epoch: initial_epoch,
                                test: test,
                                verbose: verbose,
                                accuracy: accuracy,
                                io: io)
      end

      # Start training by iterator.
      # Setup the model before use this method.
      # @param [DNN::Iterator] train_iterator Iterator used for training.
      # @param [Integer] epochs Number of training.
      # @param [Integer] batch_size Batch size used for one training.
      # @param [Integer] initial_epoch Initial epoch.
      # @param [Array | NilClass] test If you to test the model for every 1 epoch,
      #                                specify [x_test, y_test]. Don't test to the model, specify nil.
      # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
      # @param [Boolean] accuracy Set true to compute the accuracy.
      # @param [IO] io Specifies the IO object to use for logging.
      def start_train_by_iterator(train_iterator, epochs,
                                  batch_size: 1,
                                  initial_epoch: 1,
                                  test: nil,
                                  verbose: true,
                                  accuracy: true,
                                  io: $stdout)
        raise DNNError, "The model is not optimizer setup complete." unless @model.optimizer
        raise DNNError, "The model is not loss_func setup complete." unless @model.loss_func
        @model.check_early_stop_requested # Clear early stop request.
        @train_iterator = train_iterator
        @max_epochs = epochs
        @batch_size = batch_size
        @epoch = initial_epoch
        @test = test
        @verbose = verbose
        @accuracy = accuracy
        @io = io
        @state = :start_epoch
        @max_steps = train_iterator.max_steps(batch_size)
        @num_train_datas = train_iterator.num_usable_datas(batch_size)
        @line_first_pos = 0
        @model.call_callbacks(:before_train)
      end

      # Check if it is currently evaluating.
      # @return [Boolean] Returns true if currently training.
      def training?
        @state != :none
      end

      # Update trainer.
      def update
        case @state
        when :start_epoch
          start_epoch
        when :start_step
          start_step
        when :train_step
          train_step
        when :end_step
          end_step
        when :end_epoch
          end_epoch
        when :start_evaluate
          start_evaluate
        when :evaluating
          evaluating
        when :end_evaluate
          end_evaluate
        when :end_training
          end_training
        end
      end

      private

      def start_epoch
        @model.last_log[:epoch] = @epoch
        @model.call_callbacks(:before_epoch)
        @io.puts "【 epoch #{@epoch}/#{@max_epochs} 】" if @verbose
        @step = 1
        @state = :start_step
      end

      def start_step
        @model.last_log[:step] = @step
        @state = :train_step
      end

      def train_step
        (x_batch, y_batch) = @train_iterator.next_batch(@batch_size)
        @model.call_callbacks(:before_train_on_batch)
        train_step_met = @model.train_step(x_batch, y_batch)
        @model.last_log.merge!(train_step_met)
        @model.call_callbacks(:after_train_on_batch)
        num_trained_datas = @step * @batch_size
        num_trained_datas = num_trained_datas > @num_train_datas ? @num_train_datas : num_trained_datas
        if @io == $stdout
          log = "\r"
        else
          @line_first_pos = @io.pos
          log = ""
        end
        40.times do |i|
          if i < num_trained_datas * 40 / @num_train_datas
            log << "="
          elsif i == num_trained_datas * 40 / @num_train_datas
            log << ">"
          else
            log << "_"
          end
        end
        log << "  #{num_trained_datas}/#{@num_train_datas} "
        log << metrics_to_str(train_step_met)
        @io.print log if @verbose
        if @model.check_early_stop_requested
          @io.puts("\nEarly stopped.") if @verbose
          @state = :end_training
        else
          @state = :end_step
        end
      end

      def end_step
        @step += 1
        if @step <= @max_steps
          unless @io == $stdout
            @io.pos = @line_first_pos
          end
          @state = :start_step
        else
          @state = :end_epoch
        end
      end

      def end_epoch
        @epoch += 1
        if @test
          @state = :start_evaluate
        else
          @io.puts "" if @verbose
          @model.call_callbacks(:after_epoch)
          if @epoch <= @max_epochs
            @train_iterator.reset
            @state = :start_epoch
          else
            @state = :none
          end
        end
      end

      def start_evaluate
        @evaluator = ModelEvaluator.new(@model)
        if @test.is_a?(Array)
          @evaluator.start_evaluate(@test[0], @test[1], batch_size: @batch_size, accuracy: @accuracy)
        else
          @evaluator.start_evaluate_by_iterator(@test, batch_size: @batch_size, accuracy: @accuracy)
        end
        @state = :evaluating
      end

      def evaluating
        @evaluator.update
        unless @evaluator.evaluating?
          @state = :end_evaluate
        end
      end

      def end_evaluate
        if @verbose
          metrics = if @accuracy
                      { test_accuracy: @model.last_log[:test_accuracy], test_loss: @model.last_log[:test_loss] }
                    else
                      { test_loss: @model.last_log[:test_loss] }
                    end
          @io.print "  " + metrics_to_str(metrics)
        end
        @io.puts "" if @verbose
        @model.call_callbacks(:after_epoch)
        if @epoch <= @max_epochs
          @train_iterator.reset
          if @model.check_early_stop_requested
            @io.puts("Early stopped.") if @verbose
            @state = :end_training
          else
            @state = :start_epoch
          end
        else
          @state = :end_training
        end
      end

      def end_training
        @model.call_callbacks(:after_train)
        @state = :none
      end

      def metrics_to_str(mertics)
        mertics.map { |key, values|
          str_values = if values.is_a?(Array)
                         values_fmt = values.map { |v| sprintf('%.4f', Utils.to_f(v)) }
                         "[#{values_fmt.join(", ")}]"
                       else
                         sprintf('%.4f', Utils.to_f(values))
                       end
          "#{key}: #{str_values}"
        }.join(", ")
      end
    end

    class ModelEvaluator
      def initialize(model)
        @model = model
        @state = :none
      end

      # Start evaluate model and get accuracy and loss of test data.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @param [Integer] batch_size Batch size used for one test.
      # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
      #                 If accuracy is not needed returns in the form [nil, mean_loss].
      def start_evaluate(x, y, batch_size: 100, accuracy: true)
        Utils.check_input_data_type("x", x, Xumo::SFloat)
        Utils.check_input_data_type("y", y, Xumo::SFloat)
        start_evaluate_by_iterator(Iterator.new(x, y, random: false), batch_size: batch_size, accuracy: accuracy)
      end

      # Start Evaluate model by iterator.
      # @param [DNN::Iterator] test_iterator Iterator used for testing.
      # @param [Integer] batch_size Batch size used for one test.
      # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
      #                 If accuracy is not needed returns in the form [nil, mean_loss].
      def start_evaluate_by_iterator(test_iterator, batch_size: 100, accuracy: true)
        @test_iterator = test_iterator
        @num_test_datas = test_iterator.num_datas
        @batch_size = batch_size >= @num_test_datas ? @num_test_datas : batch_size
        @accuracy = accuracy
        if @loss_func.is_a?(Array)
          @total_correct = Array.new(@loss_func.length, 0)
          @sum_loss = Array.new(@loss_func.length, 0)
        else
          @total_correct = 0
          @sum_loss = 0
        end
        @step = 1
        @max_steps = (@num_test_datas.to_f / @batch_size).ceil
        @state = :start_step
      end

      # Check if it is currently evaluating.
      # @return [Boolean] Returns true if currently evaluating.
      def evaluating?
        @state != :none
      end

      # Update evaluator.
      def update
        case @state
        when :start_step
          start_step
        when :test_step
          test_step
        when :end_step
          end_step
        when :end_evaluate
          end_evaluate
        end
      end

      private

      def start_step
        @model.last_log[:step] = @step
        @state = :test_step
      end

      def test_step
        (x_batch, y_batch) = @test_iterator.next_batch(@batch_size)
        @model.call_callbacks(:before_test_on_batch)
        test_met = @model.test_step(x_batch, y_batch, accuracy: @accuracy)
        @model.call_callbacks(:after_test_on_batch)
        if @loss_func.is_a?(Array)
          @loss_func.each_index do |i|
            @total_correct[i] += test_met[:test_accuracy][i] if @accuracy
            @sum_loss[i] += test_met[:test_loss][i]
          end
        else
          @total_correct += test_met[:test_accuracy] if @accuracy
          @sum_loss += test_met[:test_loss]
        end
        @state = :end_step
      end

      def end_step
        @step += 1
        if @step <= @max_steps
          @state = :start_step
        else
          @state = :end_evaluate
        end
      end

      def end_evaluate
        acc = nil
        if @loss_func.is_a?(Array)
          mean_loss = Array.new(@loss_func.length, 0)
          acc = Array.new(@loss_func.length, 0) if accuracy
          @loss_func.each_index do |i|
            mean_loss[i] += @sum_loss[i] / @max_steps
            acc[i] += @total_correct[i].to_f / @num_test_datas if @accuracy
          end
        else
          mean_loss = @sum_loss / @max_steps
          acc = @total_correct.to_f / @num_test_datas if @accuracy
        end
        @model.last_log[:test_loss] = mean_loss
        @model.last_log[:test_accuracy] = acc
        @state = :none
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
