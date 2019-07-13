require "zlib"
require "json"
require "base64"

module DNN
  module Models

    class BaseModel
      # @return [Bool] Setting false prevents learning of parameters.
      attr_accessor :trainable

      # Load marshal model.
      # @param [String] file_name File name of marshal model to load.
      def self.load(file_name)
        Marshal.load(Zlib::Inflate.inflate(File.binread(file_name)))
      end

      def initialize
        @trainable = true
        @optimizer = nil
      end

      # Load json model parameters.
      # @param [String] json_str json string to load model parameters.
      def load_json_params(json_str)
        hash = JSON.parse(json_str, symbolize_names: true)
        has_param_layers_params = hash[:params]
        has_param_layers_index = 0
        has_param_layers = get_all_layers.select { |layer| layer.is_a?(Layers::HasParamLayer) }.uniq
        has_param_layers.each do |layer|
          hash_params = has_param_layers_params[has_param_layers_index]
          hash_params.each do |key, (shape, base64_param)|
            bin = Base64.decode64(base64_param)
            data = Xumo::SFloat.from_binary(bin).reshape(*shape)
            layer.get_params[key].data = data
          end
          has_param_layers_index += 1
        end
      end
      
      # Convert model parameters to json string.
      # @return [String] json string.
      def params_to_json
        has_param_layers = get_all_layers.select { |layer| layer.is_a?(Layers::HasParamLayer) }.uniq
        has_param_layers_params = has_param_layers.map do |layer|
          layer.get_params.map { |key, param|
            base64_data = Base64.encode64(param.data.to_binary)
            [key, [param.data.shape, base64_data]]
          }.to_h
        end
        hash = {version: VERSION, params: has_param_layers_params}
        JSON.dump(hash)
      end

      # Set optimizer and loss_func to model and recompile. But does not build layers.
      # @param [DNN::Optimizers::Optimizer] optimizer Optimizer to use for learning.
      # @param [DNN::Losses::Loss] loss_func Loss function to use for learning.
      def setup(optimizer, loss_func)
        unless optimizer.is_a?(Optimizers::Optimizer)
          raise TypeError.new("optimizer:#{optimizer.class} is not an instance of DNN::Optimizers::Optimizer class.")
        end
        unless loss_func.is_a?(Losses::Loss)
          raise TypeError.new("loss_func:#{loss_func.class} is not an instance of DNN::Losses::Loss class.")
        end
        @compiled = true
        @optimizer = optimizer
        @loss_func = loss_func
      end

      # Start training.
      # Compile the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @param [Integer] epochs Number of training.
      # @param [Integer] batch_size Batch size used for one training.
      # @param [Array or NilClass] test If you to test the model for every 1 epoch,
      #                            specify [x_test, y_test]. Don't test to the model, specify nil.
      # @param [Bool] verbose Set true to display the log. If false is set, the log is not displayed.
      # @param [Lambda] before_epoch_cbk Process performed before one training.
      # @param [Lambda] after_epoch_cbk Process performed after one training.
      # @param [Lambda] before_batch_cbk Set the proc to be performed before batch processing.
      # @param [Lambda] after_batch_cbk Set the proc to be performed after batch processing.
      def train(x, y, epochs,
                batch_size: 1,
                test: nil,
                verbose: true,
                before_epoch_cbk: nil,
                after_epoch_cbk: nil,
                before_batch_cbk: nil,
                after_batch_cbk: nil)
        raise DNN_Error.new("The model is not compiled.") unless compiled?
        check_xy_type(x, y)
        dataset = Dataset.new(x, y)
        num_train_datas = x.shape[0]
        (1..epochs).each do |epoch|
          before_epoch_cbk.call(epoch) if before_epoch_cbk
          puts "【 epoch #{epoch}/#{epochs} 】" if verbose
          (num_train_datas.to_f / batch_size).ceil.times do |index|
            x_batch, y_batch = dataset.next_batch(batch_size)
            loss_value = train_on_batch(x_batch, y_batch,
                                        before_batch_cbk: before_batch_cbk, after_batch_cbk: after_batch_cbk)
            if loss_value.is_a?(Numo::SFloat)
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
          if verbose && test
            acc, test_loss = accurate(test[0], test[1], batch_size,
                                      before_batch_cbk: before_batch_cbk, after_batch_cbk: after_batch_cbk)
            print "  accurate: #{acc}, test loss: #{sprintf('%.8f', test_loss)}"
          end
          puts "" if verbose
          after_epoch_cbk.call(epoch) if after_epoch_cbk
        end
      end
    
      # Training once.
      # Compile the model before use this method.
      # @param [Numo::SFloat] x Input training data.
      # @param [Numo::SFloat] y Output training data.
      # @param [Lambda] before_batch_cbk Set the proc to be performed before batch processing.
      # @param [Lambda] after_batch_cbk Set the proc to be performed after batch processing.
      # @return [Float | Numo::SFloat] Return loss value in the form of Float or Numo::SFloat.
      def train_on_batch(x, y, before_batch_cbk: nil, after_batch_cbk: nil)
        raise DNN_Error.new("The model is not compiled.") unless compiled?
        check_xy_type(x, y)
        # input_data_shape_check(x, y)
        before_batch_cbk.call(true) if before_batch_cbk
        x = forward(x, true)
        loss_value = @loss_func.forward(x, y, get_all_layers)
        dy = @loss_func.backward(y, get_all_layers)
        backward(dy)
        update
        after_batch_cbk.call(loss_value, true) if after_batch_cbk
        loss_value
      end
    
      # Evaluate model and get accurate of test data.
      # @param [Numo::SFloat] x Input test data.
      # @param [Numo::SFloat] y Output test data.
      # @param [Lambda] before_batch_cbk Set the proc to be performed before batch processing.
      # @param [Lambda] after_batch_cbk Set the proc to be performed after batch processing.
      # @return [Array] Returns the test data accurate and mean loss in the form [accurate, mean_loss].
      def accurate(x, y, batch_size = 100, before_batch_cbk: nil, after_batch_cbk: nil)
        check_xy_type(x, y)
        # input_data_shape_check(x, y)
        batch_size = batch_size >= x.shape[0] ? x.shape[0] : batch_size
        dataset = Dataset.new(x, y, false)
        total_correct = 0
        sum_loss = 0
        max_iter = (x.shape[0].to_f / batch_size)
        max_iter.ceil.times do |i|
          x_batch, y_batch = dataset.next_batch(batch_size)
          correct, loss_value = test_on_batch(x_batch, y_batch, before_batch_cbk: before_batch_cbk, after_batch_cbk: after_batch_cbk)
          total_correct += correct
          sum_loss += loss_value.is_a?(Xumo::SFloat) ? loss_value.mean : loss_value
        end
        mean_loss = sum_loss / max_iter
        [total_correct.to_f / x.shape[0], mean_loss]
      end

      def test_on_batch(x, y, before_batch_cbk: nil, after_batch_cbk: nil)
        before_batch_cbk.call(false) if before_batch_cbk
        x = forward(x, false)
        correct = evaluate(x, y)
        loss_value = loss(x, y)
        after_batch_cbk.call(loss_value, false) if after_batch_cbk
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

      private def loss(y, t)
        @loss_func.forward(y, t, get_all_layers)
      end

      # Predict data.
      # @param [Numo::SFloat] x Input data.
      def predict(x)
        check_xy_type(x)
        # input_data_shape_check(x)
        forward(x, false)
      end

      # Predict one data.
      # @param [Numo::SFloat] x Input data. However, x is single data.
      def predict1(x)
        check_xy_type(x)
        predict(x.reshape(1, *x.shape))[0, false]
      end

      # Save the model in marshal format.
      # @param [String] file_name name to save model.
      def save(file_name)
        bin = Zlib::Deflate.deflate(Marshal.dump(self))
        begin
          File.binwrite(file_name, bin)
        rescue Errno::ENOENT => ex
          dir_name = file_name.match(%r`(.*)/.+$`)[1]
          Dir.mkdir(dir_name)
          File.binwrite(file_name, bin)
        end
      end

      # @return [DNN::BaseModel] Copy this model.
      def copy
        Marshal.load(Marshal.dump(self))
      end

      # Get the all layers.
      # @return [Array] all layers array.
      def get_all_layers
        layers.map { |layer|
          layer.is_a?(BaseModel) ? layer.get_all_layers : layer
        }.flatten
      end

      def update
        return unless trainable
        all_trainable_layers = layers.map { |layer|
          if layer.is_a?(BaseModel)
            layer.trainable ? layer.get_all_layers : nil
          else
            layer
          end
        }.flatten.compact.uniq
        @optimizer.update(all_trainable_layers)
      end

      # @return [DNN::Optimizers::Optimizer] optimizer Return the optimizer to use for learning.
      def optimizer
        raise DNN_Error.new("The model is not compiled.") unless compiled?
        @optimizer
      end

      # @return [DNN::Losses::Loss] loss Return the loss to use for learning.
      def loss_func
        raise DNN_Error.new("The model is not compiled.") unless compiled?
        @loss_func
      end

      # @return [Bool] Returns whether the model is learning.
      def compiled?
        @compiled
      end

      private def check_xy_type(x, y = nil)
        unless x.is_a?(Xumo::SFloat)
          raise TypeError.new("x:#{x.class.name} is not an instance of #{Xumo::SFloat.name} class.")
        end
        if y && !y.is_a?(Xumo::SFloat)
          raise TypeError.new("y:#{y.class.name} is not an instance of #{Xumo::SFloat.name} class.")
        end
      end
    end


    # This class deals with the model of the network.
    class Model < BaseModel
      def initialize
        super
        @last_link = nil
      end

      # Get the layer that the model has.
      def get_layer(*args)
        if args.length == 1
          index = args[0]
          layers[index]
        else
          layer_class, index = args
          layers.select { |layer| layer.is_a?(layer_class) }[index]
        end
      end

      def layers
        layers = []
        get_layers = -> link do
          return unless link
          layers << link.layer
          if link.is_a?(TwoInputLink)
            get_layers.(link.prev1)
            get_layers.(link.prev2)
          else
            get_layers.(link.prev)
          end
        end
        get_layers.(@last_link)
        layers.reverse
      end
    
      def forward(x, learning_phase)
        y, @last_link = call(x, learning_phase)
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

      def output_shape
        @last_link.layer.output_shape
      end

    end


    # This class deals with the model of the network.
    class Sequential < BaseModel
      # @return [Array] All layers possessed by the model.
      attr_accessor :layers

      # Load json model.
      # @param [String] json_str json string to load model.
      # @return [DNN::Sequential]
      def self.load_json(json_str)
        hash = JSON.parse(json_str, symbolize_names: true)
        model = self.from_hash(hash)
        model.compile(Utils.from_hash(hash[:optimizer]), Utils.from_hash(hash[:loss]))
        model
      end

      def self.from_hash(hash)
        model = self.new
        model.layers = hash[:layers].map { |hash_layer| Utils.from_hash(hash_layer) }
        model
      end
    
      def initialize
        super
        @layers = []
        @compiled = false
      end

      # Convert model to json string.
      # @return [String] json string.
      def to_json
        hash = self.to_hash
        hash[:version] = VERSION
        JSON.pretty_generate(hash)
      end

      # Add layer to the model.
      # @param [DNN::Layers::Layer] layer Layer to add to the model.
      # @return [DNN::BaseModel] return self.
      def <<(layer)
        if !layer.is_a?(Layers::Layer) && !layer.is_a?(BaseModel)
          raise TypeError.new("layer is not an instance of the DNN::Layers::Layer class or DNN::BaseModel class.")
        end
        @layers << layer
        self
      end

      # Set optimizer and loss_func to model and build all layers.
      # @param [DNN::Optimizers::Optimizer] optimizer Optimizer to use for learning.
      # @param [DNN::Losses::Loss] loss_func Loss function to use for learning.
      def compile(optimizer, loss_func)
        raise DNN_Error.new("The model is already compiled.") if compiled?
        setup(optimizer, loss_func)
        @compiled = true
        layers_check
        build
      end

      def build(super_model = nil)
        @super_model = super_model
        shape = if super_model
          super_model.get_prev_layer(self).output_shape
        else
          @layers.first.build
        end
        layers = super_model ? @layers : @layers[1..-1]
        layers.each do |layer|
          if layer.is_a?(Sequential)
            layer.build(self)
            layer.setup(@optimizer, @loss_func)
          else
            layer.build(shape)
          end
          shape = layer.output_shape
        end
      end

      # @return [Array] Return the input shape of the model.
      def input_shape
        @layers.first.input_shape
      end

      # @return [Array] Return the output shape of the model.
      def output_shape
        @layers.last.output_shape
      end

      private def first_layer
        @layers.first
      end

      private def last_layer
        @layers.last
      end

      # Get the layer that the model has.
      def get_layer(*args)
        if args.length == 1
          index = args[0]
          @layers[index]
        else
          layer_class, index = args
          @layers.select { |layer| layer.is_a?(layer_class) }[index]
        end
      end
    
      def forward(x, learning_phase)
        @layers.each do |layer|
          x = if layer.is_a?(BaseModel)
            layer.forward(x, learning_phase)
          else
            layer.learning_phase = learning_phase if layer.respond_to?(:learning_phase)
            layer.forward(x)
          end
        end
        x
      end
    
      def backward(dy)
        @layers.reverse.each do |layer|
          dy = layer.backward(dy)
        end
        dy
      end

      def get_prev_layer(layer)
        layer_index = @layers.index(layer)
        prev_layer = if layer_index == 0
          if @super_model
            @super_model.layers[@super_model.layers.index(self) - 1]
          else
            self
          end
        else
          @layers[layer_index - 1]
        end
        if prev_layer.is_a?(Layers::Layer)
          prev_layer
        elsif prev_layer.is_a?(BaseModel)
          prev_layer.layers.last
        end
      end

      def to_hash
        hash_layers = @layers.map { |layer| layer.to_hash }
        {class: self.class.name, layers: hash_layers, optimizer: @optimizer.to_hash, loss: @loss_func.to_hash}
      end

      private

      def layers_check
        if !@layers.first.is_a?(Layers::InputLayer) && !@layers.first.is_a?(Layers::Embedding) && !@super_model
          raise TypeError.new("The first layer is not an InputLayer or Embedding.")
        end
      end

      def input_data_shape_check(x, y = nil)
        unless @layers.first.input_shape == x.shape[1..-1]
          raise DNN_ShapeError.new("The shape of x does not match the input shape. x shape is #{x.shape[1..-1]}, but input shape is #{@layers.first.input_shape}.")
        end
        if y && @layers.last.output_shape != y.shape[1..-1]
          raise DNN_ShapeError.new("The shape of y does not match the input shape. y shape is #{y.shape[1..-1]}, but output shape is #{@layers.last.output_shape}.")
        end
      end
    end

  end
end
