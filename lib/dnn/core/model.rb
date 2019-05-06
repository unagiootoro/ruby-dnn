require "zlib"
require "json"
require "base64"

module DNN

  # This class deals with the model of the network.
  class Model
    # @return [Array] All layers possessed by the model.
    attr_accessor :layers
    # @return [Bool] Setting false prevents learning of parameters.
    attr_accessor :trainable

    # Load marshal model.
    # @param [String] file_name File name of marshal model to load.
    def self.load(file_name)
      Marshal.load(Zlib::Inflate.inflate(File.binread(file_name)))
    end

    # Load json model.
    # @param [String] json_str json string to load model.
    # @return [DNN::Model]
    def self.load_json(json_str)
      hash = JSON.parse(json_str, symbolize_names: true)
      model = self.load_hash(hash)
      model.compile(Utils.load_hash(hash[:optimizer]), Utils.load_hash(hash[:loss]))
      model
    end

    def self.load_hash(hash)
      model = self.new
      model.layers = hash[:layers].map { |hash_layer| Utils.load_hash(hash_layer) }
      model
    end
  
    def initialize
      @layers = []
      @trainable = true
      @optimizer = nil
      @compiled = false
    end

    # Load json model parameters.
    # @param [String] json_str json string to load model parameters.
    def load_json_params(json_str)
      hash = JSON.parse(json_str, symbolize_names: true)
      has_param_layers_params = hash[:params]
      has_param_layers_index = 0
      has_param_layers = get_all_layers.select { |layer| layer.is_a?(Layers::HasParamLayer) }
      has_param_layers.each do |layer|
        hash_params = has_param_layers_params[has_param_layers_index]
        hash_params.each do |key, (shape, base64_param)|
          bin = Base64.decode64(base64_param)
          data = Xumo::SFloat.from_binary(bin).reshape(*shape)
          layer.params[key].data = data
        end
        has_param_layers_index += 1
      end
    end

    # Save the model in marshal format.
    # @param [String] File file_name name to save model.
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

    # Convert model to json string.
    # @return [String] json string.
    def to_json
      hash = self.to_hash
      hash[:version] = VERSION
      JSON.pretty_generate(hash)
    end
    
    def params_to_json
      has_param_layers = get_all_layers.select { |layer| layer.is_a?(Layers::HasParamLayer) }
      has_param_layers_params = has_param_layers.map do |layer|
        layer.params.map { |key, param|
          base64_data = Base64.encode64(param.data.to_binary)
          [key, [param.data.shape, base64_data]]
        }.to_h
      end
      hash = {version: VERSION, params: has_param_layers_params}
      JSON.dump(hash)
    end

    # Add layer to the model.
    # @param [DNN::Layers::Layer] layer Layer to add to the model.
    # @return [DNN::Model] return self.
    def <<(layer)
      # Due to a bug in saving nested models, temporarily prohibit model nesting.
      # if !layer.is_a?(Layers::Layer) && !layer.is_a?(Model)
      #   raise TypeError.new("layer is not an instance of the DNN::Layers::Layer class or DNN::Model class.")
      # end
      unless layer.is_a?(Layers::Layer)
        raise TypeError.new("layer:#{layer.class.name} is not an instance of the DNN::Layers::Layer class.")
      end
      @layers << layer
      self
    end

    # Set optimizer and loss to model and build all layers.
    # @param [DNN::Optimizers::Optimizer] optimizer Optimizer to use for learning.
    # @param [DNN::Losses::Loss] loss Lptimizer to use for learning.
    def compile(optimizer, loss)
      raise DNN_Error.new("The model is already compiled.") if compiled?
      unless optimizer.is_a?(Optimizers::Optimizer)
        raise TypeError.new("optimizer:#{optimizer.class} is not an instance of DNN::Optimizers::Optimizer class.")
      end
      unless loss.is_a?(Losses::Loss)
        raise TypeError.new("loss:#{loss.class} is not an instance of DNN::Losses::Loss class.")
      end
      @compiled = true
      layers_check
      @optimizer = optimizer
      @loss = loss
      build
      layers_shape_check
    end

    # Set optimizer and loss to model and recompile. But does not build layers.
    # @param [DNN::Optimizers::Optimizer] optimizer Optimizer to use for learning.
    # @param [DNN::Losses::Loss] loss Lptimizer to use for learning.
    def recompile(optimizer, loss)
      unless optimizer.is_a?(Optimizers::Optimizer)
        raise TypeError.new("optimizer:#{optimizer.class} is not an instance of DNN::Optimizers::Optimizer class.")
      end
      unless loss.is_a?(Losses::Loss)
        raise TypeError.new("loss:#{loss.class} is not an instance of DNN::Losses::Loss class.")
      end
      @compiled = true
      layers_check
      @optimizer = optimizer
      @loss = loss
      layers_shape_check
    end

    def build(super_model = nil)
      @super_model = super_model
      shape = if super_model
        super_model.output_shape
      else
        @layers.first.build
      end
      @layers[1..-1].each do |layer|
        if layer.is_a?(Model)
          layer.build(self)
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

    # @return [DNN::Optimizers::Optimizer] optimizer Return the optimizer to use for learning.
    def optimizer
      raise DNN_Error.new("The model is not compiled.") unless compiled?
      @optimizer ? @optimizer : @super_model.optimizer
    end

    # @return [DNN::Losses::Loss] loss Return the loss to use for learning.
    def loss
      raise DNN_Error.new("The model is not compiled.") unless compiled?
      @loss ? @loss : @super_model.loss
    end

    # @return [Bool] Returns whether the model is learning.
    def compiled?
      @compiled
    end

    # Start training.
    # Compile the model before use this method.
    # @param [Numo::SFloat] x Input training data.
    # @param [Numo::SFloat] y Output training data.
    # @param [Integer] epochs Number of training.
    # @option options [Array or NilClass] :test (nil) If you to test the model for every 1 epoch,
    #     specify [x_test, y_test]. Don't test to the model, specify nil.                     
    # @option options [Bool] :verbose (true) Set true to display the log. If false is set, the log is not displayed.
    # @option options [nil] :batch_proc Set proc to process per batch.
    def train(x, y, epochs,
              batch_size: 1,
              test: nil,
              verbose: true,
              batch_proc: nil,
              &epoch_proc)
      raise DNN_Error.new("The model is not compiled.") unless compiled?
      check_xy_type(x, y)
      dataset = Dataset.new(x, y)
      num_train_datas = x.shape[0]
      (1..epochs).each do |epoch|
        puts "【 epoch #{epoch}/#{epochs} 】" if verbose
        (num_train_datas.to_f / batch_size).ceil.times do |index|
          x_batch, y_batch = dataset.get_batch(batch_size)
          loss = train_on_batch(x_batch, y_batch, &batch_proc)
          if loss.nan?
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
          log << "  #{num_trained_datas}/#{num_train_datas} loss: #{sprintf('%.8f', loss)}"
          print log if verbose
        end
        if verbose && test
          acc = accurate(test[0], test[1], batch_size, &batch_proc)
          print "  accurate: #{acc}"
        end
        puts "" if verbose
        epoch_proc.call(epoch) if epoch_proc
      end
    end
  
    # Training once.
    # Compile the model before use this method.
    # @param [Numo::SFloat] x Input training data.
    # @param [Numo::SFloat] y Output training data.
    def train_on_batch(x, y, &batch_proc)
      raise DNN_Error.new("The model is not compiled.") unless compiled?
      check_xy_type(x, y)
      input_data_shape_check(x, y)
      x, y = batch_proc.call(x, y) if batch_proc
      out = forward(x, true)
      loss_value = @loss.forward(out, y, get_all_layers)
      dout = @loss.backward(y)
      backward(dout)
      @loss.d_regularize(get_all_layers)
      update
      loss_value
    end
  
    # Evaluate model and get accurate of test data.
    # @param [Numo::SFloat] x Input test data.
    # @param [Numo::SFloat] y Output test data.
    def accurate(x, y, batch_size = 100, &batch_proc)
      check_xy_type(x, y)
      input_data_shape_check(x, y)
      batch_size = batch_size >= x.shape[0] ? x.shape[0] : batch_size
      correct = 0
      (x.shape[0].to_f / batch_size).ceil.times do |i|
        x_batch = Xumo::SFloat.zeros(batch_size, *x.shape[1..-1])
        y_batch = Xumo::SFloat.zeros(batch_size, *y.shape[1..-1])
        batch_size.times do |j|
          k = i * batch_size + j
          break if k >= x.shape[0]
          x_batch[j, false] = x[k, false]
          y_batch[j, false] = y[k, false]
        end
        x_batch, y_batch = batch_proc.call(x_batch, y_batch) if batch_proc
        out = forward(x_batch, false)
        batch_size.times do |j|
          if @layers.last.output_shape == [1]
            correct += 1 if out[j, 0].round == y_batch[j, 0].round
          else
            correct += 1 if out[j, true].max_index == y_batch[j, true].max_index
          end
        end
      end
      correct.to_f / x.shape[0]
    end

    # Predict data.
    # @param [Numo::SFloat] x Input data.
    def predict(x)
      check_xy_type(x)
      input_data_shape_check(x)
      forward(x, false)
    end

    # Predict one data.
    # @param [Numo::SFloat] x Input data.
    def predict1(x)
      check_xy_type(x)
      predict(Xumo::SFloat.cast([x]))[0, false]
    end

    # @return [DNN::Model] Copy this model.
    def copy
      Marshal.load(Marshal.dump(self))
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

    # Get the all layers.
    # @return [Array] all layers array.
    def get_all_layers
      @layers.map { |layer|
        layer.is_a?(Model) ? layer.get_all_layers : layer
      }.flatten
    end
  
    # TODO
    # It is not good to write the Layer class name directly in the Model class. I will fix it later.
    def forward(x, learning_phase)01
      @layers.each do |layer|
        x = if layer.is_a?(Layers::Dropout) || layer.is_a?(Layers::BatchNormalization) || layer.is_a?(Model)
          layer.forward(x, learning_phase)
        else
          layer.forward(x)
        end
      end
      x
    end
  
    def backward(dout)
      @layers.reverse.each do |layer|
        dout = layer.backward(dout)
      end
      dout
    end

    def update
      return unless @trainable
      @layers.each do |layer|
        if layer.is_a?(Layers::HasParamLayer)
          layer.update(@optimizer)
        elsif layer.is_a?(Model)
          layer.update
        end
      end
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
      elsif prev_layer.is_a?(Model)
        prev_layer.layers.last
      end
    end

    def to_hash
      hash_layers = @layers.map { |layer| layer.to_hash }
      {class: Model.name, layers: hash_layers, optimizer: @optimizer.to_hash, loss: @loss.to_hash}
    end

    private

    def layers_check
      unless @layers.first.is_a?(Layers::InputLayer)
        raise TypeError.new("The first layer is not an InputLayer.")
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

    def layers_shape_check
      @layers.each.with_index do |layer, i|
        prev_shape = layer.input_shape
        if layer.is_a?(Layers::Dense)
          if prev_shape.length != 1
            raise DNN_ShapeError.new("layer index(#{i}) Dense:  The shape of the previous layer is #{prev_shape}. The shape of the previous layer must be 1 dimensional.")
          end
        elsif layer.is_a?(Layers::Conv2D) || layer.is_a?(Layers::MaxPool2D)
          if prev_shape.length != 3
            raise DNN_ShapeError.new("layer index(#{i}) Conv2D:  The shape of the previous layer is #{prev_shape}. The shape of the previous layer must be 3 dimensional.")
          end
        elsif layer.is_a?(Layers::RNN)
          if prev_shape.length != 2
            layer_name = layer.class.name.match("\:\:(.+)$")[1]
            raise DNN_ShapeError.new("layer index(#{i}) #{layer_name}:  The shape of the previous layer is #{prev_shape}. The shape of the previous layer must be 3 dimensional.")
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

end
