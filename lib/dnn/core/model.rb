require "json"

module DNN
  # This class deals with the model of the network.
  class Model
    attr_accessor :layers
    attr_reader :optimizer

    def self.load(file_name)
      Marshal.load(File.binread(file_name))
    end

    def self.load_json(json_str)
      hash = JSON.parse(json_str, symbolize_names: true)
      model = self.new
      model.layers = hash[:layers].map { |hash_layer| Util.load_hash(hash_layer) }
      model.compile(Util.load_hash(hash[:optimizer]))
      model
    end
  
    def initialize
      @layers = []
      @optimizer = nil
      @training = false
      @compiled = false
    end

    def load_json_params(json_str)
      has_param_layers_params = JSON.parse(json_str, symbolize_names: true)
      has_param_layers_index = 0
      @layers.each do |layer|
        next unless layer.is_a?(HasParamLayer)
        hash_params = has_param_layers_params[has_param_layers_index]
        hash_params.each do |key, param|
          layer.params[key] = Xumo::SFloat.cast(param)
        end
        has_param_layers_index += 1
      end
    end
  
    def save(file_name)
      marshal = Marshal.dump(self)
      begin
        File.binwrite(file_name, marshal)
      rescue Errno::ENOENT => ex
        dir_name = file_name.match(%r`(.*)/.+$`)[1]
        Dir.mkdir(dir_name)
        File.binwrite(file_name, marshal)
      end
    end

    def to_json
      hash_layers = @layers.map { |layer| layer.to_hash }
      hash = {version: VERSION, layers: hash_layers, optimizer: @optimizer.to_hash}
      JSON.dump(hash)
    end
    
    def params_to_json
      has_param_layers = @layers.select { |layer| layer.is_a?(HasParamLayer) }
      has_param_layers_params = has_param_layers.map do |layer|
        layer.params.map { |key, param| [key, param.to_a] }.to_h
      end
      JSON.dump(has_param_layers_params)
    end
  
    def <<(layer)
      unless layer.is_a?(Layers::Layer)
        raise TypeError.new("layer is not an instance of the DNN::Layers::Layer class.")
      end
      @layers << layer
      self
    end
  
    def compile(optimizer)
      unless optimizer.is_a?(Optimizers::Optimizer)
        raise TypeError.new("optimizer is not an instance of the DNN::Optimizers::Optimizer class.")
      end
      @compiled = true
      layers_check
      @optimizer = optimizer
      @layers.each do |layer|
        layer.build(self)
      end
      layers_shape_check
    end

    def compiled?
      @compiled
    end

    def training?
      @training
    end

    def train(x, y, epochs,
              batch_size: 1,
              test: nil,
              verbose: true,
              batch_proc: nil,
              &epoch_proc)
      unless compiled?
        raise DNN_Error.new("The model is not compiled.")
      end
      num_train_data = x.shape[0]
      (1..epochs).each do |epoch|
        puts "【 epoch #{epoch}/#{epochs} 】" if verbose
        (num_train_data.to_f / batch_size).ceil.times do |index|
          x_batch, y_batch = Util.get_minibatch(x, y, batch_size)
          loss = train_on_batch(x_batch, y_batch, &batch_proc)
          if loss.nan?
            puts "\nloss is nan" if verbose
            return
          end
          num_trained_data = (index + 1) * batch_size
          num_trained_data = num_trained_data > num_train_data ? num_train_data : num_trained_data
          log = "\r"
          40.times do |i|
            if i < num_trained_data * 40 / num_train_data
              log << "="
            elsif i == num_trained_data * 40 / num_train_data
              log << ">"
            else
              log << "_"
            end
          end
          log << "  #{num_trained_data}/#{num_train_data} loss: #{sprintf('%.8f', loss)}"
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
  
    def train_on_batch(x, y, &batch_proc)
      x, y = batch_proc.call(x, y) if batch_proc
      forward(x, true)
      loss = @layers[-1].loss(y)
      backward(y)
      @layers.each { |layer| layer.update if layer.respond_to?(:update) }
      loss
    end
  
    def accurate(x, y, batch_size = 1, &batch_proc)
      batch_size = batch_size >= x.shape[0] ? batch_size : x.shape[0]
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
          if @layers[-1].shape == [1]
            correct += 1 if out[j, 0].round == y_batch[j, 0].round
          else
            correct += 1 if out[j, true].max_index == y_batch[j, true].max_index
          end
        end
      end
      correct.to_f / x.shape[0]
    end
  
    def predict(x)
      forward(x, false)
    end

    def predict1(x)
      predict(Xumo::SFloat.cast([x]))[0, false]
    end
  
    def forward(x, training)
      unless compiled?
        raise DNN_Error.new("The model is not compiled.")
      end
      @training = training
      @layers.each do |layer|
        x = layer.forward(x)
      end
      x
    end
  
    def backward(y)
      dout = y
      @layers[0..-1].reverse.each do |layer|
        dout = layer.backward(dout)
      end
      dout
    end

    private

    def layers_check
      unless @layers.first.is_a?(Layers::InputLayer)
        raise TypeError.new("The first layer is not an InputLayer.")
      end
      unless @layers.last.is_a?(Layers::OutputLayer)
        raise TypeError.new("The last layer is not an OutputLayer.")
      end
    end

    def layers_shape_check
      @layers.each.with_index do |layer, i|
        if layer.is_a?(Layers::Dense)
          prev_shape = layer.prev_layer.shape
          if prev_shape.length != 1
            raise DNN_ShapeError.new("layer index(#{i}) Dense:  The shape of the previous layer is #{prev_shape}. The shape of the previous layer must be 1 dimensional.")
          end
        elsif layer.is_a?(Layers::Conv2D) || layer.is_a?(Layers::MaxPool2D)
          prev_shape = layer.prev_layer.shape
          if prev_shape.length != 3
            raise DNN_ShapeError.new("layer index(#{i}) Conv2D:  The shape of the previous layer is #{prev_shape}. The shape of the previous layer must be 3 dimensional.")
          end
        end
      end
    end
  end
end
