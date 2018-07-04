module DNN
  class Model
    attr_reader :layers
    attr_reader :optimizer
    attr_reader :batch_size
    attr_reader :training
  
    def self.load(file_name)
      Marshal.load(File.binread(file_name))
    end
  
    def initialize
      @layers = []
      @optimizer = nil
      @batch_size = nil
    end
  
    def save(file_name)
      dir_name = file_name.match(%r`(.*)/.+$`)[1]
      Dir.mkdir(dir_name) unless Dir.exist?(dir_name)
      File.binwrite(file_name, Marshal.dump(self))
    end
  
    def <<(layer)
      unless layer.is_a?(Layers::Layer)
        raise DNN_TypeError.new("layer is not an instance of the DNN::Layers::Layer class.")
      end
      @layers << layer
      self
    end
  
    def compile(optimizer)
      unless optimizer.is_a?(Optimizers::Optimizer)
        raise DNN_TypeError.new("optimizer is not an instance of the DNN::Optimizers::Optimizer class.")
      end
      layers_check
      @optimizer = optimizer
      @layers.each do |layer|
        layer.init(self)
      end
      layers_shape_check
    end
  
    def train(x, y, epochs,
              batch_size: 1,
              batch_proc: nil,
              verbose: true,
              &epoch_proc)
      @batch_size = batch_size
      num_train_data = x.shape[0]
      (1..epochs).each do |epoch|
        puts "【 epoch #{epoch}/#{epochs} 】" if verbose
        (num_train_data.to_f / @batch_size).ceil.times do |index|
          x_batch, y_batch = Util.get_minibatch(x, y, @batch_size)
          loss = train_on_batch(x_batch, y_batch, @batch_size, &batch_proc)
          if loss.nan?
            puts "\nloss is nan" if verbose
            return
          end
          num_trained_data = (index + 1) * batch_size
          num_trained_data = num_trained_data > num_train_data ? num_train_data : num_trained_data
          log = "\r"
          20.times do |i|
            if i < num_trained_data * 20 / num_train_data
              log << "■"
            else
              log << "・"
            end
          end
          log << "  #{num_trained_data}/#{num_train_data} loss: #{loss}"
          print log if verbose
        end
        puts "" if verbose
        epoch_proc.call(epoch) if epoch_proc
      end
    end
  
    def train_on_batch(x, y, batch_size, &batch_proc)
      @batch_size = batch_size
      x, y = batch_proc.call(x, y) if batch_proc
      forward(x, true)
      backward(y)
      @layers.each { |layer| layer.update if layer.respond_to?(:update) }
      @layers[-1].loss(y)
    end
  
    def test(x, y, batch_size = nil, &batch_proc)
      @batch_size = batch_size if batch_size
      acc = accurate(x, y, @batch_size, &batch_proc)
      puts "accurate: #{acc}"
      acc
    end
  
    def accurate(x, y, batch_size = nil, &batch_proc)
      @batch_size = batch_size if batch_size
      correct = 0
      (x.shape[0].to_f / @batch_size).ceil.times do |i|
        x_batch = SFloat.zeros(@batch_size, *x.shape[1..-1])
        y_batch = SFloat.zeros(@batch_size, *y.shape[1..-1])
        @batch_size.times do |j|
          k = i * @batch_size + j
          break if k >= x.shape[0]
          x_batch[j, false] = x[k, false]
          y_batch[j, false] = y[k, false]
        end
        x_batch, y_batch = batch_proc.call(x_batch, y_batch) if batch_proc
        out = forward(x_batch, false)
        @batch_size.times do |j|
         correct += 1 if out[j, true].max_index == y_batch[j, true].max_index
        end
      end
      correct.to_f / x.shape[0]
    end
  
    def predict(x)
      forward(x, false)
    end
  
    private
  
    def forward(x, training)
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
    end

    def layers_check
      unless @layers.first.is_a?(Layers::InputLayer)
        raise DNN_Error.new("The first layer is not an InputLayer.")
      end
      unless @layers.last.is_a?(Layers::OutputLayer)
        raise DNN_Error.new("The last layer is not an OutputLayer.")
      end
    end

    def layers_shape_check
      @layers.each.with_index do |layer, i|
        if layer.is_a?(Layers::Dense)
          prev_shape = layer.prev_layer.shape
          if prev_shape.length != 1
            raise DNN_SharpError.new("layer index(#{i}) Dense:  The shape of the previous layer is #{prev_shape}. The shape of the previous layer must be 1 dimensional.")
          end
        elsif layer.is_a?(Layers::Conv2D) || layer.is_a?(Layers::MaxPool2D)
          prev_shape = layer.prev_layer.shape
          if prev_shape.length != 3
            raise DNN_SharpError.new("layer index(#{i}) Conv2D:  The shape of the previous layer is #{prev_shape}. The shape of the previous layer must be 3 dimensional.")
          end
        end
      end
    end
  end
end
