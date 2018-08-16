module DNN
  module Layers

    # Super class of all optimizer classes.
    class Layer
      def initialize
        @built = false
      end

      # Build the layer.
      def build(model)
        @built = true
        @model = model
      end
      
      # Does the layer have already been built?
      def built?
        @built
      end

      # Forward propagation.
      def forward() end

      # Backward propagation.
      def backward() end
    
      # Get the shape of the layer.
      def shape
        prev_layer.shape
      end

      # Layer to a hash.
      def to_hash(merge_hash = nil)
        hash = {class: self.class.name}
        hash.merge!(merge_hash) if merge_hash
        hash
      end
    
      # Get the previous layer.
      def prev_layer
        @model.layers[@model.layers.index(self) - 1]
      end
    end
    
    
    # This class is a superclass of all classes with learning parameters.
    class HasParamLayer < Layer
      attr_reader :params # The parameters of the layer.
      attr_reader :grads  # Differential value of parameter of layer.
    
      def initialize
        super
        @params = {}
        @grads = {}
      end
    
      def build(model)
        super
        init_params
      end
    
      # Update the parameters.
      def update
        @model.optimizer.update(self)
      end
    
      private
      
      # Initialize of the parameters.
      def init_params() end
    end
    
    
    class InputLayer < Layer
      attr_reader :shape

      def self.load_hash(hash)
        self.new(hash[:shape])
      end

      def initialize(dim_or_shape)
        super()
        @shape = dim_or_shape.is_a?(Array) ? dim_or_shape : [dim_or_shape]
      end

      def forward(x)
        x
      end
    
      def backward(dout)
        dout
      end

      def to_hash
        super({shape: @shape})
      end
    end
    
    
    class Dense < HasParamLayer
      include Initializers

      attr_reader :num_nodes
      attr_reader :weight_decay

      def self.load_hash(hash)
        self.new(hash[:num_nodes],
                 weight_initializer: Util.load_hash(hash[:weight_initializer]),
                 bias_initializer: Util.load_hash(hash[:bias_initializer]),
                 weight_decay: hash[:weight_decay])
      end
    
      def initialize(num_nodes,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     weight_decay: 0)
        super()
        @num_nodes = num_nodes
        @weight_initializer = (weight_initializer || RandomNormal.new)
        @bias_initializer = (bias_initializer || Zeros.new)
        @weight_decay = weight_decay
      end
    
      def forward(x)
        @x = x
        @x.dot(@params[:weight]) + @params[:bias]
      end
    
      def backward(dout)
        @grads[:weight] = @x.transpose.dot(dout)
        if @weight_decay > 0
          dridge = @weight_decay * @params[:weight]
          @grads[:weight] += dridge
        end
        @grads[:bias] = dout.sum(0)
        dout.dot(@params[:weight].transpose)
      end
    
      def shape
        [@num_nodes]
      end

      def ridge
        if @weight_decay > 0
          0.5 * @weight_decay * (@params[:weight]**2).sum
        else
          0
        end
      end

      def to_hash
        super({num_nodes: @num_nodes,
               weight_initializer: @weight_initializer.to_hash,
               bias_initializer: @bias_initializer.to_hash,
               weight_decay: @weight_decay})
      end
    
      private
    
      def init_params
        num_prev_nodes = prev_layer.shape[0]
        @params[:weight] = Xumo::SFloat.new(num_prev_nodes, @num_nodes)
        @params[:bias] = Xumo::SFloat.new(@num_nodes)
        @weight_initializer.init_param(self, :weight)
        @bias_initializer.init_param(self, :bias)
      end
    end
    

    class Flatten < Layer
      def forward(x)
        @shape = x.shape
        x.reshape(x.shape[0], x.shape[1..-1].reduce(:*))
      end
    
      def backward(dout)
        dout.reshape(*@shape)
      end
    
      def shape
        [prev_layer.shape.reduce(:*)]
      end
    end


    class Reshape < Layer
      attr_reader :shape
      
      def initialize(shape)
        super()
        @shape = shape
        @x_shape = nil
      end

      def self.load_hash(hash)
        self.new(hash[:shape])
      end

      def forward(x)
        @x_shape = x.shape
        x.reshape(x.shape[0], *@shape)
      end

      def backward(dout)
        dout.reshape(*@x_shape)
      end

      def to_hash
        super({shape: @shape})
      end
    end


    class OutputLayer < Layer
      private
    
      def ridge
        @model.layers.select { |layer| layer.respond_to?(:ridge) }
                     .reduce(0) { |sum, layer| sum + layer.ridge }
      end
    end
    
    
    class Dropout < Layer
      attr_reader :dropout_ratio

      def self.load_hash(hash)
        self.new(hash[:dropout_ratio])
      end

      def initialize(dropout_ratio = 0.5)
        super()
        @dropout_ratio = dropout_ratio
        @mask = nil
      end
    
      def forward(x)
        if @model.training?
          @mask = Xumo::SFloat.ones(*x.shape).rand < @dropout_ratio
          x[@mask] = 0
        else
          x *= (1 - @dropout_ratio)
        end
        x
      end
    
      def backward(dout)
        dout[@mask] = 0 if @model.training?
        dout
      end

      def to_hash
        super({dropout_ratio: @dropout_ratio})
      end
    end
    
    
    class BatchNormalization < HasParamLayer
      attr_reader :momentum

      def self.load_hash(hash)
        running_mean = Xumo::SFloat.cast(hash[:running_mean])
        running_var = Xumo::SFloat.cast(hash[:running_var])
        self.new(momentum: hash[:momentum], running_mean: running_mean, running_var: running_var)
      end

      def initialize(momentum: 0.9, running_mean: nil, running_var: nil)
        super()
        @momentum = momentum
        @running_mean = running_mean
        @running_var = running_var
      end

      def build(model)
        super
        @running_mean ||= Xumo::SFloat.zeros(*shape)
        @running_var ||= Xumo::SFloat.zeros(*shape)
      end

      def forward(x)
        if @model.training?
          mean = x.mean(0)
          @xc = x - mean
          var = (@xc**2).mean(0)
          @std = Xumo::NMath.sqrt(var + 1e-7)
          xn = @xc / @std
          @xn = xn
          @running_mean = @momentum * @running_mean + (1 - @momentum) * mean
          @running_var = @momentum * @running_var + (1 - @momentum) * var
        else
          xc = x - @running_mean
          xn = xc / Xumo::NMath.sqrt(@running_var + 1e-7)
        end
        @params[:gamma] * xn + @params[:beta]
      end
    
      def backward(dout)
        batch_size = dout.shape[0]
        @grads[:beta] = dout.sum(0)
        @grads[:gamma] = (@xn * dout).sum(0)
        dxn = @params[:gamma] * dout
        dxc = dxn / @std
        dstd = -((dxn * @xc) / (@std**2)).sum(0)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / batch_size) * @xc * dvar
        dmean = dxc.sum(0)
        dxc - dmean / batch_size
      end

      def to_hash
        super({momentum: @momentum,
               running_mean: @running_mean.to_a,
               running_var: @running_var.to_a})
      end
    
      private
    
      def init_params
        @params[:gamma] = Xumo::SFloat.ones(*shape)
        @params[:beta] = Xumo::SFloat.zeros(*shape)
      end
    end
  end
  
end
