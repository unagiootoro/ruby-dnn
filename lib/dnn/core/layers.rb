module DNN
  module Layers

    # Super class of all optimizer classes.
    class Layer
      def initialize
        @built = false
      end

      # Build the layer.
      def build(model)
        @model = model
        @built = true
      end
      
      # Does the layer have already been built?
      def built?
        @built
      end

      # Forward propagation.
      # Classes that inherit from this class must implement this method.
      def forward(x)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward'")
      end

      # Backward propagation.
      # Classes that inherit from this class must implement this method.
      def backward(dout)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'update'")
      end
    
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
        @model.get_prev_layer(self)
      end
    end
    
    
    # This class is a superclass of all classes with learning parameters.
    class HasParamLayer < Layer
      attr_accessor :trainable # Setting false prevents learning of parameters.
      attr_reader :params      # The parameters of the layer.
    
      def initialize
        super()
        @params = {}
        @trainable = true
      end
    
      def build(model)
        @model = model
        unless @built
          @built = true
          init_params
        end
      end
    
      # Update the parameters.
      def update
        @model.optimizer.update(@params) if @trainable
      end
    
      private
      
      # Initialize of the parameters.
      # Classes that inherit from this class must implement this method.
      def init_params
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'init_params'")
      end
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


    # It is a superclass of all connection layers.
    class Connection < HasParamLayer
      attr_reader :l1_lambda # L1 regularization
      attr_reader :l2_lambda # L2 regularization
      attr_reader :weight_initializer
      attr_reader :bias_initializer

      def initialize(weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     l1_lambda: 0,
                     l2_lambda: 0)
        super()
        @weight_initializer = weight_initializer
        @bias_initializer = bias_initializer
        @l1_lambda = l1_lambda
        @l2_lambda = l2_lambda
        @params[:weight] = @weight = Param.new
        @params[:bias] = @bias = Param.new
      end

      def lasso
        if @l1_lambda > 0
          @l1_lambda * @weight.data.abs.sum
        else
          0
        end
      end

      def ridge
        if @l2_lambda > 0
          0.5 * @l2_lambda * (@weight.data**2).sum
        else
          0
        end
      end

      def dlasso
        if @l1_lambda > 0
          dlasso = Xumo::SFloat.ones(*@weight.data.shape)
          dlasso[@weight.data < 0] = -1
          @weight.grad += @l1_lambda * dlasso
        end
      end

      def dridge
        if @l2_lambda > 0
          @weight.grad += @l2_lambda * @weight.data
        end
      end

      def to_hash(merge_hash)
        super({weight_initializer: @weight_initializer.to_hash,
               bias_initializer: @bias_initializer.to_hash,
               l1_lambda: @l1_lambda,
               l2_lambda: @l2_lambda}.merge(merge_hash))
      end

      private

      def init_params
        @weight_initializer.init_param(self, @weight)
        @bias_initializer.init_param(self, @bias)
      end
    end
    
    
    class Dense < Connection
      attr_reader :num_nodes

      def self.load_hash(hash)
        self.new(hash[:num_nodes],
                 weight_initializer: Util.load_hash(hash[:weight_initializer]),
                 bias_initializer: Util.load_hash(hash[:bias_initializer]),
                 l1_lambda: hash[:l1_lambda],
                 l2_lambda: hash[:l2_lambda])
      end
    
      def initialize(num_nodes,
                     weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     l1_lambda: 0,
                     l2_lambda: 0)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              l1_lambda: l1_lambda, l2_lambda: l2_lambda)
        @num_nodes = num_nodes
      end
    
      def forward(x)
        @x = x
        @x.dot(@weight.data) + @bias.data
      end
    
      def backward(dout)
        @weight.grad = @x.transpose.dot(dout)
        @bias.grad = dout.sum(0)
        dout.dot(@weight.data.transpose)
      end
    
      def shape
        [@num_nodes]
      end

      def to_hash
        super({num_nodes: @num_nodes})
      end
    
      private
    
      def init_params
        num_prev_nodes = prev_layer.shape[0]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes)
        @bias.data = Xumo::SFloat.new(@num_nodes)
        super()
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
      # Classes that inherit from this class must implement this method.
      def loss(x)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward'")
      end

      def dloss
        @model.get_all_layers.select { |layer| layer.is_a?(Connection) }.each do |layer|
          layer.dlasso
          layer.dridge
        end
      end
      
      private

      def lasso
        @model.get_all_layers.select { |layer| layer.is_a?(Connection) }
                             .reduce(0) { |sum, layer| sum + layer.lasso }
      end
    
      def ridge
        @model.get_all_layers.select { |layer| layer.is_a?(Connection) }
                             .reduce(0) { |sum, layer| sum + layer.ridge }
      end
    end
    
    
    class Dropout < Layer
      attr_reader :dropout_ratio

      def self.load_hash(hash)
        self.new(hash[:dropout_ratio], hash[:seed])
      end

      def initialize(dropout_ratio = 0.5, seed = rand(1 << 63))
        super()
        @dropout_ratio = dropout_ratio
        @seed = seed
        @mask = nil
      end
    
      def forward(x)
        if @model.training?
          Xumo::SFloat.srand(@seed)
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
        super({dropout_ratio: @dropout_ratio, seed: @seed})
      end
    end
    
    
    class BatchNormalization < HasParamLayer
      attr_reader :momentum

      def self.load_hash(hash)
        self.new(momentum: hash[:momentum])
      end

      def initialize(momentum: 0.9)
        super()
        @momentum = momentum
      end

      def forward(x)
        if @model.training?
          mean = x.mean(0)
          @xc = x - mean
          var = (@xc**2).mean(0)
          @std = Xumo::NMath.sqrt(var + 1e-7)
          xn = @xc / @std
          @xn = xn
          @running_mean.data = @momentum * @running_mean.data + (1 - @momentum) * mean
          @running_var.data = @momentum * @running_var.data + (1 - @momentum) * var
        else
          xc = x - @running_mean.data
          xn = xc / Xumo::NMath.sqrt(@running_var.data + 1e-7)
        end
        @gamma.data * xn + @beta.data
      end
    
      def backward(dout)
        batch_size = dout.shape[0]
        @beta.grad = dout.sum(0)
        @gamma.grad = (@xn * dout).sum(0)
        dxn = @gamma.data * dout
        dxc = dxn / @std
        dstd = -((dxn * @xc) / (@std**2)).sum(0)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / batch_size) * @xc * dvar
        dmean = dxc.sum(0)
        dxc - dmean / batch_size
      end

      def to_hash
        super({momentum: @momentum})
      end
    
      private
    
      def init_params
        @params[:gamma] = @gamma = Param.new(Xumo::SFloat.ones(*shape))
        @params[:beta] = @beta = Param.new(Xumo::SFloat.zeros(*shape))
        @params[:running_mean] = @running_mean = Param.new(Xumo::SFloat.zeros(*shape))
        @params[:running_var] = @running_var = Param.new(Xumo::SFloat.zeros(*shape))
      end
    end
  end
  
end
