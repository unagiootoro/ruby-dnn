module DNN
  module Layers

    # Super class of all optimizer classes.
    class Layer
      attr_reader :input_shape

      def initialize
        @built = false
      end

      # Build the layer.
      def build(input_shape)
        @input_shape = input_shape
        @built = true
      end
      
      # Does the layer have already been built?
      def built?
        @built
      end

      # Forward propagation.
      def forward(x)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward'")
      end

      # Backward propagation.
      def backward(dy)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'update'")
      end

      def output_shape
        @input_shape
      end

      # Layer to a hash.
      def to_hash(merge_hash = nil)
        hash = {class: self.class.name}
        hash.merge!(merge_hash) if merge_hash
        hash
      end
    end
    
    
    # This class is a superclass of all classes with learning parameters.
    class HasParamLayer < Layer
      # @return [Bool] trainable Setting false prevents learning of parameters.
      attr_accessor :trainable
      # @return [Array] The parameters of the layer.
      attr_reader :params
    
      def initialize
        super()
        @params = {}
        @trainable = true
      end
    
      # Update the parameters.
      def update(optimizer)
        optimizer.update(@params) if @trainable
      end
    end
    
    
    class InputLayer < Layer
      def self.from_hash(hash)
        self.new(hash[:input_shape])
      end

      def initialize(input_dim_or_shape)
        super()
        @input_shape = input_dim_or_shape.is_a?(Array) ? input_dim_or_shape : [input_dim_or_shape]
      end

      def build
        @built = true
        @input_shape
      end

      def forward(x)
        x
      end
    
      def backward(dy)
        dy
      end

      def to_hash
        super({input_shape: @input_shape})
      end
    end


    # It is a superclass of all connection layers.
    class Connection < HasParamLayer
      # @return [DNN::Initializers] weight initializer.
      attr_reader :weight_initializer
      # @return [DNN::Initializers] bias initializer.
      attr_reader :bias_initializer
      # @return [Float] L1 regularization.
      attr_reader :l1_lambda
      # @return [Float] L2 regularization.
      attr_reader :l2_lambda

      # @param [DNN::Initializers] weight_initializer weight initializer.
      # @param [DNN::Initializers] bias_initializer bias initializer.
      # @param [Float] l1_lambda L1 regularization
      # @param [Float] l2_lambda L2 regularization
      # @param [Bool] use_bias whether to use bias.
      def initialize(weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     l1_lambda: 0,
                     l2_lambda: 0,
                     use_bias: true)
        super()
        @weight_initializer = weight_initializer
        @bias_initializer = bias_initializer
        @l1_lambda = l1_lambda
        @l2_lambda = l2_lambda
        @params[:weight] = @weight = Param.new(nil, 0)
        # For compatibility on or before with v0.9.3, setting use_bias to nil use bias.
        # Therefore, setting use_bias to nil is deprecated.
        if use_bias || use_bias == nil
          @params[:bias] = @bias = Param.new(nil, 0)
        else
          @bias = nil
        end
      end

      def regularizers
        regularizers = []
        regularizers << Lasso.new(@l1_lambda, @weight) if @l1_lambda > 0
        regularizers << Ridge.new(@l2_lambda, @weight) if @l2_lambda > 0
        regularizers
      end

      # @return [Bool] Return whether to use bias.
      def use_bias
        @bias ? true : false
      end

      def to_hash(merge_hash)
        super({weight_initializer: @weight_initializer.to_hash,
               bias_initializer: @bias_initializer.to_hash,
               l1_lambda: @l1_lambda,
               l2_lambda: @l2_lambda}.merge(merge_hash))
      end
    end
    
    
    # Full connnection layer.
    class Dense < Connection
      # @return [Integer] number of nodes.
      attr_reader :num_nodes

      def self.from_hash(hash)
        self.new(hash[:num_nodes],
                 weight_initializer: Utils.from_hash(hash[:weight_initializer]),
                 bias_initializer: Utils.from_hash(hash[:bias_initializer]),
                 l1_lambda: hash[:l1_lambda],
                 l2_lambda: hash[:l2_lambda],
                 use_bias: hash[:use_bias])
      end

      # @param [Integer] num_nodes number of nodes.
      def initialize(num_nodes,
                     weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     l1_lambda: 0,
                     l2_lambda: 0,
                     use_bias: true)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              l1_lambda: l1_lambda, l2_lambda: l2_lambda, use_bias: use_bias)
        @num_nodes = num_nodes
      end

      def build(input_shape)
        super
        num_prev_nodes = input_shape[0]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes)
        @weight_initializer.init_param(self, @weight)
        if @bias
          @bias.data = Xumo::SFloat.new(@num_nodes)
          @bias_initializer.init_param(self, @bias) 
        end
      end
    
      def forward(x)
        @x = x
        y = x.dot(@weight.data)
        y += @bias.data if @bias
        y
      end
    
      def backward(dy)
        @weight.grad += @x.transpose.dot(dy)
        @bias.grad += dy.sum(0) if @bias
        dy.dot(@weight.data.transpose)
      end
    
      def output_shape
        [@num_nodes]
      end

      def to_hash
        super({num_nodes: @num_nodes})
      end
    end
    

    class Flatten < Layer
      def forward(x)
        x.reshape(x.shape[0], *output_shape)
      end
    
      def backward(dy)
        dy.reshape(dy.shape[0], *@input_shape)
      end

      def output_shape
        [@input_shape.reduce(:*)]
      end
    end


    class Reshape < Layer
      def self.from_hash(hash)
        self.new(hash[:output_shape])
      end

      def initialize(output_shape)
        super()
        @output_shape = output_shape
      end

      def forward(x)
        x.reshape(x.shape[0], *@output_shape)
      end

      def backward(dy)
        dy.reshape(dy.shape[0], *@input_shape)
      end

      def output_shape
        @output_shape
      end

      def to_hash
        super({output_shape: @output_shape})
      end
    end

    
    class Dropout < Layer
      # @return [Float] dropout ratio.
      attr_reader :dropout_ratio
      # @return [Float] Use 'weight scaling inference rule'.
      attr_reader :use_scale

      def self.from_hash(hash)
        self.new(hash[:dropout_ratio], seed: hash[:seed], use_scale: hash[:use_scale])
      end

      def initialize(dropout_ratio = 0.5, seed: rand(1 << 31), use_scale: true)
        super()
        @dropout_ratio = dropout_ratio
        @seed = seed
        @use_scale = use_scale
        @mask = nil
      end

      def forward(x, learning_phase)
        if learning_phase
          Xumo::SFloat.srand(@seed)
          @mask = Xumo::SFloat.ones(*x.shape).rand < @dropout_ratio
          x[@mask] = 0
        else
          x *= (1 - @dropout_ratio) if @use_scale
        end
        x
      end
    
      def backward(dy)
        dy[@mask] = 0
        dy
      end

      def to_hash
        super({dropout_ratio: @dropout_ratio, seed: @seed, use_scale: @use_scale})
      end
    end
    
    
    class BatchNormalization < HasParamLayer
      # @return [Integer] The axis to normalization.
      attr_reader :axis
      # @return [Float] Exponential moving average of mean and variance.
      attr_reader :momentum

      def self.from_hash(hash)
        self.new(axis: hash[:axis], momentum: hash[:momentum])
      end

      # @param [integer] axis The axis to normalization.
      # @param [Float] momentum Exponential moving average of mean and variance.
      def initialize(axis: 0, momentum: 0.9)
        super()
        # For compatibility on or before with v0.9.3, setting axis to nil will use 0.
        # Therefore, setting axis to nil is deprecated.
        axis ||= 0
        @axis = axis
        @momentum = momentum
      end

      def build(input_shape)
        super
        @params[:gamma] = @gamma = Param.new(Xumo::SFloat.ones(*output_shape))
        @params[:beta] = @beta = Param.new(Xumo::SFloat.zeros(*output_shape))
        @params[:running_mean] = @running_mean = Param.new(Xumo::SFloat.zeros(*output_shape))
        @params[:running_var] = @running_var = Param.new(Xumo::SFloat.zeros(*output_shape))
      end

      def forward(x, learning_phase)
        if learning_phase
          mean = x.mean(axis: @axis, keepdims: true)
          @xc = x - mean
          var = (@xc**2).mean(axis: @axis, keepdims: true)
          @std = NMath.sqrt(var + 1e-7)
          xn = @xc / @std
          @xn = xn
          @running_mean.data = @momentum * @running_mean.data + (1 - @momentum) * mean
          @running_var.data = @momentum * @running_var.data + (1 - @momentum) * var
        else
          xc = x - @running_mean.data
          xn = xc / NMath.sqrt(@running_var.data + 1e-7)
        end
        @gamma.data * xn + @beta.data
      end
    
      def backward(dy)
        batch_size = dy.shape[@axis]
        @beta.grad = dy.sum(axis: @axis, keepdims: true)
        @gamma.grad = (@xn * dy).sum(axis: @axis, keepdims: true)
        dxn = @gamma.data * dy
        dxc = dxn / @std
        dstd = -((dxn * @xc) / (@std**2)).sum(axis: @axis, keepdims: true)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / batch_size) * @xc * dvar
        dmean = dxc.sum(axis: @axis, keepdims: true)
        dxc - dmean / batch_size
      end

      def to_hash
        super({axis: @axis, momentum: @momentum})
      end
    end
  end
  
end
