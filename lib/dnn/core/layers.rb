module DNN
  module Layers

    # Super class of all optimizer classes.
    class Layer
      # @return [Bool] learning_phase Return the true if learning.
      attr_accessor :learning_phase
      # @return [Array] Return the shape of the input data.
      attr_reader :input_shape

      def initialize
        @built = false
      end

      # Build the layer.
      # @param [Array] input_shape Setting the shape of the input data.
      def build(input_shape)
        @input_shape = input_shape
        @learning_phase = true
        @built = true
      end
      
      # Does the layer have already been built?
      # @return [Bool] If layer have already been built then return true.
      def built?
        @built
      end

      # Forward propagation.
      def forward(x)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward'")
      end

      # Backward propagation.
      def backward(dy)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'backward'")
      end

      # Please reimplement this method as needed.
      # The default implementation return input_shape.
      # @return [Array] Return the shape of the output data.
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
      # @return [DNN::Initializers::Initializer] Weight initializer.
      attr_reader :weight_initializer
      # @return [DNN::Initializers::Initializer] Bias initializer.
      attr_reader :bias_initializer
      # @return [DNN::Regularizers::Regularizer] Weight regularization.
      attr_reader :weight_regularizer
      # @return [DNN::Regularizers::Regularizer] Bias regularization.
      attr_reader :bias_regularizer

      # @param [DNN::Initializers::Initializer] weight_initializer Weight initializer.
      # @param [DNN::Initializers::Initializer] bias_initializer Bias initializer.
      # @param [DNN::Regularizers::Regularizer] weight_regularizer Weight regularization.
      # @param [DNN::Regularizers::Regularizer] bias_regularizer Bias regularization.
      # @param [Bool] use_bias whether to use bias.
      def initialize(weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true)
        super()
        @weight_initializer = weight_initializer
        @bias_initializer = bias_initializer
        @weight_regularizer = weight_regularizer
        @bias_regularizer = bias_regularizer
        @params[:weight] = @weight = Param.new(nil, 0)
        if use_bias
          @params[:bias] = @bias = Param.new(nil, 0)
        else
          @bias = nil
        end
      end

      def regularizers
        regularizers = []
        regularizers << @weight_regularizer if @weight_regularizer
        regularizers << @bias_regularizer if @bias_regularizer
        regularizers
      end

      # @return [Bool] Return whether to use bias.
      def use_bias
        @bias ? true : false
      end

      def to_hash(merge_hash)
        super({weight_initializer: @weight_initializer.to_hash,
               bias_initializer: @bias_initializer.to_hash,
               weight_regularizer: @weight_regularizer&.to_hash,
               bias_regularizer: @bias_regularizer&.to_hash,
               use_bias: use_bias}.merge(merge_hash))
      end

      private def init_weight_and_bias
        @weight_initializer.init_param(self, @weight)
        @weight_regularizer.param = @weight if @weight_regularizer
        if @bias
          @bias_initializer.init_param(self, @bias)
          @bias_regularizer.param = @bias if @bias_regularizer
        end
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
                 weight_regularizer: Utils.from_hash(hash[:weight_regularizer]),
                 bias_regularizer: Utils.from_hash(hash[:bias_regularizer]),
                 use_bias: hash[:use_bias])
      end

      # @param [Integer] num_nodes number of nodes.
      def initialize(num_nodes,
                     weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              weight_regularizer: weight_regularizer, bias_regularizer: bias_regularizer, use_bias: use_bias)
        @num_nodes = num_nodes
      end

      def build(input_shape)
        unless input_shape.length == 1
          raise DNN_ShapeError.new("Input shape is #{input_shape}. But input shape must be 1 dimensional.")
        end
        super
        num_prev_nodes = input_shape[0]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes)
        @bias.data = Xumo::SFloat.new(@num_nodes) if @bias
        init_weight_and_bias
      end

      def forward(x)
        @x = x
        y = x.dot(@weight.data)
        y += @bias.data if @bias
        y
      end
    
      def backward(dy)
        if @trainable
          @weight.grad += @x.transpose.dot(dy)
          @bias.grad += dy.sum(0) if @bias
        end
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
      attr_accessor :dropout_ratio
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

      def forward(x)
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
    
  end
  
end
