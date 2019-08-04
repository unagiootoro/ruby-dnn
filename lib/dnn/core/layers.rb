module DNN
  module Layers

    # Super class of all layer classes.
    class Layer
      attr_reader :input_shape

      def self.call(x, *args)
        self.new(*args).(x)
      end

      def initialize
        @built = false
      end

      # Forward propagation and create a link.
      # @param [Array] input Array of the form [x_input_data, prev_link].
      def call(input)
        x, prev_link, model = *input
        build(x.shape[1..-1]) unless built?
        y = forward(x)
        link = Link.new(prev_link, self)
        [y, link, model]
      end

      # Build the layer.
      # @param [Array] input_shape Setting the shape of the input data.
      def build(input_shape)
        @input_shape = input_shape
        @learning_phase = true
        @built = true
      end

      # @return [Boolean] If layer have already been built then return true.
      def built?
        @built
      end

      # Forward propagation.
      # @param [Numo::SFloat] x Input data.
      def forward(x)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward'")
      end

      # Backward propagation.
      # @param [Numo::SFloat] dy Differential value of output data.
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
        hash = { class: self.class.name }
        hash.merge!(merge_hash) if merge_hash
        hash
      end
    end


    # This class is a superclass of all classes with learning parameters.
    class HasParamLayer < Layer
      # @return [Boolean] Setting false prevents learning of parameters.
      attr_accessor :trainable

      def initialize
        super()
        @trainable = true
      end

      # @return [Array] The parameters of the layer.
      def get_params
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'get_params'")
      end
    end


    class InputLayer < Layer
      def self.call(input)
        shape = input.is_a?(Array) ? input[0].shape : input.shape
        self.new(shape[1..-1]).(input)
      end

      def self.from_hash(hash)
        self.new(hash[:input_shape])
      end

      # @param [Array] input_dim_or_shape Setting the shape or dimension of the input data.
      def initialize(input_dim_or_shape)
        super()
        @input_shape = input_dim_or_shape.is_a?(Array) ? input_dim_or_shape : [input_dim_or_shape]
      end

      def call(input)
        build
        x, prev_link, model = *input
        link = prev_link ? Link.new(prev_link, self) : Link.new(nil, self)
        [forward(x), link, model]
      end

      def build
        @built = true
        @input_shape
      end

      def forward(x)
        unless x.shape[1..-1] == @input_shape
          raise DNN_ShapeError.new("The shape of x does not match the input shape. input shape is #{@input_shape}, but x shape is #{x.shape[1..-1]}.")
        end
        x
      end

      def backward(dy)
        dy
      end

      def to_hash
        super(input_shape: @input_shape)
      end
    end


    # It is a superclass of all connection layers.
    class Connection < HasParamLayer
      attr_reader :weight
      attr_reader :bias
      attr_reader :weight_initializer
      attr_reader :bias_initializer
      attr_reader :weight_regularizer
      attr_reader :bias_regularizer

      # @param [DNN::Initializers::Initializer] weight_initializer Weight initializer.
      # @param [DNN::Initializers::Initializer] bias_initializer Bias initializer.
      # @param [DNN::Regularizers::Regularizer | NilClass] weight_regularizer Weight regularization.
      # @param [DNN::Regularizers::Regularizer | NilClass] bias_regularizer Bias regularization.
      # @param [Boolean] use_bias whether to use bias.
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
        @weight = Param.new(nil, 0)
        if use_bias
          @bias = Param.new(nil, 0)
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

      # @return [Boolean] Return whether to use bias.
      def use_bias
        @bias ? true : false
      end

      def to_hash(merge_hash)
        super({ weight_initializer: @weight_initializer.to_hash,
                bias_initializer: @bias_initializer.to_hash,
                weight_regularizer: @weight_regularizer&.to_hash,
                bias_regularizer: @bias_regularizer&.to_hash,
                use_bias: use_bias }.merge(merge_hash))
      end

      def get_params
        { weight: @weight, bias: @bias }
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


    class Dense < Connection
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
        super(num_nodes: @num_nodes)
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
        super(output_shape: @output_shape)
      end
    end


    class Dropout < Layer
      attr_accessor :dropout_ratio
      attr_reader :use_scale

      def self.from_hash(hash)
        self.new(hash[:dropout_ratio], seed: hash[:seed], use_scale: hash[:use_scale])
      end

      # @param [Float] dropout_ratio Nodes dropout ratio.
      # @param [Integer] seed Seed of random number used for masking.
      # @param [Boolean] use_scale Use 'weight scaling inference rule'.
      def initialize(dropout_ratio = 0.5, seed: rand(1 << 31), use_scale: true)
        super()
        @dropout_ratio = dropout_ratio
        @seed = seed
        @use_scale = use_scale
        @mask = nil
        @rnd = Random.new(@seed)
      end

      def call(input)
        x, prev_link, model = *input
        build(x.shape[1..-1]) unless built?
        y = forward(x, model.learning_phase)
        link = Link.new(prev_link, self)
        [y, link, model]
      end

      def forward(x, learning_phase)
        if learning_phase
          Xumo::SFloat.srand(@rnd.rand(1 << 31))
          @mask = Xumo::SFloat.ones(*x.shape).rand < @dropout_ratio
          x[@mask] = 0
        elsif @use_scale
          x *= (1 - @dropout_ratio)
        end
        x
      end

      def backward(dy)
        dy[@mask] = 0
        dy
      end

      def to_hash
        super(dropout_ratio: @dropout_ratio, seed: @seed, use_scale: @use_scale)
      end
    end

  end
end
