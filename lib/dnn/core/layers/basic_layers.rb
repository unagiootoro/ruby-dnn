module DNN
  module Layers

    module LayerNode
      def forward(input)
        x = input.data
        prev = (input.is_a?(Tensor) ? input.link : input)
        y = forward_node(x)
        link = Link.new(prev, self)
        prev.next = link if prev.is_a?(Link)
        Tensor.convert(y, link)
      end

      def forward_node(x)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward_node'"
      end

      def backward_node(dy)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'backward_node'"
      end
    end

    # Super class of all layer classes.
    class Layer
      attr_reader :input_shape
      attr_reader :output_shape

      def self.call(x, *args)
        new(*args).(x)
      end

      def self.from_hash(hash)
        return nil unless hash
        layer_class = DNN.const_get(hash[:class])
        layer = layer_class.allocate
        raise DNNError, "#{layer.class} is not an instance of #{self} class." unless layer.is_a?(self)
        layer.load_hash(hash)
        layer
      end

      def initialize
        @built = false
      end

      # Forward propagation and create a link.
      # @param [Tensor | Param] input Input tensor or param.
      # @return [Tensor] Output tensor.
      def call(input)
        input = Tensor.convert(input) if !input.is_a?(Tensor) && !input.is_a?(Param)
        build(input.data.shape[1..-1]) unless built?
        forward(input)
      end

      # Build the layer.
      # @param [Array] input_shape Setting the shape of the input data.
      def build(input_shape)
        @input_shape = input_shape
        @output_shape = compute_output_shape
        @built = true
      end

      # @return [Boolean] If layer have already been built then return true.
      def built?
        @built
      end

      # Forward propagation.
      # @param [Tensor] input Input tensor or param.
      # @return [Tensor] Output tensor.
      def forward(input)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward'"
      end

      # Please reimplement this method as needed.
      # The default implementation return input_shape.
      # @return [Array] Return the shape of the output data.
      def compute_output_shape
        @input_shape
      end

      def <<(tensor)
        self.(tensor)
      end

      # Layer to a hash.
      def to_hash(merge_hash = nil)
        hash = { class: self.class.name }
        hash.merge!(merge_hash) if merge_hash
        hash
      end

      def load_hash(hash)
        initialize
      end

      # Clean the layer state.
      def clean
        input_shape = @input_shape
        hash = to_hash
        instance_variables.each do |ivar|
          instance_variable_set(ivar, nil)
        end
        load_hash(hash)
        build(input_shape)
      end
    end

    # This class is a superclass of all classes with learning parameters.
    class TrainableLayer < Layer
      # @return [Boolean] Setting false prevents learning of parameters.
      attr_accessor :trainable

      def initialize
        super()
        @trainable = true
      end

      # @return [Array] The parameters of the layer.
      def get_params
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'get_params'"
      end

      def clean
        input_shape = @input_shape
        hash = to_hash
        params = get_params
        instance_variables.each do |ivar|
          instance_variable_set(ivar, nil)
        end
        load_hash(hash)
        build(input_shape)
        params.each do |(key, param)|
          param.data = nil
          param.grad = Xumo::SFloat[0] if param.grad
          instance_variable_set("@#{key}", param)
        end
      end
    end

    class InputLayer < Layer
      # @param [Array] input_dim_or_shape Setting the shape or dimension of the input data.
      def initialize(input_dim_or_shape)
        super()
        @input_shape = input_dim_or_shape.is_a?(Array) ? input_dim_or_shape : [input_dim_or_shape]
      end

      def build(input_shape)
        super(@input_shape)
      end

      def forward(x)
        unless x.shape[1..-1] == @input_shape
          raise DNNShapeError, "The shape of x does not match the input shape. input shape is #{@input_shape}, but x shape is #{x.shape[1..-1]}."
        end
        x
      end

      def to_proc
        method(:call).to_proc
      end

      def to_hash
        super(input_shape: @input_shape)
      end

      def load_hash(hash)
        initialize(hash[:input_shape])
      end
    end

    # It is a superclass of all connection layers.
    class Connection < TrainableLayer
      attr_reader :weight
      attr_reader :bias
      attr_reader :weight_initializer
      attr_reader :bias_initializer
      attr_reader :weight_regularizer
      attr_reader :bias_regularizer

      # @param [DNN::Initializers::Initializer] weight_initializer Weight initializer.
      # @param [DNN::Initializers::Initializer] bias_initializer Bias initializer.
      # @param [DNN::Regularizers::Regularizer | NilClass] weight_regularizer Weight regularizer.
      # @param [DNN::Regularizers::Regularizer | NilClass] bias_regularizer Bias regularizer.
      # @param [Boolean] use_bias Whether to use bias.
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
        @weight = Param.new(nil, Xumo::SFloat[0])
        @bias = use_bias ? Param.new(nil, Xumo::SFloat[0]) : nil
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
      include LayerNode

      attr_reader :num_units

      # @param [Integer] num_units Number of nodes.
      def initialize(num_units,
                     weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              weight_regularizer: weight_regularizer, bias_regularizer: bias_regularizer, use_bias: use_bias)
        @num_units = num_units
      end

      def build(input_shape)
        unless input_shape.length == 1
          raise DNNShapeError, "Input shape is #{input_shape}. But input shape must be 1 dimensional."
        end
        super
        num_prev_units = input_shape[0]
        @weight.data = Xumo::SFloat.new(num_prev_units, @num_units)
        @bias.data = Xumo::SFloat.new(@num_units) if @bias
        init_weight_and_bias
      end

      def forward_node(x)
        @x = x
        y = x.dot(@weight.data)
        y += @bias.data if @bias
        y
      end

      def backward_node(dy)
        if @trainable
          @weight.grad += @x.transpose.dot(dy)
          @bias.grad += dy.sum(0) if @bias
        end
        dy.dot(@weight.data.transpose)
      end

      def compute_output_shape
        [@num_units]
      end

      def to_hash
        super(num_units: @num_units)
      end

      def load_hash(hash)
        initialize(hash[:num_units],
                   weight_initializer: Initializers::Initializer.from_hash(hash[:weight_initializer]),
                   bias_initializer: Initializers::Initializer.from_hash(hash[:bias_initializer]),
                   weight_regularizer: Regularizers::Regularizer.from_hash(hash[:weight_regularizer]),
                   bias_regularizer: Regularizers::Regularizer.from_hash(hash[:bias_regularizer]),
                   use_bias: hash[:use_bias])
      end
    end

    class Flatten < Layer
      include LayerNode

      def forward_node(x)
        x.reshape(x.shape[0], *@output_shape)
      end

      def backward_node(dy)
        dy.reshape(dy.shape[0], *@input_shape)
      end

      def compute_output_shape
        [@input_shape.reduce(:*)]
      end
    end

    class Reshape < Layer
      include LayerNode

      def initialize(shape)
        super()
        @shape = shape
      end

      def compute_output_shape
        @shape
      end

      def forward_node(x)
        x.reshape(x.shape[0], *@output_shape)
      end

      def backward_node(dy)
        dy.reshape(dy.shape[0], *@input_shape)
      end

      def to_hash
        super(shape: @shape)
      end

      def load_hash(hash)
        initialize(hash[:shape])
      end
    end

    class Lasso < Layer
      include LayerNode

      attr_accessor :l1_lambda

      # @param [Float] l1_lambda L1 regularizer coefficient.
      def initialize(l1_lambda = 0.01)
        super()
        @l1_lambda = l1_lambda
      end

      def forward_node(x)
        @x = x
        @l1_lambda * x.abs.sum
      end

      def backward_node(dy)
        dx = Xumo::SFloat.ones(*@x.shape)
        dx[@x < 0] = -1
        @l1_lambda * dx
      end

      def to_hash
        super(l1_lambda: @l1_lambda)
      end

      def load_hash(hash)
        initialize(hash[:l1_lambda])
      end
    end

    class Ridge < Layer
      include LayerNode

      attr_accessor :l2_lambda

      # @param [Float] l2_lambda L2 regularizer coefficient.
      def initialize(l2_lambda = 0.01)
        super()
        @l2_lambda = l2_lambda
      end

      def forward_node(x)
        @x = x
        0.5 * @l2_lambda * (x**2).sum
      end

      def backward_node(dy)
        @l2_lambda * @x
      end

      def to_hash
        super(l2_lambda: @l2_lambda)
      end

      def load_hash(hash)
        initialize(hash[:l2_lambda])
      end
    end

    class Dropout < Layer
      include LayerNode

      attr_accessor :dropout_ratio
      attr_reader :use_scale

      # @param [Float] dropout_ratio Nodes dropout ratio.
      # @param [Integer] seed Seed of random number used for masking.
      # @param [Boolean] use_scale Set to true to scale the output according to the dropout ratio.
      def initialize(dropout_ratio = 0.5, seed: rand(1 << 31), use_scale: true)
        super()
        @dropout_ratio = dropout_ratio
        @seed = seed
        @use_scale = use_scale
        @mask = nil
        @rnd = Random.new(@seed)
      end

      def forward_node(x)
        if DNN.learning_phase
          Xumo::SFloat.srand(@rnd.rand(1 << 31))
          @mask = Xumo::SFloat.new(*x.shape).rand < @dropout_ratio
          x[@mask] = 0
        elsif @use_scale
          x *= (1 - @dropout_ratio)
        end
        x
      end

      def backward_node(dy)
        dy[@mask] = 0
        dy
      end

      def to_hash
        super(dropout_ratio: @dropout_ratio, seed: @seed, use_scale: @use_scale)
      end

      def load_hash(hash)
        initialize(hash[:dropout_ratio], seed: hash[:seed], use_scale: hash[:use_scale])
      end
    end

  end
end
