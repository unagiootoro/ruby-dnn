module DNN
  module Layers

    module LayerNode
      def forward(*inputs)
        xs = inputs.map(&:data)
        prevs = inputs.map { |input| input.is_a?(Tensor) ? input.link : input }
        ys = forward_node(*xs)
        num_outputs = (ys.is_a?(Array) ? ys.length : 1)
        link = Link.new(prevs: prevs, layer_node: self, num_outputs: num_outputs)
        prevs.map { |prev| prev.next = link if prev.is_a?(Link) }
        Tensor.new(ys, link)
      end

      def forward_node(*xs)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward_node'"
      end

      def backward_node(*dys)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'backward_node'"
      end
    end

    # Super class of all layer classes.
    class Layer
      attr_reader :input_shape
      attr_reader :output_shape

      def self.call(x, *args, **kwargs)
        new(*args, **kwargs).(x)
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
        input = Tensor.new(input) if !input.is_a?(Tensor) && !input.is_a?(Param)
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

      def forward(x)
        @x = x
        y = x.dot(@weight)
        y += @bias if @bias
        y
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
      def forward(x)
        Functions::FunctionSpace.reshape(x, [x.shape[0], x.shape[1..-1].reduce(:*)])
      end
    end

    class Reshape < Layer
      def initialize(shape)
        super()
        @shape = shape
      end

      def compute_output_shape
        @shape
      end

      def forward(x)
        Functions::FunctionSpace.reshape(x, [x.shape[0], *@shape])
      end

      def to_hash
        super(shape: @shape)
      end

      def load_hash(hash)
        initialize(hash[:shape])
      end
    end

    class Dropout < Layer
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

      def forward(x)
        if DNN.learning_phase
          Xumo::SFloat.srand(@rnd.rand(1 << 31))
          mask = Tensor.new(Xumo::SFloat.cast(Xumo::SFloat.new(*x.shape).rand >= @dropout_ratio))
          x = x * mask
        elsif @use_scale
          x *= (1 - @dropout_ratio)
        end
        x
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
