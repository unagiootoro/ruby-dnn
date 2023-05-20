module DNN
  module Layers

    # Super class of all layer classes.
    class Layer
      attr_reader :input_shapes

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
        @learning_phase = false
      end

      # Forward propagation and create a link.
      # @param [Array] inputs Input tensor or param list.
      # @return [Tensor] Output tensor.
      def call(*inputs)
        inputs.compact!
        build(*inputs.map { |input| input.shape[1..-1] }) unless built?
        forward(*inputs)
      end

      # Build the layer.
      # @param [Array] input_shapes Setting the shape of the input datas.
      def build(*input_shapes)
        @input_shapes = input_shapes
        @built = true
      end

      # @return [Boolean] If layer have already been built then return true.
      def built?
        @built
      end

      # Forward propagation.
      # @param [Array] inputs Input tensor or param.
      # @return [Tensor | Array] Output tensor or it list.
      def forward(*inputs)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward'"
      end

      def <<(tensor)
        self.(tensor)
      end

      # @return [Boolean] Returns whether the layer is in the learning phase.
      def learning_phase?
        @learning_phase
      end

      # @return [Boolean] learning_phase Specifies whether it is in the learning phase.
      def set_learning_phase(learning_phase)
        @learning_phase = learning_phase
      end

      # @return [Boolean] Setting false prevents learning of parameters.
      def trainable?
        get_trainable_variables.each_value do |variable|
          return true if variable.requires_grad
        end
        false
      end

      # @param [Boolean] trainable Specifies whether to allow learning.
      def set_trainable(trainable)
        get_trainable_variables.each_value do |variable|
          variable.requires_grad = trainable
        end
      end

      # @return [Hash] The variables of the layer.
      def get_variables
        {}
      end

      # @return [Hash] The trainable variables of the layer.
      def get_trainable_variables
        {}
      end

      # Clean the layer state.
      def clean
        input_shapes = @input_shapes
        hash = to_hash
        params = get_variables
        instance_variables.each do |ivar|
          instance_variable_set(ivar, nil)
        end
        load_hash(hash)
        build(*input_shapes)
        params.each do |(key, param)|
          param.data = nil
          param.grad = Xumo::SFloat[0] if param.grad
          instance_variable_set("@#{key}", param)
        end
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
    end

    class InputLayer < Layer
      # @param [Array] input_dim_or_shape Setting the shape or dimension of the input data.
      def initialize(input_dim_or_shape)
        super()
        @input_shape = input_dim_or_shape.is_a?(Array) ? input_dim_or_shape : [input_dim_or_shape]
      end

      def build(input_shape)
        super(input_shape)
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
    class Connection < Layer
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
        @weight = Variable.new(nil, Xumo::SFloat[0])
        @bias = use_bias ? Variable.new(nil, Xumo::SFloat[0]) : nil
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

      def get_variables
        { weight: @weight, bias: @bias }
      end

      def get_trainable_variables
        { weight: @weight, bias: @bias }
      end

      private def init_weight_and_bias
        @weight_initializer.init_param(@weight, @input_shapes)
        @weight_regularizer.param = @weight if @weight_regularizer
        if @bias
          @bias_initializer.init_param(@bias, @input_shapes)
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
        if learning_phase?
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
