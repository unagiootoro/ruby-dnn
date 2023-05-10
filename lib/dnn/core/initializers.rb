module DNN
  module Initializers

    class Initializer
      def self.from_hash(hash)
        return nil unless hash
        initializer_class = DNN.const_get(hash[:class])
        initializer = initializer_class.allocate
        raise DNNError, "#{initializer.class} is not an instance of #{self} class." unless initializer.is_a?(self)
        initializer.load_hash(hash)
        initializer
      end

      # @param [Boolean | Integer] seed Seed of random number used for initialize parameter.
      #                                 Set true to determine seed as random.
      def initialize(seed: false)
        @seed = seed == true ? rand(1 << 31) : seed
      end

      # Initialization of learning parameters.
      # @param [DNN::Layers::Layer] layer Layer that owns learning parameters.
      # @param [DNN::Variable] param Learning parameter to be initialized.
      def init_param(param, input_shapes)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'init_param'"
      end

      def to_hash(merge_hash = nil)
        hash = { class: self.class.name, seed: @seed }
        hash.merge!(merge_hash) if merge_hash
        hash
      end

      def load_hash(hash)
        initialize
      end
    end

    class Zeros < Initializer
      def init_param(param, input_shapes)
        param.data = param.data.fill(0)
      end
    end

    class Const < Initializer
      attr_reader :const

      # @param [Float] const Constant value of initialization.
      def initialize(const)
        super()
        @const = const
      end

      def init_param(param, input_shapes)
        param.data = param.data.fill(@const)
      end

      def to_hash
        super(const: @const)
      end

      def load_hash(hash)
        initialize(hash[:const])
      end
    end

    class RandomNormal < Initializer
      attr_reader :mean
      attr_reader :std

      # @param [Float] mean Average of initialization value.
      # @param [Float] std Variance of initialization value.
      def initialize(mean = 0, std = 0.05, seed: true)
        super(seed: seed)
        @mean = mean
        @std = std
      end

      def init_param(param, input_shapes)
        Xumo::SFloat.srand(@seed)
        param.data = param.data.rand_norm(@mean, @std)
      end

      def to_hash
        super(mean: @mean, std: @std)
      end

      def load_hash(hash)
        initialize(hash[:mean], hash[:std], seed: hash[:seed])
      end
    end

    class RandomUniform < Initializer
      attr_reader :min
      attr_reader :max

      # @param [Float] min Min of initialization value.
      # @param [Float] max Max of initialization value.
      def initialize(min = -0.05, max = 0.05, seed: true)
        super(seed: seed)
        @min = min
        @max = max
      end

      def init_param(param, input_shapes)
        Xumo::SFloat.srand(@seed)
        param.data = param.data.rand(@min, @max)
      end

      def to_hash
        super(min: @min, max: @max)
      end

      def load_hash(hash)
        initialize(hash[:min], hash[:max], seed: hash[:seed])
      end
    end

    class Xavier < Initializer
      def initialize(seed: true)
        super
      end

      def init_param(param, input_shapes)
        Xumo::SFloat.srand(@seed)
        num_prev_units = input_shapes.reduce(0) { |result, input_shape| result + input_shape.reduce(:*) }
        param.data = param.data.rand_norm / Math.sqrt(num_prev_units)
      end
    end

    class He < Initializer
      def initialize(seed: true)
        super
      end

      def init_param(param, input_shapes)
        Xumo::SFloat.srand(@seed)
        num_prev_units = input_shapes.reduce(0) { |result, input_shape| result + input_shape.reduce(:*) }
        param.data = param.data.rand_norm / Math.sqrt(num_prev_units) * Math.sqrt(2)
      end
    end

  end
end
