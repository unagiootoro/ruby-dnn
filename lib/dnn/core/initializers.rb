module DNN
  module Initializers

    class Initializer
      # @param [Boolean | Integer] seed Seed of random number used for masking.
      #                            Set true to determine seed as random.
      def initialize(seed: false)
        @seed = seed == true ? rand(1 << 31) : seed
      end

      # Initialization of learning parameters.
      # @param [DNN::Layers::Layer] layer Layer that owns learning parameters.
      # @param [DNN::Param] param Learning parameter to be initialized.
      def init_param(layer, param)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'init_params'")
      end

      def to_hash(merge_hash = nil)
        hash = {class: self.class.name, seed: @seed}
        hash.merge!(merge_hash) if merge_hash
        hash
      end
    end


    class Zeros < Initializer
      def init_param(layer, param)
        param.data = param.data.fill(0)
      end
    end


    class Const < Initializer
      attr_reader :const

      def self.from_hash(hash)
        self.new(hash[:const])
      end

      # @param [Float] const Constant value of initialization.
      def initialize(const)
        super()
        @const = const
      end

      def init_param(layer, param)
        param.data = param.data.fill(@const)
      end

      def to_hash
        super({const: @const})
      end
    end
    
    
    class RandomNormal < Initializer
      attr_reader :mean
      attr_reader :std
      
      def self.from_hash(hash)
        self.new(hash[:mean], hash[:std], seed: hash[:seed])
      end

      # @param [Float] mean Average of initialization value.
      # @param [Float] std Variance of initialization value.
      def initialize(mean = 0, std = 0.05, seed: true)
        super(seed: seed)
        @mean = mean
        @std = std
      end

      def init_param(layer, param)
        Xumo::SFloat.srand(@seed)
        param.data = param.data.rand_norm(@mean, @std)
      end

      def to_hash
        super({mean: @mean, std: @std})
      end
    end


    class RandomUniform < Initializer
      attr_reader :min
      attr_reader :max

      def self.from_hash(hash)
        self.new(hash[:min], hash[:max], seed: hash[:seed])
      end

      # @param [Float] min Min of initialization value.
      # @param [Float] max Max of initialization value.
      def initialize(min = -0.05, max = 0.05, seed: true)
        super(seed: seed)
        @min = min
        @max = max
      end

      def init_param(layer, param)
        Xumo::SFloat.srand(@seed)
        param.data = param.data.rand(@min, @max)
      end

      def to_hash
        super({min: @min, max: @max})
      end
    end
    
    
    class Xavier < Initializer
      def initialize(seed: true)
        super
      end

      def init_param(layer, param)
        Xumo::SFloat.srand(@seed)
        num_prev_nodes = layer.input_shape.reduce(:*)
        param.data = param.data.rand_norm / Math.sqrt(num_prev_nodes)
      end
    end
    
    
    class He < Initializer
      def initialize(seed: true)
        super
      end

      def init_param(layer, param)
        Xumo::SFloat.srand(@seed)
        num_prev_nodes = layer.input_shape.reduce(:*)
        param.data = param.data.rand_norm / Math.sqrt(num_prev_nodes) * Math.sqrt(2)
      end
    end

  end
end
