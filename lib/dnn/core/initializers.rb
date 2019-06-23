module DNN
  module Initializers

    class Initializer
      def initialize(seed = false)
        @seed = seed == true ? rand(1 << 31) : seed
      end

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
        self.new(hash[:mean], hash[:std], hash[:seed])
      end

      def initialize(mean = 0, std = 0.05, seed = true)
        super(seed)
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
        self.new(hash[:min], hash[:max], hash[:seed])
      end

      def initialize(min = -0.05, max = 0.05, seed = true)
        super(seed)
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
      def initialize(seed = true)
        super
      end

      def init_param(layer, param)
        Xumo::SFloat.srand(@seed)
        num_prev_nodes = layer.input_shape.reduce(:*)
        param.data = param.data.rand_norm / Math.sqrt(num_prev_nodes)
      end
    end
    
    
    class He < Initializer
      def initialize(seed = true)
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
