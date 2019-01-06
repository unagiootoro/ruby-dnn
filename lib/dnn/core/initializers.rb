module DNN
  module Initializers

    class Initializer
      # Classes that inherit from this class must implement this method.
      # def init_param(param) end

      def to_hash(merge_hash = nil)
        hash = {class: self.class.name}
        hash.merge!(merge_hash) if merge_hash
        hash
      end
    end


    class Zeros < Initializer
      def init_param(param)
        param.data = param.data.fill(0)
      end
    end
    
    
    class RandomNormal < Initializer
      attr_reader :mean
      attr_reader :std
      
      def self.load_hash(hash)
        self.new(hash[:mean], hash[:std])
      end

      def initialize(mean = 0, std = 0.05)
        @mean = mean
        @std = std
      end

      def init_param(param)
        param.data = param.data.rand_norm(@mean, @std)
      end

      def to_hash
        super({mean: @mean, std: @std})
      end
    end


    class RandomUniform < Initializer
      attr_reader :min
      attr_reader :max

      def self.load_hash(hash)
        self.new(hash[:min], hash[:max])
      end

      def initialize(min = -0.05, max = 0.05)
        @min = min
        @max = max
      end

      def init_param(param)
        param.data = param.data.rand(@min, @max)
      end

      def to_hash
        super({min: @min, max: @max})
      end
    end
    
    
    class Xavier < Initializer
      def init_param(param)
        num_prev_nodes = param.layer.prev_layer.shape.reduce(:*)
        param.data = param.data.rand_norm / Math.sqrt(num_prev_nodes)
      end
    end
    
    
    class He < Initializer
      def init_param(param)
        num_prev_nodes = param.layer.prev_layer.shape.reduce(:*)
        param.data = param.data.rand_norm / Math.sqrt(num_prev_nodes) * Math.sqrt(2)
      end
    end

  end
end
