module DNN
  module Initializers

    class Initializer
      def init_param(layer, param_key, param)
        layer.params[param_key] = param
      end
    end


    class Zeros < Initializer
      def init_param(layer, param_key)
        super(layer, param_key, layer.params[param_key].fill(0))
      end
    end
    
    
    class RandomNormal < Initializer
      def initialize(mean = 0, std = 0.05)
        @mean = mean
        @std = std
      end
    
      def init_param(layer, param_key)
        super(layer, param_key, layer.params[param_key].rand_norm(@mean, @std))
      end
    end
    
    
    class Xavier < Initializer
      def init_param(layer, param_key)
        prev_dim = layer.prev_layer.shape.reduce(:*)
        super(layer, param_key, layer.params[param_key].rand_norm / Math.sqrt(prev_dim))
      end
    end
    
    
    class He < Initializer
      def init_param(layer, param_key)
        prev_dim = layer.prev_layer.shape.reduce(:*)
        super(layer, param_key, layer.params[param_key].rand_norm / Math.sqrt(prev_dim) * Math.sqrt(2))
      end
    end

  end
end
