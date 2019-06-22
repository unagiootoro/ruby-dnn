module DNN
  module Optimizers

    # Super class of all optimizer classes.
    class Optimizer
      # @return [Float] Return the Learning rate.
      attr_accessor :learning_rate

      def initialize(learning_rate)
        @learning_rate = learning_rate
      end

      # Update layers has param.
      def update(layers)
        target_params = layers.select { |layer| layer.is_a?(HasParamLayer) && layer.trainable }
                              .map { |layer| layer.params.values }.flatten
                              .select { |param| param.grad }
        target_params.each do |param|
          update_param(param)
          param.grad = 0
        end
      end

      def to_hash(merge_hash = nil)
        hash = {class: self.class.name, learning_rate: @learning_rate}
        hash.merge!(merge_hash) if merge_hash
        hash
      end

      # Update param.
      # Classes that inherit from this class must implement this method.
      private def update_param(param)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'update_param'")
      end
    end


    class SGD < Optimizer
      # @return [Float] Return the momentum coefficient.
      attr_accessor :momentum

      def self.from_hash(hash)
        self.new(hash[:learning_rate], momentum: hash[:momentum])
      end

      # @param [Float] learning_rate Learning rate.
      # @param [Float] momentum momentum coefficient.
      def initialize(learning_rate = 0.01, momentum: 0)
        super(learning_rate)
        @momentum = momentum
        @v = {}
      end

      def to_hash
        super(momentum: @momentum)
      end

      private def update_param(param)
        amount = param.grad * @learning_rate
        if @momentum > 0
          @v[param] ||= 0
          amount += @momentum * @v[param]
          @v[param] = amount
        end
        param.data -= amount
      end
    end


    class Nesterov < Optimizer
      attr_accessor :momentum
      
      def self.from_hash(hash)
        self.new(hash[:learning_rate], momentum: hash[:momentum])
      end

      # @param [Float] learning_rate Learning rate.
      # @param [Float] momentum momentum coefficient.
      def initialize(learning_rate = 0.01, momentum: 0.9)
        super(learning_rate)
        @momentum = momentum
        @v = {}
      end

      def to_hash
        super(momentum: @momentum)
      end
    
      private def update_param(param)
        @v[param] ||= 0
        amount = param.grad * @learning_rate
        @v[param] = @v[param] * @momentum - amount
        param.data = (param.data + @momentum**2 * @v[param]) - (1 + @momentum) * amount
      end
    end
    
    
    class AdaGrad < Optimizer
      # @return [Float] Return the eps value.
      attr_accessor :eps

      # @param [Float] learning_rate Learning rate.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(learning_rate = 0.01, eps: 1e-7)
        super(learning_rate)
        @eps = eps
        @g = {}
      end

      def self.from_hash(hash)
        self.new(hash[:learning_rate], eps: hash[:eps])
      end
    
      private def update_param(param)
        @g[param] ||= 0
        @g[param] += param.grad**2
        param.data -= (@learning_rate / NMath.sqrt(@g[param] + @eps)) * param.grad
      end

      def to_hash
        super(eps: @eps)
      end
    end
    

    class RMSProp < Optimizer
      # @return [Float] Return the alpha value.
      attr_accessor :alpha
      # @return [Float] Return the eps value.
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(hash[:learning_rate], alpha: hash[:alpha], eps: hash[:eps])
      end

      # @param [Float] learning_rate Learning rate.
      # @param [Float] alpha Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(learning_rate = 0.001, alpha: 0.9, eps: 1e-7)
        super(learning_rate)
        @alpha = alpha
        @eps = eps
        @g = {}
      end

      def to_hash
        super(alpha: @alpha, eps: @eps)
      end

      private def update_param(param)
        @g[param] ||= 0
        @g[param] = @alpha * @g[param] + (1 - @alpha) * param.grad**2
        param.data -= (@learning_rate / NMath.sqrt(@g[param] + @eps)) * param.grad
      end
    end


    class AdaDelta < Optimizer
      # @return [Float] Return the rho value.
      attr_accessor :rho
      # @return [Float] Return the eps value.
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(rho: hash[:rho], eps: hash[:eps])
      end

      # @param [Float] rho Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(rho: 0.95, eps: 1e-6)
        super(nil)
        @rho = rho
        @eps = eps
        @h = {}
        @s = {}
      end

      def to_hash
        super(rho: @rho, eps: @eps)
      end

      private def update_param(param)
        @h[param] ||= Xumo::SFloat.zeros(*param.data.shape)
        @s[param] ||= Xumo::SFloat.zeros(*param.data.shape)
        @h[param] = @rho * @h[param] + (1 - @rho) * param.grad**2
        v = (NMath.sqrt(@s[param] + @eps) / NMath.sqrt(@h[param] + @eps)) * param.grad
        @s[param] = @rho * @s[param] + (1 - @rho) * v**2
        param.data -= v
      end
    end


    class Adam < Optimizer
      # @return [Float] Return the alpha value.
      attr_accessor :alpha
      # @return [Float] Return the beta1 value.
      attr_accessor :beta1
      # @return [Float] Return the beta2 value.
      attr_accessor :beta2
      # @return [Float] Return the eps value.
      attr_accessor :eps
      
      def self.from_hash(hash)
        self.new(alpha: hash[:alpha], beta1: hash[:beta1], beta2: hash[:beta2], eps: hash[:eps])
      end

      # @param [Float] alpha Value used to calculate learning rate.
      # @param [Float] beta1 Moving average index of beta1.
      # @param [Float] beta2 Moving average index of beta2.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(alpha: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-7)
        super(nil)
        @alpha = alpha
        @beta1 = beta1
        @beta2 = beta2
        @eps = eps
        @iter = 0
        @m = {}
        @v = {}
      end

      def update(layers)
        @iter += 1
        learning_rate = @alpha * Math.sqrt(1 - @beta2**@iter) / (1 - @beta1**@iter) 
        target_params = layers.select { |layer| layer.is_a?(HasParamLayer) && layer.trainable }
                              .map { |layer| layer.params.values }.flatten
                              .select { |param| param.grad }
        target_params.each do |param|
          update_param(param, learning_rate)
          param.grad = 0
        end
      end

      def to_hash
        super(alpha: @alpha, beta1: @beta1, beta2: @beta2, eps: @eps)
      end

      private def update_param(param, learning_rate)
        @m[param] ||= 0
        @v[param] ||= 0
        @m[param] += (1 - @beta1) * (param.grad - @m[param])
        @v[param] += (1 - @beta2) * (param.grad**2 - @v[param])
        param.data -= learning_rate * @m[param] / NMath.sqrt(@v[param] + @eps)
      end
    end


    class RMSPropGraves < Optimizer
      # @return [Float] Return the alpha value.
      attr_accessor :alpha
      # @return [Float] Return the eps value.
      attr_accessor :eps
      
      def self.from_hash(hash)
        self.new(hash[:learning_rate], alpha: hash[:alpha], eps: hash[:eps])
      end

      # @param [Float] learning_rate Learning rate.
      # @param [Float] alpha Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(learning_rate = 0.0001, alpha: 0.95, eps: 0.0001)
        super(learning_rate)
        @alpha = alpha
        @eps = eps
        @m = {}
        @v = {}
      end

      def to_hash
        super(alpha: @alpha, eps: @eps)
      end

      private def update_param(param)
        @m[param] ||= 0
        @v[param] ||= 0
        @m[param] = @alpha * @m[param] + (1 - @alpha) * param.grad
        @v[param] = @alpha * @v[param] + (1 - @alpha) * param.grad**2
        param.data -= (@learning_rate / NMath.sqrt(@v[param] - @m[param]**2 + @eps)) * param.grad
      end
    end

  end
end
