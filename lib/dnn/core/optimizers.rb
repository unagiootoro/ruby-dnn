module DNN
  module Optimizers

    # Super class of all optimizer classes.
    class Optimizer
      # @return [Float] Return the learning rate.
      attr_accessor :lr
      # @return [Float] Return the gradient clip value.
      attr_accessor :clip_norm

      def initialize(lr, clip_norm: nil)
        @lr = lr
        @clip_norm = clip_norm
      end

      # Update layers has param.
      def update(layers)
        target_params = layers.select { |layer| layer.is_a?(HasParamLayer) && layer.trainable }
                              .map { |layer| layer.get_params.values }.flatten.compact
                              .select { |param| param.grad }
        clipping(target_params) if @clip_norm
        update_params(target_params)
        target_params.each do |param|
          param.grad = 0
        end
      end

      def to_hash(merge_hash = nil)
        hash = {class: self.class.name, lr: @lr, clip_norm: @clip_norm}
        hash.merge!(merge_hash) if merge_hash
        hash
      end

      # Update params.
      private def update_params(params)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'update_param'")
      end

      private def clipping(params)
        norm = Math.sqrt(params.reduce(0) { |sum, param| sum + (param.grad == 0 ? 0 : (param.grad**2).sum) })
        return if norm <= @clip_norm
        rate = @clip_norm / (norm + 1e-7)
        params.each do |param|
          param.grad *= rate
        end
      end
    end


    class SGD < Optimizer
      # @return [Float] Return the momentum coefficient.
      attr_accessor :momentum

      def self.from_hash(hash)
        self.new(hash[:lr], momentum: hash[:momentum], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] momentum momentum coefficient.
      def initialize(lr = 0.01, momentum: 0, clip_norm: nil)
        super(lr, clip_norm: clip_norm)
        @momentum = momentum
        @v = {}
      end

      def to_hash
        super(momentum: @momentum)
      end

      private def update_params(params)
        params.each do |param|
          amount = param.grad * @lr
          if @momentum > 0
            @v[param] ||= 0
            amount += @momentum * @v[param]
            @v[param] = amount
          end
          param.data -= amount
        end
      end
    end


    class Nesterov < Optimizer
      attr_accessor :momentum
      
      def self.from_hash(hash)
        self.new(hash[:lr], momentum: hash[:momentum], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] momentum momentum coefficient.
      def initialize(lr = 0.01, momentum: 0.9, clip_norm: nil)
        super(lr, clip_norm: clip_norm)
        @momentum = momentum
        @v = {}
      end

      def to_hash
        super(momentum: @momentum)
      end
    
      private def update_params(params)
        params.each do |param|
          @v[param] ||= 0
          amount = param.grad * @lr
          @v[param] = @v[param] * @momentum - amount
          param.data = (param.data + @momentum**2 * @v[param]) - (1 + @momentum) * amount
        end
      end
    end
    
    
    class AdaGrad < Optimizer
      # @return [Float] Return the eps value.
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(hash[:lr], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(lr = 0.01, eps: 1e-7, clip_norm: nil)
        super(lr, clip_norm: clip_norm)
        @eps = eps
        @g = {}
      end
    
      private def update_params(params)
        params.each do |param|
          @g[param] ||= 0
          @g[param] += param.grad**2
          param.data -= (@lr / Xumo::NMath.sqrt(@g[param] + @eps)) * param.grad
        end
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
        self.new(hash[:lr], alpha: hash[:alpha], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] alpha Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(lr = 0.001, alpha: 0.9, eps: 1e-7, clip_norm: nil)
        super(lr, clip_norm: clip_norm)
        @alpha = alpha
        @eps = eps
        @g = {}
      end

      def to_hash
        super(alpha: @alpha, eps: @eps)
      end

      private def update_params(params)
        params.each do |param|
          @g[param] ||= 0
          @g[param] = @alpha * @g[param] + (1 - @alpha) * param.grad**2
          param.data -= (@lr / Xumo::NMath.sqrt(@g[param] + @eps)) * param.grad
        end
      end
    end


    class AdaDelta < Optimizer
      # @return [Float] Return the rho value.
      attr_accessor :rho
      # @return [Float] Return the eps value.
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(rho: hash[:rho], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] rho Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(rho: 0.95, eps: 1e-6, clip_norm: nil)
        super(nil, clip_norm: clip_norm)
        @rho = rho
        @eps = eps
        @h = {}
        @s = {}
      end

      def to_hash
        super(rho: @rho, eps: @eps)
      end

      private def update_params(params)
        params.each do |param|
          @h[param] ||= Xumo::SFloat.zeros(*param.data.shape)
          @s[param] ||= Xumo::SFloat.zeros(*param.data.shape)
          @h[param] = @rho * @h[param] + (1 - @rho) * param.grad**2
          v = (Xumo::NMath.sqrt(@s[param] + @eps) / Xumo::NMath.sqrt(@h[param] + @eps)) * param.grad
          @s[param] = @rho * @s[param] + (1 - @rho) * v**2
          param.data -= v
        end
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
        self.new(alpha: hash[:alpha], beta1: hash[:beta1], beta2: hash[:beta2], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] alpha Value used to calculate learning rate.
      # @param [Float] beta1 Moving average index of beta1.
      # @param [Float] beta2 Moving average index of beta2.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(alpha: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-7, clip_norm: nil)
        super(nil, clip_norm: clip_norm)
        @alpha = alpha
        @beta1 = beta1
        @beta2 = beta2
        @eps = eps
        @t = 0
        @m = {}
        @v = {}
      end

      def to_hash
        super(alpha: @alpha, beta1: @beta1, beta2: @beta2, eps: @eps)
      end

      private def update_params(params)
        @t += 1
        lr = @alpha * Math.sqrt(1 - @beta2**@t) / (1 - @beta1**@t) 
        params.each do |param|
          @m[param] ||= 0
          @v[param] ||= 0
          @m[param] += (1 - @beta1) * (param.grad - @m[param])
          @v[param] += (1 - @beta2) * (param.grad**2 - @v[param])
          param.data -= lr * @m[param] / Xumo::NMath.sqrt(@v[param] + @eps)
        end
      end
    end


    class RMSPropGraves < Optimizer
      # @return [Float] Return the alpha value.
      attr_accessor :alpha
      # @return [Float] Return the eps value.
      attr_accessor :eps
      
      def self.from_hash(hash)
        self.new(hash[:lr], alpha: hash[:alpha], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] alpha Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(lr = 0.0001, alpha: 0.95, eps: 0.0001, clip_norm: nil)
        super(lr, clip_norm: clip_norm)
        @alpha = alpha
        @eps = eps
        @m = {}
        @v = {}
      end

      def to_hash
        super(alpha: @alpha, eps: @eps)
      end

      private def update_params(params)
        params.each do |param|
          @m[param] ||= 0
          @v[param] ||= 0
          @m[param] = @alpha * @m[param] + (1 - @alpha) * param.grad
          @v[param] = @alpha * @v[param] + (1 - @alpha) * param.grad**2
          param.data -= (@lr / Xumo::NMath.sqrt(@v[param] - @m[param]**2 + @eps)) * param.grad
        end
      end
    end

  end
end
