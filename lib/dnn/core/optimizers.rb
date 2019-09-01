module DNN
  module Optimizers

    # Super class of all optimizer classes.
    class Optimizer
      attr_reader :status
      attr_accessor :clip_norm

      def self.load(dumped)
        opt = Utils.hash_to_obj(dumped[:hash])
        dumped[:status].each do |key, state|
          state = state.clone
          opt.status[key] = state
          opt.instance_variable_set("@#{key}", state)
        end
        opt
      end

      # @param [Float | NilClass] clip_norm Gradient clip norm.
      def initialize(clip_norm: nil)
        @clip_norm = clip_norm
      end

      # Update layers has params.
      def update(layers)
        target_params = layers.select { |layer| layer.is_a?(HasParamLayer) && layer.trainable }
                              .map { |layer| layer.get_params.values }.flatten.compact
                              .select { |param| param.grad }
        clip_grads(target_params) if @clip_norm
        update_params(target_params)
        target_params.each do |param|
          param.grad = Xumo::SFloat.zeros(*param.data.shape)
        end
      end

      def dump
        { hash: to_hash, status: @status }
      end

      def to_hash(merge_hash = nil)
        hash = { class: self.class.name, clip_norm: @clip_norm }
        hash.merge!(merge_hash) if merge_hash
        hash
      end

      # Update params.
      private def update_params(params)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'update_params'")
      end

      private def clip_grads(params)
        norm = Math.sqrt(params.reduce(0) { |sum, param| sum + (param.grad == 0 ? 0 : (param.grad ** 2).sum) })
        return if norm <= @clip_norm
        rate = @clip_norm / (norm + 1e-7)
        params.each do |param|
          param.grad *= rate
        end
      end
    end


    class SGD < Optimizer
      attr_accessor :lr
      attr_accessor :momentum

      def self.from_hash(hash)
        self.new(hash[:lr], momentum: hash[:momentum], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] momentum Momentum coefficient.
      def initialize(lr = 0.01, momentum: 0, clip_norm: nil)
        super(clip_norm: clip_norm)
        @lr = lr
        @momentum = momentum
        @v = {}
        @status = { v: @v }
      end

      def to_hash
        super(lr: @lr, momentum: @momentum)
      end

      private def update_params(params)
        params.each do |param|
          amount = param.grad * @lr
          if @momentum > 0
            @v[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
            amount += @momentum * @v[param.tag]
            @v[param.tag] = amount
          end
          param.data -= amount
        end
      end
    end


    class Nesterov < Optimizer
      attr_accessor :lr
      attr_accessor :momentum

      def self.from_hash(hash)
        self.new(hash[:lr], momentum: hash[:momentum], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] momentum Momentum coefficient.
      def initialize(lr = 0.01, momentum: 0.9, clip_norm: nil)
        super(clip_norm: clip_norm)
        @lr = lr
        @momentum = momentum
        @v = {}
        @status = [:v]
      end

      def to_hash
        super(lr: @lr, momentum: @momentum)
      end

      private def update_params(params)
        params.each do |param|
          @v[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          amount = param.grad * @lr
          @v[param.tag] = @v[param.tag] * @momentum - amount
          param.data = (param.data + @momentum ** 2 * @v[param.tag]) - (1 + @momentum) * amount
        end
      end
    end


    class AdaGrad < Optimizer
      attr_accessor :lr
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(hash[:lr], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(lr = 0.01, eps: 1e-7, clip_norm: nil)
        super(clip_norm: clip_norm)
        @lr = lr
        @eps = eps
        @g = {}
        @status = { g: @g }
      end

      private def update_params(params)
        params.each do |param|
          @g[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @g[param.tag] += param.grad ** 2
          param.data -= (@lr / Xumo::NMath.sqrt(@g[param.tag] + @eps)) * param.grad
        end
      end

      def to_hash
        super(lr: @lr, eps: @eps)
      end
    end


    class RMSProp < Optimizer
      attr_accessor :lr
      attr_accessor :alpha
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(hash[:lr], alpha: hash[:alpha], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] alpha Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(lr = 0.001, alpha: 0.9, eps: 1e-7, clip_norm: nil)
        super(clip_norm: clip_norm)
        @lr = lr
        @alpha = alpha
        @eps = eps
        @g = {}
        @status = { g: @g }
      end

      def to_hash
        super(lr: @lr, alpha: @alpha, eps: @eps)
      end

      private def update_params(params)
        params.each do |param|
          @g[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @g[param.tag] = @alpha * @g[param.tag] + (1 - @alpha) * param.grad ** 2
          param.data -= (@lr / Xumo::NMath.sqrt(@g[param.tag] + @eps)) * param.grad
        end
      end
    end


    class AdaDelta < Optimizer
      attr_accessor :rho
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(rho: hash[:rho], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] rho Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(rho: 0.95, eps: 1e-6, clip_norm: nil)
        super(clip_norm: clip_norm)
        @rho = rho
        @eps = eps
        @h = {}
        @s = {}
        @status = { h: @h, s: @s }
      end

      def to_hash
        super(rho: @rho, eps: @eps)
      end

      private def update_params(params)
        params.each do |param|
          @h[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @s[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @h[param.tag] = @rho * @h[param.tag] + (1 - @rho) * param.grad ** 2
          v = (Xumo::NMath.sqrt(@s[param.tag] + @eps) / Xumo::NMath.sqrt(@h[param.tag] + @eps)) * param.grad
          @s[param.tag] = @rho * @s[param.tag] + (1 - @rho) * v ** 2
          param.data -= v
        end
      end
    end


    class RMSPropGraves < Optimizer
      attr_accessor :lr
      attr_accessor :alpha
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(hash[:lr], alpha: hash[:alpha], eps: hash[:eps], clip_norm: hash[:clip_norm])
      end

      # @param [Float] lr Learning rate.
      # @param [Float] alpha Moving average index of past slopes.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(lr = 0.0001, alpha: 0.95, eps: 0.0001, clip_norm: nil)
        super(clip_norm: clip_norm)
        @lr = lr
        @alpha = alpha
        @eps = eps
        @m = {}
        @v = {}
        @status = { m: @m, v: @v }
      end

      def to_hash
        super(lr: @lr, alpha: @alpha, eps: @eps)
      end

      private def update_params(params)
        params.each do |param|
          @m[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @v[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @m[param.tag] = @alpha * @m[param.tag] + (1 - @alpha) * param.grad
          @v[param.tag] = @alpha * @v[param.tag] + (1 - @alpha) * param.grad ** 2
          param.data -= (@lr / Xumo::NMath.sqrt(@v[param.tag] - @m[param.tag] ** 2 + @eps)) * param.grad
        end
      end
    end


    class Adam < Optimizer
      attr_accessor :alpha
      attr_accessor :beta1
      attr_accessor :beta2
      attr_accessor :eps
      attr_reader :amsgrad

      def self.from_hash(hash)
        self.new(alpha: hash[:alpha], beta1: hash[:beta1], beta2: hash[:beta2],
                 eps: hash[:eps], amsgrad: hash[:amsgrad], clip_norm: hash[:clip_norm])
      end

      # @param [Float] alpha Value used to calculate learning rate.
      # @param [Float] beta1 Moving average index of beta1.
      # @param [Float] beta2 Moving average index of beta2.
      # @param [Float] eps Value to avoid division by zero.
      # @param [Boolean] amsgrad Setting the true enable amsgrad.
      def initialize(alpha: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-7, amsgrad: false, clip_norm: nil)
        super(clip_norm: clip_norm)
        @alpha = alpha
        @beta1 = beta1
        @beta2 = beta2
        @eps = eps
        @amsgrad = amsgrad
        @t = 0
        @m = {}
        @v = {}
        @s = amsgrad ? {} : nil
        @status = { t: @t, m: @m, v: @v, s: @s }
      end

      def to_hash
        {
          class: self.class.name, alpha: @alpha, beta1: @beta1, beta2: @beta2,
          eps: @eps, amsgrad: @amsgrad, clip_norm: @clip_norm
        }
      end

      private def update_params(params)
        @t += 1
        lr = @alpha * Math.sqrt(1 - @beta2 ** @t) / (1 - @beta1 ** @t)
        params.each do |param|
          @m[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @v[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @m[param.tag] += (1 - @beta1) * (param.grad - @m[param.tag])
          @v[param.tag] += (1 - @beta2) * (param.grad ** 2 - @v[param.tag])
          if @amsgrad
            @s[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
            @s[param.tag] = Xumo::SFloat.maximum(@s[param.tag], @v[param.tag])
            param.data -= lr * @m[param.tag] / Xumo::NMath.sqrt(@s[param.tag] + @eps)
          else
            param.data -= lr * @m[param.tag] / Xumo::NMath.sqrt(@v[param.tag] + @eps)
          end
        end
      end
    end


    class AdaBound < Adam
      attr_accessor :final_lr
      attr_accessor :gamma

      def self.from_hash(hash)
        self.new(alpha: hash[:alpha], beta1: hash[:beta1], beta2: hash[:beta2],
                 final_lr: hash[:final_lr], gamma: hash[:gamma], eps: hash[:eps], amsgrad: hash[:amsgrad], clip_norm: hash[:clip_norm])
      end

      # @param [Float] final_lr Final learning rate.
      # @param [Float] gamma Lower and upper range value.
      def initialize(alpha: 0.001, beta1: 0.9, beta2: 0.999, final_lr: 0.1, gamma: 0.001, eps: 1e-7, amsgrad: false, clip_norm: nil)
        super(alpha: alpha, beta1: beta1, beta2: beta2, eps: eps, amsgrad: amsgrad, clip_norm: clip_norm)
        @final_lr = final_lr
        @gamma = gamma
      end

      def to_hash
        {
          class: self.class.name, alpha: @alpha, beta1: @beta1, beta2: @beta2,
          final_lr: @final_lr, gamma: @gamma, eps: @eps, amsgrad: amsgrad, clip_norm: @clip_norm
        }
      end

      private def update_params(params)
        @t += 1
        lr = @alpha * Math.sqrt(1 - @beta2 ** @t) / (1 - @beta1 ** @t)
        final_lr = @final_lr * lr / @alpha
        lower_bound = final_lr * (1 - 1 / (@gamma * @t + 1))
        upper_bound = final_lr * (1 + 1 / (@gamma * @t))
        params.each do |param|
          @m[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @v[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
          @m[param.tag] += (1 - @beta1) * (param.grad - @m[param.tag])
          @v[param.tag] += (1 - @beta2) * (param.grad ** 2 - @v[param.tag])
          if @amsgrad
            @s[param.tag] ||= Xumo::SFloat.zeros(*param.data.shape)
            @s[param.tag] = Xumo::SFloat.maximum(@s[param.tag], @v[param.tag])
            param.data -= clip_lr(lr / (Xumo::NMath.sqrt(@s[param.tag]) + @eps), lower_bound, upper_bound) * @m[param.tag]
          else
            param.data -= clip_lr(lr / (Xumo::NMath.sqrt(@v[param.tag]) + @eps), lower_bound, upper_bound) * @m[param.tag]
          end
        end
      end

      private def clip_lr(lr, lower_bound, upper_bound)
        lr[lr < lower_bound] = lower_bound
        lr[lr > upper_bound] = upper_bound
        lr
      end
    end

  end
end
