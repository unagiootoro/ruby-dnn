module DNN
  module Optimizers

    # Super class of all optimizer classes.
    class Optimizer
      attr_accessor :learning_rate

      def initialize(learning_rate)
        @learning_rate = learning_rate
      end

      # Update params.
      # Classes that inherit from this class must implement this method.
      def update(params)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'update'")
      end

      def to_hash(merge_hash = nil)
        hash = {class: self.class.name, learning_rate: @learning_rate}
        hash.merge!(merge_hash) if merge_hash
        hash
      end
    end


    class SGD < Optimizer
      attr_accessor :momentum

      def self.load_hash(hash)
        self.new(hash[:learning_rate], momentum: hash[:momentum])
      end

      def initialize(learning_rate = 0.01, momentum: 0)
        super(learning_rate)
        @momentum = momentum
        @v = {}
      end
    
      def update(params)
        params.select { |key, param| param.is_a?(LearningParam) }.each_value do |param|
          amount = param.grad * @learning_rate
          if @momentum > 0
            @v[param] ||= 0
            amount += @momentum * @v[param]
            @v[param] = amount
          end
          param.data -= amount
        end
      end

      def to_hash
        super({momentum: @momentum})
      end
    end


    class Nesterov < SGD
      def self.load_hash(hash)
        self.new(hash[:learning_rate], momentum: hash[:momentum])
      end

      def initialize(learning_rate = 0.01, momentum: 0.9)
        super(learning_rate, momentum: momentum)
      end
    
      def update(params)
        params.select { |key, param| param.is_a?(LearningParam) }.each_value do |param|
          @v[param] ||= 0
          amount = param.grad * @learning_rate
          @v[param] = @v[param] * @momentum - amount
          param.data = (param.data + @momentum**2 * @v[param]) - (1 + @momentum) * amount
        end
      end
    end
    
    
    class AdaGrad < Optimizer
      def initialize(learning_rate = 0.01)
        super(learning_rate)
        @g = {}
      end

      def self.load_hash(hash)
        self.new(hash[:learning_rate])
      end
    
      def update(params)
        params.select { |key, param| param.is_a?(LearningParam) }.each_value do |param|
          @g[param] ||= 0
          @g[param] += param.grad**2
          param.data -= (@learning_rate / Xumo::NMath.sqrt(@g[param] + 1e-7)) * param.grad
        end
      end
    end
    
    
    class RMSProp < Optimizer
      attr_accessor :alpha

      def self.load_hash(hash)
        self.new(hash[:learning_rate], alpha: hash[:alpha])
      end
    
      def initialize(learning_rate = 0.001, alpha: 0.9)
        super(learning_rate)
        @alpha = alpha
        @g = {}
      end
    
      def update(params)
        params.select { |key, param| param.is_a?(LearningParam) }.each_value do |param|
          @g[param] ||= 0
          @g[param] = @alpha * @g[param] + (1 - @alpha) * param.grad**2
          param.data -= (@learning_rate / Xumo::NMath.sqrt(@g[param] + 1e-7)) * param.grad
        end
      end

      def to_hash
        super({alpha: @alpha})
      end
    end


    class AdaDelta < Optimizer
      attr_accessor :rho

      def self.load_hash(hash)
        self.new(rho: hash[:rho])
      end

      def initialize(rho: 0.95)
        super(nil)
        @rho = rho
        @h = {}
        @s = {}
      end

      def update(params)
        params.select { |key, param| param.is_a?(LearningParam) }.each_value do |param|
          @h[param] ||= Xumo::SFloat.zeros(*param.data.shape)
          @s[param] ||= Xumo::SFloat.zeros(*param.data.shape)
          @h[param] = @rho * @h[param] + (1 - @rho) * param.grad**2
          v = (Xumo::NMath.sqrt(@s[param] + 1e-6) / Xumo::NMath.sqrt(@h[param] + 1e-6)) * param.grad
          @s[param] = @rho * @s[param] + (1 - @rho) * v**2
          param.data -= v
        end
      end

      def to_hash
        super({rho: @rho})
      end
    end


    class Adam < Optimizer
      attr_accessor :beta1
      attr_accessor :beta2
      
      def self.load_hash(hash)
        self.new(hash[:learning_rate], beta1: hash[:beta1], beta2: hash[:beta2])
      end

      def initialize(learning_rate = 0.001, beta1: 0.9, beta2: 0.999)
        super(learning_rate)
        @beta1 = beta1
        @beta2 = beta2
        @iter = 0
        @m = {}
        @v = {}
      end

      def update(params)
        @iter += 1
        lr = @learning_rate * Math.sqrt(1 - @beta2**@iter) / (1 - @beta1**@iter) 
        params.select { |key, param| param.is_a?(LearningParam) }.each_value do |param|
          @m[param] ||= 0
          @v[param] ||= 0
          @m[param] += (1 - @beta1) * (param.grad - @m[param])
          @v[param] += (1 - @beta2) * (param.grad**2 - @v[param])
          param.data -= lr * @m[param] / Xumo::NMath.sqrt(@v[param] + 1e-7)
        end
      end

      def to_hash
        super({beta1: @beta1, beta2: @beta2})
      end
    end

  end
end
