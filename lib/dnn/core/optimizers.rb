module DNN
  module Optimizers

    # Super class of all optimizer classes.
    class Optimizer
      attr_accessor :learning_rate

      def initialize(learning_rate)
        @learning_rate = learning_rate
      end

      # Update layer has params.
      def update(layer) end

      def to_hash(merge_hash = nil)
        hash = {name: self.class.name, learning_rate: @learning_rate}
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
    
      def update(layer)
        @v[layer] ||= {}
        layer.params.each_key do |key|
          amount = layer.grads[key] * @learning_rate
          if @momentum > 0
            @v[layer][key] ||= 0
            amount += @momentum * @v[layer][key]
            @v[layer][key] = amount
          end
          layer.params[key] -= amount
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
    
      def update(layer)
        @v[layer] ||= {}
        layer.params.each_key do |key|
          @v[layer][key] ||= 0
          amount = layer.grads[key] * @learning_rate
          @v[layer][key] = @v[layer][key] * @momentum - amount
          layer.params[key] = (layer.params[key] + @momentum**2 * @v[layer][key]) - (1 + @momentum) * amount
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
    
      def update(layer)
        @g[layer] ||= {}
        layer.params.each_key do |key|
          @g[layer][key] ||= 0
          @g[layer][key] += layer.grads[key]**2
          layer.params[key] -= (@learning_rate / NMath.sqrt(@g[layer][key] + 1e-7)) * layer.grads[key]
        end
      end
    end
    
    
    class RMSProp < Optimizer
      attr_accessor :muse

      def self.load_hash(hash)
        self.new(hash[:learning_rate], hash[:muse])
      end
    
      def initialize(learning_rate = 0.001, muse = 0.9)
        super(learning_rate)
        @muse = muse
        @g = {}
      end
    
      def update(layer)
        @g[layer] ||= {}
        layer.params.each_key do |key|
          @g[layer][key] ||= 0
          @g[layer][key] = @muse * @g[layer][key] + (1 - @muse) * layer.grads[key]**2
          layer.params[key] -= (@learning_rate / NMath.sqrt(@g[layer][key] + 1e-7)) * layer.grads[key]
        end
      end

      def to_hash
        super({muse: @muse})
      end
    end


    class Adam < Optimizer
      include Xumo
      
      attr_accessor :beta1
      attr_accessor :beta2

      def initialize(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999)
        super(learning_rate)
        @beta1 = beta1
        @beta2 = beta2
        @iter = 0
        @m = {}
        @v = {}
      end

      def self.load_hash(hash)
        self.new(hash[:learning_rate], hash[:beta1], hash[:beta2])
      end

      def update(layer)
        @iter += 1
        @m[layer] ||= {}
        @v[layer] ||= {}
        lr = @learning_rate * Math.sqrt(1 - @beta2**@iter) / (1 - @beta1**@iter) 
        layer.params.each_key do |key|
          @m[layer][key] ||= 0
          @v[layer][key] ||= 0
          @m[layer][key] += (1 - @beta1) * (layer.grads[key] - @m[layer][key])
          @v[layer][key] += (1 - @beta2) * (layer.grads[key]**2 - @v[layer][key])
          layer.params[key] -= lr * @m[layer][key] / NMath.sqrt(@v[layer][key] + 1e-7)
        end
      end

      def to_hash
        super({beta1: @beta1, beta2: @beta2})
      end
    end

  end
end
