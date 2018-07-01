module DNN
  module Optimizers

    #Super class of all optimizer classes.
    class Optimizer
      attr_accessor :learning_rate

      def initialize(learning_rate)
        @learning_rate = learning_rate
      end

      #Update layer has params.
      def update(layer) end
    end


    class SGD < Optimizer
      attr_accessor :momentum

      def initialize(learning_rate = 0.01, momentum: 0)
        super(learning_rate)
        @momentum = momentum
        @amounts = {}
      end
    
      def update(layer)
        amount = if @amounts[layer]
          @amounts[layer]
        else
          @amounts[layer] = {}
        end
        layer.params.each_key do |key|
          amount[key] = layer.grads[key] * @learning_rate
          if @momentum > 0
            @amounts[layer][key] ||= 0
            amount[key] += @momentum * @amounts[layer][key]
            @amounts[layer] = amount
          end
          layer.params[key] -= amount[key]
        end
      end
    end
    
    
    class AdaGrad    
      def initialize(learning_rate = 0.01)
        super(learning_rate)
        @g = {}
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
    end


    class Adam < Optimizer
      include Numo
      
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
    end

  end
end
