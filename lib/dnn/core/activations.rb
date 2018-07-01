module DNN
  module Activations
    Layer = Layers::Layer
    OutputLayer = Layers::OutputLayer


    module SigmoidFunction
      def forward(x)
        @out = 1.0 / (1 + NMath.exp(-x))
      end
    end


    class Sigmoid < Layer
      include SigmoidFunction
    
      def backward(dout)
        dout * (1.0 - @out) * @out
      end
    end


    class Tanh < Layer
      include Numo

      def forward(x)
        @x = x
        NMath.tanh(x)
      end
      
      def backward(dout)
        dout * (1.0 / NMath.cosh(@x)**2)
      end
    end
    
    
    class ReLU < Layer
      def forward(x)
        @x = x.clone
        x[x < 0] = 0
        x
      end
    
      def backward(dout)
        @x[@x > 0] = 1
        @x[@x <= 0] = 0
        dout * @x
      end
    end


    class LeakyReLU < Layer
      def initialize(alpha = 0.3)
        @alpha = alpha
      end

      def forward(x)
        @x = x.clone
        a = Numo::SFloat.ones(x.shape)
        a[x <= 0] = @alpha
        x * a
      end

      def backward(dout)
        @x[@x > 0] = 1
        @x[@x <= 0] = @alpha
        dout * @x
      end
    end


    class IdentityWithLoss < OutputLayer
      def forward(x)
        @out = x
      end
    
      def backward(y)
        @out - y
      end
    
      def loss(y)
        0.5 * ((@out - y) ** 2).sum / @model.batch_size + ridge
      end
    end
    
    
    class SoftmaxWithLoss < OutputLayer
      def forward(x)
        @out = NMath.exp(x) / NMath.exp(x).sum(1).reshape(x.shape[0], 1)
      end
    
      def backward(y)
        @out - y
      end
    
      def loss(y)
        -(y * NMath.log(@out + 1e-7)).sum / @model.batch_size + ridge
      end
    end


    class SigmoidWithLoss < OutputLayer
      include Numo
      include SigmoidFunction

      def backward(y)
        @out - y
      end

      def loss(y)
        -(y * NMath.log(@out + 1e-7) + (1 - y) * NMath.log(1 - @out + 1e-7)).sum / @model.batch_size + ridge
      end
    end

  end
end
