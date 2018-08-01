module DNN
  module Activations
    Layer = Layers::Layer

    class Sigmoid < Layer
      def forward(x)
        @out = 1 / (1 + NMath.exp(-x))
      end
    
      def backward(dout)
        dout * (1 - @out) * @out
      end
    end


    class Tanh < Layer
      include Xumo

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
      include Xumo

      attr_reader :alpha

      def initialize(alpha = 0.3)
        @alpha = alpha
      end

      def self.load_hash(hash)
        self.new(hash[:alpha])
      end

      def forward(x)
        @x = x.clone
        a = SFloat.ones(x.shape)
        a[x <= 0] = @alpha
        x * a
      end

      def backward(dout)
        @x[@x > 0] = 1
        @x[@x <= 0] = @alpha
        dout * @x
      end

      def to_hash
        {name: self.class.name, alpha: alpha}
      end
    end


    class IdentityMSE < Layers::OutputLayer
      def forward(x)
        @out = x
      end
    
      def backward(y)
        @out - y
      end
    
      def loss(y)
        batch_size = y.shape[0]
        0.5 * ((@out - y)**2).sum / batch_size + ridge
      end
    end


    class IdentityMAE < Layers::OutputLayer
      def forward(x)
        @out = x
      end
    
      def backward(y)
        dout = @out - y
        dout[dout >= 0] = 1
        dout[dout < 0] = -1
        dout
      end
    
      def loss(y)
        batch_size = y.shape[0]
        (@out - y).abs.sum / batch_size + ridge
      end
    end
    
    
    class SoftmaxWithLoss < Layers::OutputLayer
      def forward(x)
        @out = NMath.exp(x) / NMath.exp(x).sum(1).reshape(x.shape[0], 1)
      end
    
      def backward(y)
        @out - y
      end
    
      def loss(y)
        batch_size = y.shape[0]
        -(y * NMath.log(@out + 1e-7)).sum / batch_size + ridge
      end
    end


    class SigmoidWithLoss < Layers::OutputLayer
      include Xumo

      def initialize
        @sigmoid = Sigmoid.new
      end

      def forward(x)
        @out = @sigmoid.forward(x)
      end

      def backward(y)
        @out - y
      end

      def loss(y)
        batch_size = y.shape[0]
        -(y * NMath.log(@out + 1e-7) + (1 - y) * NMath.log(1 - @out + 1e-7)).sum / batch_size + ridge
      end
    end

  end
end
