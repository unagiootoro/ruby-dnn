module DNN
  module Activations

    class Sigmoid < Layers::Layer
      def forward(x)
        @out = 1 / (1 + Xumo::NMath.exp(-x))
      end
    
      def backward(dout)
        dout * (1 - @out) * @out
      end
    end


    class Tanh < Layers::Layer
      def forward(x)
        @out = Xumo::NMath.tanh(x)
      end
      
      def backward(dout)
        dout * (1 - @out**2)
      end
    end
    
    
    class ReLU < Layers::Layer
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


    class LeakyReLU < Layers::Layer
      attr_reader :alpha

      def initialize(alpha = 0.3)
        @alpha = alpha
      end

      def self.load_hash(hash)
        self.new(hash[:alpha])
      end

      def forward(x)
        @x = x.clone
        a = Xumo::SFloat.ones(x.shape)
        a[x <= 0] = @alpha
        x * a
      end

      def backward(dout)
        @x[@x > 0] = 1
        @x[@x <= 0] = @alpha
        dout * @x
      end

      def to_hash
        {class: self.class.name, alpha: alpha}
      end
    end


    class Softsign < Layers::Layer
      def forward(x)
        @x = x
        x / (1 + x.abs)
      end

      def backward(dout)
        dout * (1 / (1 + @x.abs)**2)
      end
    end


    class Softplus < Layers::Layer
      def forward(x)
        @x = x
        Xumo::NMath.log(1 + Xumo::NMath.exp(x))
      end

      def backward(dout)
        dout * (1 / (1 + Xumo::NMath.exp(-@x)))
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


    class IdentityHuber < Layers::OutputLayer
      def forward(x)
        @out = x
      end

      def loss(y)
        loss = loss_l1(y)
        @loss = loss > 1 ? loss : loss_l2(y)
      end

      def backward(y)
        dout = @out - y
        if @loss > 1
          dout[dout >= 0] = 1
          dout[dout < 0] = -1
        end
        dout
      end

      private

      def loss_l1(y)
        batch_size = y.shape[0]
        (@out - y).abs.sum / batch_size
      end

      def loss_l2(y)
        batch_size = y.shape[0]
        0.5 * ((@out - y)**2).sum / batch_size
      end
    end
    
    
    class SoftmaxWithLoss < Layers::OutputLayer
      def forward(x)
        @out = Xumo::NMath.exp(x) / Xumo::NMath.exp(x).sum(1).reshape(x.shape[0], 1)
      end
    
      def backward(y)
        @out - y
      end
    
      def loss(y)
        batch_size = y.shape[0]
        -(y * Xumo::NMath.log(@out + 1e-7)).sum / batch_size + ridge
      end
    end


    class SigmoidWithLoss < Layers::OutputLayer
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
        -(y * Xumo::NMath.log(@out + 1e-7) + (1 - y) * Xumo::NMath.log(1 - @out + 1e-7)).sum / batch_size + ridge
      end
    end

  end
end
