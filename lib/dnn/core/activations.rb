module DNN
  module Activations

    class Sigmoid < Layers::Layer
      def forward(x)
        @out = Utils.sigmoid(x)
      end
    
      def backward(dout)
        dout * (1 - @out) * @out
      end
    end


    class Tanh < Layers::Layer
      NMath = Xumo::NMath

      def forward(x)
        @out = NMath.tanh(x)
      end
      
      def backward(dout)
        dout * (1 - @out**2)
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
      NMath = Xumo::NMath

      def forward(x)
        @x = x
        NMath.log(1 + NMath.exp(x))
      end

      def backward(dout)
        dout * (1 / (1 + NMath.exp(-@x)))
      end
    end


    class Swish < Layers::Layer
      NMath = Xumo::NMath

      def forward(x)
        @x = x
        @out = x * (1 / (1 + NMath.exp(-x)))
      end
    
      def backward(dout)
        dout * (@out + (1 / (1 + NMath.exp(-@x))) * (1 - @out))
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


    class ELU < Layers::Layer
      NMath = Xumo::NMath

      attr_reader :alpha

      def self.load_hash(hash)
        self.new(hash[:alpha])
      end

      def initialize(alpha = 1.0)
        @alpha = alpha
      end

      def forward(x)
        @x = x
        x1 = Xumo::SFloat.zeros(x.shape)
        x1[x >= 0] = 1
        x1 *= x
        x2 = Xumo::SFloat.zeros(x.shape)
        x2[x < 0] = 1
        x2 *= @alpha * NMath.exp(x) - @alpha
        x1 + x2
      end

      def backward(dout)
        dx = Xumo::SFloat.ones(@x.shape)
        dx[@x < 0] = 0
        dx2 = Xumo::SFloat.zeros(@x.shape)
        dx2[@x < 0] = 1
        dx2 *= @alpha * NMath.exp(@x)
        dout * (dx + dx2)
      end

      def to_hash
        {class: self.class.name, alpha: @alpha}
      end
    end

  end
end
