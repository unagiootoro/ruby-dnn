module DNN
  module Layers

    class Sigmoid < Layer
      include LayerNode

      def forward_node(x)
        @y = 1 / (1 + Xumo::NMath.exp(-x))
      end

      def backward_node(dy)
        dy * (1 - @y) * @y
      end
    end

    class Tanh < Layer
      include LayerNode

      def forward_node(x)
        @y = Xumo::NMath.tanh(x)
      end

      def backward_node(dy)
        dy * (1 - @y**2)
      end
    end

    class Softsign < Layer
      include LayerNode

      def forward_node(x)
        @x = x
        x / (1 + x.abs)
      end

      def backward_node(dy)
        dy * (1 / (1 + @x.abs)**2)
      end
    end

    class Softplus < Layer
      include LayerNode

      def forward_node(x)
        @x = x
        Xumo::NMath.log(1 + Xumo::NMath.exp(x))
      end

      def backward_node(dy)
        dy * (1 / (1 + Xumo::NMath.exp(-@x)))
      end
    end

    class Swish < Layer
      include LayerNode

      def forward_node(x)
        @x = x
        @y = x * (1 / (1 + Xumo::NMath.exp(-x)))
      end

      def backward_node(dy)
        dy * (@y + (1 / (1 + Xumo::NMath.exp(-@x))) * (1 - @y))
      end
    end

    class ReLU < Layer
      include LayerNode

      def forward_node(x)
        @x = x
        Xumo::SFloat.maximum(0, x)
      end

      def backward_node(dy)
        dy * Xumo::SFloat.cast(@x > 0)
      end
    end

    class LeakyReLU < Layer
      include LayerNode

      attr_reader :alpha

      # @param [Float] alpha The slope when the output value is negative.
      def initialize(alpha = 0.3)
        super()
        @alpha = alpha
      end

      def forward_node(x)
        @x = x
        a = Xumo::SFloat.ones(x.shape)
        a[x <= 0] = @alpha
        x * a
      end

      def backward_node(dy)
        dx = Xumo::SFloat.ones(@x.shape)
        dx[@x <= 0] = @alpha
        dy * dx
      end

      def to_hash
        super(alpha: @alpha)
      end

      def load_hash(hash)
        initialize(hash[:alpha])
      end
    end

    class ELU < Layer
      include LayerNode

      attr_reader :alpha

      # @param [Float] alpha The slope when the output value is negative.
      def initialize(alpha = 1.0)
        super()
        @alpha = alpha
      end

      def forward_node(x)
        @x = x
        x1 = Xumo::SFloat.zeros(x.shape)
        x1[x >= 0] = 1
        x1 *= x
        x2 = Xumo::SFloat.zeros(x.shape)
        x2[x < 0] = 1
        x2 *= @alpha * Xumo::NMath.exp(x) - @alpha
        x1 + x2
      end

      def backward_node(dy)
        dx = Xumo::SFloat.ones(@x.shape)
        dx[@x < 0] = 0
        dx2 = Xumo::SFloat.zeros(@x.shape)
        dx2[@x < 0] = 1
        dx2 *= @alpha * Xumo::NMath.exp(@x)
        dy * (dx + dx2)
      end

      def to_hash
        super(alpha: @alpha)
      end

      def load_hash(hash)
        initialize(hash[:alpha])
      end
    end

    class Mish < Layer
      include LayerNode

      def forward_node(x)
        @x = x
        x * Xumo::NMath.tanh(Softplus.new.forward_node(x))
      end

      def backward_node(dy)
        omega = 4 * (@x + 1) + 4 * Xumo::NMath.exp(2 * @x) + Xumo::NMath.exp(3 * @x) + Xumo::NMath.exp(@x) * (4 * @x + 6)
        delta = 2 * Xumo::NMath.exp(@x) + Xumo::NMath.exp(2 * @x) + 2
        dy * (Xumo::NMath.exp(@x) * omega) / delta**2
      end
    end

  end
end
