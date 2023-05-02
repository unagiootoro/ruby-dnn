module DNN
  module Functions

    class Sigmoid < Function
      def forward(x)
        @y = 1 / (1 + Xumo::NMath.exp(-x))
      end

      def backward(dy)
        dy * (1 - @y) * @y
      end
    end

    class Tanh < Function
      def forward(x)
        @y = Xumo::NMath.tanh(x)
      end

      def backward(dy)
        dy * (1 - @y**2)
      end
    end

    class Softsign < Function
      def forward(x)
        @x = x
        x / (1 + x.abs)
      end

      def backward(dy)
        dy * (1 / (1 + @x.abs)**2)
      end
    end

    class Softplus < Function
      def forward(x)
        @x = x
        Xumo::NMath.log(1 + Xumo::NMath.exp(x))
      end

      def backward(dy)
        dy * (1 / (1 + Xumo::NMath.exp(-@x)))
      end
    end

    class Swish < Function
      def forward(x)
        @x = x
        @y = x * (1 / (1 + Xumo::NMath.exp(-x)))
      end

      def backward(dy)
        dy * (@y + (1 / (1 + Xumo::NMath.exp(-@x))) * (1 - @y))
      end
    end

    class ReLU < Function
      def forward(x)
        @x = x
        Xumo::SFloat.maximum(0, x)
      end

      def backward(dy)
        dy * Xumo::SFloat.cast(@x > 0)
      end
    end

    class LeakyReLU < Function
      attr_reader :alpha

      # @param [Float] alpha The slope when the output value is negative.
      def initialize(alpha = 0.3)
        super()
        @alpha = alpha
      end

      def forward(x)
        @x = x
        a = Xumo::SFloat.ones(x.shape)
        a[x <= 0] = @alpha
        x * a
      end

      def backward(dy)
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

    class ELU < Function
      attr_reader :alpha

      # @param [Float] alpha The slope when the output value is negative.
      def initialize(alpha = 1.0)
        super()
        @alpha = alpha
      end

      def forward(x)
        @x = x
        x1 = Xumo::SFloat.zeros(x.shape)
        x1[x >= 0] = 1
        x1 *= x
        x2 = Xumo::SFloat.zeros(x.shape)
        x2[x < 0] = 1
        x2 *= @alpha * Xumo::NMath.exp(x) - @alpha
        x1 + x2
      end

      def backward(dy)
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

    class Mish < Function
      def forward(x)
        @x = x
        x * Xumo::NMath.tanh(Softplus.new.forward(x))
      end

      def backward(dy)
        omega = 4 * (@x + 1) + 4 * Xumo::NMath.exp(2 * @x) + Xumo::NMath.exp(3 * @x) + Xumo::NMath.exp(@x) * (4 * @x + 6)
        delta = 2 * Xumo::NMath.exp(@x) + Xumo::NMath.exp(2 * @x) + 2
        dy * (Xumo::NMath.exp(@x) * omega) / delta**2
      end
    end

    class Softmax < Function
      def initialize(axis: 1)
        @axis = axis
      end

      def forward(x)
        @y = Xumo::NMath.exp(x) / Xumo::NMath.exp(x).sum(@axis, keepdims: true)
      end

      def backward(dy)
        dx = @y * dy
        sum_dx = dx.sum(@axis, keepdims: true)
        dx -= @y * sum_dx
        dx
      end
    end

    module FunctionSpace
      module_function

      def sigmoid(x)
        Sigmoid.new.(x)
      end

      def tanh(x)
        Tanh.new.(x)
      end

      def softsign(x)
        Softsign.new.(x)
      end

      def softplus(x)
        Softplus.new.(x)
      end

      def swish(x)
        Swish.new.(x)
      end

      def relu(x)
        ReLU.new.(x)
      end

      def leaky_relu(x, alpha: 0.3)
        LeakyReLU.new(alpha).(x)
      end

      def elu(x, alpha: 1.0)
        ELU.new(alpha).(x)
      end

      def mish(x)
        Mish.new.(x)
      end

      def softmax(x, axis: 1)
        Softmax.new(axis: axis).(x)
      end
    end

  end
end
