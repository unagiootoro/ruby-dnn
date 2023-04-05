module DNN
  module Functions
    class Add < FunctionNode
      def forward(*xs)
        (x1, x2) = xs
        @x1_shape = x1.shape
        @x2_shape = x2.shape
        x1 + x2
      end

      def backward(*dys)
        dy = dys[0]
        dx1 = MathUtils.sum_to(dy, @x1_shape)
        dx2 = MathUtils.sum_to(dy, @x2_shape)
        [dx1, dx2]
      end
    end

    class Sub < FunctionNode
      def forward(x1, x2)
        @x1_shape = x1.shape
        @x2_shape = x2.shape
        x1 - x2
      end

      def backward(dy)
        dx1 = MathUtils.sum_to(dy, @x1_shape)
        dx2 = MathUtils.sum_to(-dy, @x2_shape)
        [dx1, dx2]
      end
    end

    class Mul < FunctionNode
      def forward(x1, x2)
        @x1, @x2 = x1, x2
        x1 * x2
      end

      def backward(dy)
        dx1 = MathUtils.sum_to(dy * @x2, @x1.shape)
        dx2 = MathUtils.sum_to(dy * @x1, @x2.shape)
        [dx1, dx2]
      end
    end

    class Div < FunctionNode
      def forward(x1, x2)
        @x1, @x2 = x1, x2
        x1 / x2
      end

      def backward(dy)
        dx1 = MathUtils.sum_to(dy / @x2, @x1.shape)
        dx2 = MathUtils.sum_to(dy * -(@x1 / @x2**2), @x2.shape)
        [dx1, dx2]
      end
    end

    class Neg < FunctionNode
      def forward(x)
        -x
      end

      def backward(dy)
        -dy
      end
    end

    class Pow < FunctionNode
      def initialize(index)
        @index = index
      end

      def forward(x)
        @x = x
        x**@index
      end

      def backward(dy)
        dy * @index * @x**(@index - 1)
      end
    end

    class Dot < FunctionNode
      def forward(x1, x2)
        @x1, @x2 = x1, x2
        x1.dot(x2)
      end

      def backward(dy)
        [dy.dot(@x2.transpose), @x1.transpose.dot(dy)]
      end
    end

    class Reshape < FunctionNode
      def initialize(shape)
        super()
        @shape = shape
      end

      def forward(x)
        @x_shape = x.shape
        if DNN.use_cumo?
          _forward_gpu(x)
        else
          _forward_cpu(x)
        end
      end

      def backward(dy)
        if DNN.use_cumo?
          _backward_gpu(dy)
        else
          _backward_cpu(dy)
        end
      end

      def _forward_cpu(x)
        x.reshape(x.shape[0], *@shape)
      end

      def _backward_cpu(dy)
        dy.reshape(*@x_shape)
      end

      def _forward_gpu(x)
        x.flatten.reshape(x.shape[0], *@shape)
      end

      def _backward_gpu(dy)
        dy.flatten.reshape(*@x_shape)
      end
    end

    module FunctionSpace
      module_function

      def reshape(x, shape)
        Reshape.new(shape).(x)
      end

      def flatten(x)
        Reshape.new(x.shape[1..-1].reduce(&:*)).(x)
      end

      def concatenate(x, axis: 0)
        Concatenate.new(axis: axis).(x)
      end

      def split(x, axis: 0, dim: nil)
        Split.new(axis: axis, dim: nil).(x)
      end
    end
  end
end
