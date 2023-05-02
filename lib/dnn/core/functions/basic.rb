module DNN
  module Functions
    class Add < Function
      def initialize
        @requires_dx1 = true
        @requires_dx2 = true
      end

      def call(x1, x2)
        @requires_dx1 = x1.requires_grad
        @requires_dx2 = x2.requires_grad
        super(x1, x2)
      end

      def forward(x1, x2)
        @x1 = x1
        @x2 = x2
        @x1_shape = x1.shape
        @x2_shape = x2.shape
        x1 + x2
      end

      def backward(dy)
        dx1 = @requires_dx1 ? MathUtils.sum_to(dy, @x1_shape) : nil
        dx2 = @requires_dx2 ? MathUtils.sum_to(dy, @x2_shape) : nil
        [dx1, dx2]
      end
    end

    class Sub < Function
      def initialize
        @requires_dx1 = true
        @requires_dx2 = true
      end

      def call(x1, x2)
        @requires_dx1 = x1.requires_grad
        @requires_dx2 = x2.requires_grad
        super(x1, x2)
      end

      def forward(x1, x2)
        @x1_shape = x1.shape
        @x2_shape = x2.shape
        x1 - x2
      end

      def backward(dy)
        dx1 = @requires_dx1 ? MathUtils.sum_to(dy, @x1_shape) : nil
        dx2 = @requires_dx2 ? MathUtils.sum_to(-dy, @x2_shape) : nil
        [dx1, dx2]
      end
    end

    class Mul < Function
      def initialize
        @requires_dx1 = true
        @requires_dx2 = true
      end

      def call(x1, x2)
        @requires_dx1 = x1.requires_grad
        @requires_dx2 = x2.requires_grad
        super(x1, x2)
      end

      def forward(x1, x2)
        @x1, @x2 = x1, x2
        x1 * x2
      end

      def backward(dy)
        dx1 = @requires_dx1 ? MathUtils.sum_to(dy * @x2, @x1.shape) : nil
        dx2 = @requires_dx2 ? MathUtils.sum_to(dy * @x1, @x2.shape) : nil
        [dx1, dx2]
      end
    end

    class Div < Function
      def initialize
        @requires_dx1 = true
        @requires_dx2 = true
      end

      def call(x1, x2)
        @requires_dx1 = x1.requires_grad
        @requires_dx2 = x2.requires_grad
        super(x1, x2)
      end

      def forward(x1, x2)
        @x1, @x2 = x1, x2
        x1 / x2
      end

      def backward(dy)
        dx1 = @requires_dx1 ? MathUtils.sum_to(dy / @x2, @x1.shape) : nil
        dx2 = @requires_dx2 ? MathUtils.sum_to(dy * -(@x1 / @x2**2), @x2.shape) : nil
        [dx1, dx2]
      end
    end

    class Neg < Function
      def forward(x)
        -x
      end

      def backward(dy)
        -dy
      end
    end

    class Pow < Function
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

    class Dot < Function
      def initialize
        @requires_dx1 = true
        @requires_dx2 = true
      end

      def call(x1, x2)
        @requires_dx1 = x1.requires_grad
        @requires_dx2 = x2.requires_grad
        super(x1, x2)
      end

      def forward(x1, x2)
        @x1, @x2 = x1, x2
        x1.dot(x2)
      end

      def backward(dy)
        dx1 = @requires_dx1 ? dy.dot(@x2.transpose) : nil
        dx2 = @requires_dx2 ? @x1.transpose.dot(dy) : nil
        [dx1, dx2]
      end
    end

    class Flatten < Function
      def forward(x)
        @x_shape = x.shape
        x.flatten
      end

      def backward(dy)
        if DNN.use_cumo?
          _backward_gpu(dy)
        else
          _backward_cpu(dy)
        end
      end

      private def _backward_cpu(dy)
        dy.reshape(*@x_shape)
      end

      private def _backward_gpu(dy)
        dy.flatten.reshape(*@x_shape)
      end
    end

    class Reshape < Function
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

      private def _forward_cpu(x)
        x.reshape(*@shape)
      end

      private def _backward_cpu(dy)
        dy.reshape(*@x_shape)
      end

      private def _forward_gpu(x)
        x.flatten.reshape(*@shape)
      end

      private def _backward_gpu(dy)
        dy.flatten.reshape(*@x_shape)
      end
    end

    class Transpose < Function
      def initialize(*axes)
        @axes = axes
      end

      def forward(x)
        x.transpose(*@axes)
      end

      def backward(dy)
        axes = @axes.length == 0 ? (dy.ndim - 1).downto(0).to_a : @axes
        d_axes = (0...dy.ndim).map { |i| axes.index(i) }
        dy.transpose(*d_axes)
      end
    end

    class Concatenate < Function
      def initialize(axis: 0)
        super()
        @axis = axis
      end

      def forward(*xs)
        @xs_shapes = xs.map { |x| x.shape }
        Xumo::NArray.concatenate(xs, axis: @axis)
      end

      def backward(dy)
        sum = 0
        indices = @xs_shapes.map do |x_shape|
          sum += x_shape[@axis]
          sum
        end
        dy.split(indices, axis: @axis)
      end
    end

    class Split < Function
      def initialize(indices_or_sections, axis: 0)
        super()
        @indices_or_sections = indices_or_sections
        @axis = axis
      end

      def forward(x)
        @x_shape = x.shape
        x.split(@indices_or_sections, axis: @axis)
      end

      def backward(*ys)
        Xumo::NArray.concatenate(ys, axis: @axis)
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

      def concatenate(*xs, axis: 0)
        Concatenate.new(axis: axis).(*xs)
      end

      def split(x, indices_or_sections, axis: 0)
        Split.new(indices_or_sections, axis: axis).(x)
      end
    end
  end
end
