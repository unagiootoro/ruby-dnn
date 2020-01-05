module DNN
  module Layers

    class Add < MergeLayer
      def forward_node(x1, x2)
        x1 + x2
      end

      def backward_node(dy)
        [dy, dy]
      end
    end

    class Sub < MergeLayer
      def forward_node(x1, x2)
        x1 - x2
      end

      def backward_node(dy)
        [dy, -dy]
      end
    end

    class Mul < MergeLayer
      def forward_node(x1, x2)
        @x1, @x2 = x1, x2
        x1 * x2
      end

      def backward_node(dy)
        [dy * @x2, dy * @x1]
      end
    end

    class Div < MergeLayer
      def forward_node(x1, x2)
        @x1, @x2 = x1, x2
        x1 / x2
      end

      def backward_node(dy)
        dx1 = dy / @x2
        dx2 = dy * -(@x1 / @x2**2)
        [dx1, dx2]
      end
    end

    class Dot < MergeLayer
      def forward_node(x1, x2)
        @x1, @x2 = x1, x2
        x1.dot(x2)
      end

      def backward_node(dy)
        [dy.dot(@x2.transpose), @x1.transpose.dot(dy)]
      end
    end

    class Exp < Layer
      include LayerNode

      def forward_node(x)
        @x = x
        Xumo::NMath.exp(x)
      end

      def backward_node(dy)
        dy * Xumo::NMath.exp(@x)
      end
    end

    class Log < Layer
      include LayerNode

      def forward_node(x)
        @x = x
        Xumo::NMath.log(x)
      end

      def backward_node(dy)
        dy / @x
      end
    end

    class Pow < Layer
      include LayerNode

      def initialize(index)
        super()
        @index = index
      end

      def forward_node(x)
        @x = x
        x**@index
      end

      def backward_node(dy)
        @index * @x**(@index - 1)
      end
    end

    class Sqrt < Layer
      include LayerNode

      def forward_node(x)
        @x = x
        Xumo::NMath.sqrt(x)
      end

      def backward_node(dy)
        dy * (1.0 / 2 * Xumo::NMath.sqrt(@x))
      end
    end

    class Sum < Layer
      include LayerNode

      def initialize(axis: 0)
        super()
        @axis = axis
      end

      def forward_node(x)
        if @axis
          @dim = x.shape[@axis]
          x.sum(axis: @axis, keepdims: true)
        else
          x.sum
        end
      end

      def backward_node(dy)
        dx = dy.clone
        if @axis
          (@dim - 1).times do
            dx.concatenate(dy, axis: @axis)
          end
        end
        dx
      end
    end

    class Mean < Layer
      include LayerNode

      def initialize(axis: 0)
        super()
        @axis = axis
      end

      def forward_node(x)
        @dim = @axis ? x.shape[@axis] : x.size
        x.mean(axis: @axis, keepdims: true)
      end

      def backward_node(dy)
        dx = dy
        if @axis
          (@dim - 1).times do
            dx.concatenate(dy, axis: @axis)
          end
        end
        dx / @dim
      end
    end

  end
end
