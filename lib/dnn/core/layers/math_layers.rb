module DNN
  module Layers

    class Neg < Layer
      include LayerNode

      def forward_node(x)
        -x
      end

      def backward_node(dy)
        -dy
      end
    end

    class Add < MergeLayer
      include MergeLayerNode

      def forward_node(x1, x2)
        x1 + x2
      end

      def backward_node(dy)
        [dy, dy]
      end
    end

    class Sub < MergeLayer
      include MergeLayerNode

      def forward_node(x1, x2)
        x1 - x2
      end

      def backward_node(dy)
        [dy, -dy]
      end
    end

    class Mul < MergeLayer
      include MergeLayerNode

      def forward_node(x1, x2)
        @x1, @x2 = x1, x2
        x1 * x2
      end

      def backward_node(dy)
        [dy * @x2, dy * @x1]
      end
    end

    class Div < MergeLayer
      include MergeLayerNode

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
      include MergeLayerNode

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
        @y = Xumo::NMath.exp(x)
      end

      def backward_node(dy)
        dy * @y
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
        dy * @index * @x**(@index - 1)
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
        @x_shape = x.shape
        @dim = x.shape[@axis]
        x.sum(axis: @axis, keepdims: true)
      end

      def backward_node(dy)
        return dy if @x_shape == dy.shape
        dx = dy
        (@dim - 1).times do
          dx = dx.concatenate(dy, axis: @axis)
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
        @x_shape = x.shape
        @dim = x.shape[@axis]
        x.mean(axis: @axis, keepdims: true)
      end

      def backward_node(dy)
        return dy / @dim if @x_shape == dy.shape
        dx = dy
        (@dim - 1).times do
          dx = dx.concatenate(dy, axis: @axis)
        end
        dx / @dim
      end
    end

  end
end
