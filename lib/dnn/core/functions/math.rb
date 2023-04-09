module DNN
  module Functions
    module MathUtils
      module_function

      def align_ndim(shape1, shape2)
        if shape1.length < shape2.length
          shape2.length.times do |axis|
            unless shape1[axis] == shape2[axis]
              shape1.insert(axis, 1)
            end
          end
        elsif shape1.length > shape2.length
          shape1.length.times do |axis|
            unless shape1[axis] == shape2[axis]
              shape2.insert(axis, 1)
            end
          end
        end
        [shape1, shape2]
      end

      def broadcast_to(x, target_shape)
        return x if x.shape == target_shape
        return Xumo::SFloat.new(*target_shape).fill(x[0]) if x.size == 1
        x_shape, target_shape = align_ndim(x.shape, target_shape)
        x = x.reshape(*x_shape)
        x_shape.length.times do |axis|
          unless x.shape[axis] == target_shape[axis]
            tmp = x
            (target_shape[axis] - 1).times do
              x = x.concatenate(tmp, axis: axis)
            end
          end
        end
        x
      end

      def sum_to(x, target_shape)
        return x if x.shape == target_shape
        x_shape, target_shape = align_ndim(x.shape, target_shape)
        x = x.reshape(*x_shape)
        x_shape.length.times do |axis|
          unless x.shape[axis] == target_shape[axis]
            x = x.sum(axis: axis, keepdims: true)
          end
        end
        x
      end
    end

    class Exp < FunctionNode
      def forward(x)
        @y = Xumo::NMath.exp(x)
      end

      def backward(dy)
        dy * @y
      end
    end

    class Log < FunctionNode
      def forward(x)
        @x = x
        Xumo::NMath.log(x)
      end

      def backward(dy)
        dy / @x
      end
    end

    class Sqrt < FunctionNode
      def forward(x)
        @x = x
        Xumo::NMath.sqrt(x)
      end

      def backward(dy)
        dy * (1.0 / 2 * Xumo::NMath.sqrt(@x))
      end
    end

    class Sum < FunctionNode
      attr_reader :axis
      attr_reader :keepdims

      def initialize(axis: nil, keepdims: true)
        super()
        @axis = axis
        @keepdims = keepdims
      end

      def forward(x)
        @x_shape = x.shape
        if @axis
          x.sum(axis: @axis, keepdims: true)
        else
          x.sum
        end
      end

      def backward(dy)
        MathUtils.broadcast_to(dy, @x_shape)
      end
    end

    class Mean < FunctionNode
      attr_reader :axis
      attr_reader :keepdims

      def initialize(axis: nil, keepdims: true)
        super()
        @axis = axis
        @keepdims = keepdims
      end

      def forward(x)
        @x_shape = x.shape
        if @axis
          @dim = x.shape[@axis]
          x.mean(axis: @axis, keepdims: true)
        else
          @dim = x.size
          x.mean
        end
      end

      def backward(dy)
        MathUtils.broadcast_to(dy, @x_shape) / @dim
      end
    end

    class Abs < FunctionNode
      def forward(x)
        @x = x
        x.abs
      end

      def backward(dy)
        mask = Xumo::SFloat.ones(*@x.shape)
        mask[@x < 0] = -1
        dy * mask
      end
    end

    module FunctionSpace
      module_function

      def exp(x)
        Exp.new.(x)
      end

      def log(x)
        Log.new.(x)
      end

      def sqrt(x)
        Sqrt.new.(x)
      end

      def sum(x, axis: nil, keepdims: true)
        Sum.new(axis: axis, keepdims: keepdims).(x)
      end

      def mean(x, axis: nil, keepdims: true)
        Mean.new(axis: axis, keepdims: keepdims).(x)
      end

      def abs(x)
        Abs.new.(x)
      end
    end

  end
end
