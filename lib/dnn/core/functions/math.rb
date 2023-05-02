module DNN
  module Functions
    module MathUtils
      module_function

      def align_ndim(shape1, shape2)
        shape1 = shape1.clone
        shape2 = shape2.clone
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
        aligned_x_shape, aligned_target_shape = align_ndim(x.shape, target_shape)
        x = x.reshape(*aligned_x_shape)
        aligned_x_shape.length.times do |axis|
          unless x.shape[axis] == aligned_target_shape[axis]
            x = Numo::SFloat.concatenate([x] * aligned_target_shape[axis], axis: axis)
          end
        end
        x
      end

      def sum_to(x, target_shape)
        return x if x.shape == target_shape
        aligned_x_shape, aligned_target_shape = align_ndim(x.shape, target_shape)
        x = x.reshape(*aligned_x_shape)
        aligned_x_shape.length.times do |axis|
          unless x.shape[axis] == aligned_target_shape[axis]
            x = x.sum(axis: axis, keepdims: true)
          end
        end
        x.reshape(*target_shape)
      end
    end

    class Exp < Function
      def forward(x)
        @y = Xumo::NMath.exp(x)
      end

      def backward(dy)
        dy * @y
      end
    end

    class Log < Function
      def forward(x)
        @x = x
        Xumo::NMath.log(x)
      end

      def backward(dy)
        dy / @x
      end
    end

    class Sqrt < Function
      def forward(x)
        @x = x
        Xumo::NMath.sqrt(x)
      end

      def backward(dy)
        dy * (1.0 / 2 * Xumo::NMath.sqrt(@x))
      end
    end

    class Sum < Function
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
          x.sum(axis: @axis, keepdims: @keepdims)
        else
          x.sum
        end
      end

      def backward(dy)
        MathUtils.broadcast_to(dy, @x_shape)
      end
    end

    class Mean < Function
      attr_reader :axis
      attr_reader :keepdims

      def initialize(axis: nil, keepdims: true)
        super()
        @axis = axis
        @keepdims = keepdims
      end

      def forward(x)
        @x_shape = x.shape
        @dim = @axis ? x.shape[@axis] : x.size
        x.mean(axis: @axis, keepdims: @keepdims)
      end

      def backward(dy)
        MathUtils.broadcast_to(dy, @x_shape) / @dim
      end
    end

    class Abs < Function
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

    class Max < Function
      def initialize(axis: nil, keepdims: true)
        @axis = axis
        @keepdims = keepdims
      end

      def forward(x)
        @x_shape = x.shape
        @max_index = x.max_index(axis: @axis)
        x.max(axis: @axis, keepdims: @keepdims)
      end

      def backward(dy)
        dx = Xumo::SFloat.zeros(*@x_shape)
        if @max_index.is_a?(Integer)
          dx[@max_index] = dy.flatten
        else
          dx[@max_index.flatten] = dy.flatten
        end
        dx
      end
    end

    class BroadcastTo < Function
      def initialize(target_shape)
        @target_shape = target_shape
      end

      def forward(x)
        @x_shape = x.shape
        MathUtils.broadcast_to(x, @target_shape)
      end

      def backward(dy)
        MathUtils.sum_to(dy, @x_shape)
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

      def max(x, axis: nil, keepdims: true)
        Max.new(axis: axis, keepdims: keepdims).(x)
      end

      def broadcast_to(x, target_shape)
        BroadcastTo.new(target_shape).(x)
      end
    end

  end
end
