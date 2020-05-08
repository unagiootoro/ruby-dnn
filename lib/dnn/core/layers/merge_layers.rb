module DNN
  module Layers

    class MergeLayer < Layer
      def self.call(x1, x2, *args)
        new(*args).call(x1, x2)
      end

      def call(input1, input2)
        input1 = Tensor.convert(input1) if !input1.is_a?(Tensor) && !input1.is_a?(Param)
        input2 = Tensor.convert(input2) if !input2.is_a?(Tensor) && !input2.is_a?(Param)
        if input1.data.is_a?(Xumo::NArray)
          build(input1.data.shape[1..-1]) unless built?
        else
          build([1]) unless built?
        end
        forward(input1, input2)
      end
    end

    class Concatenate < MergeLayer
      include LayerNode

      attr_reader :axis

      def initialize(axis: 1)
        super()
        @axis = axis
      end

      def forward_node(x1, x2)
        @x1_dim = x1.shape[@axis]
        @x2_dim = x2.shape[@axis]
        x1.concatenate(x2, axis: @axis)
      end

      def backward_node(dy)
        dy.split([@x1_dim, @x1_dim + @x2_dim], axis: @axis)
      end

      def to_hash
        super(axis: @axis)
      end

      def load_hash(hash)
        initialize(axis: hash[:axis])
      end
    end

  end
end
