module DNN
  module Layers

    module MergeLayerNode
      def forward(input1, input2)
        x1 = input1.data
        x2 = input2.data
        prev1 = (input1.is_a?(Tensor) ? input1.link : input1)
        prev2 = (input2.is_a?(Tensor) ? input2.link : input2)
        y = forward_node(x1, x2)
        link = TwoInputLink.new(prev1, prev2, self)
        Tensor.convert(y, link)
      end

      def backward(dy)
        backward_node(dy)
      end

      def forward_node(x1, x2)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward_node'"
      end

      def backward_node(dy)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'backward_node'"
      end
    end

    class MergeLayer < Layer
      def self.call(x1, x2, *args)
        new(*args).call(x1, x2)
      end

      def call(input1, input2)
        input1 = Tensor.convert(input1) if !input1.is_a?(Tensor) && !input1.is_a?(Param)
        input2 = Tensor.convert(input2) if !input2.is_a?(Tensor) && !input2.is_a?(Param)
        if input1.data.is_a?(Numo::NArray)
          build(input1.data.shape[1..-1]) unless built?
        else
          build([1]) unless built?
        end
        forward(input1, input2)
      end
    end

    class Concatenate < MergeLayer
      include MergeLayerNode

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
