module DNN
  module Layers

    module MergeLayerNode
      def forward(input_tensor1, input_tensor2)
        x1 = input_tensor1.data
        x2 = input_tensor2.data
        prev_link1 = (input_tensor1.is_a?(Tensor) ? input_tensor1.link : input_tensor1)
        prev_link2 = (input_tensor2.is_a?(Tensor) ? input_tensor2.link : input_tensor2)
        y = forward_node(x1, x2)
        link = TwoInputLink.new(prev_link1, prev_link2, self)
        Tensor.new(y, link)
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

    class MergeLayer < Layers::Layer
      def self.call(x1, x2, *args)
        new(*args).call(x1, x2)
      end

      def call(input_tensor1, input_tensor2)
        input_tensor1 = Tensor.new(input_tensor1) if !input_tensor1.is_a?(Tensor) && !input_tensor1.is_a?(Param)
        input_tensor2 = Tensor.new(input_tensor2) if !input_tensor2.is_a?(Tensor) && !input_tensor2.is_a?(Param)
        if input_tensor1.data.is_a?(Numo::NArray)
          build(input_tensor1.data.shape[1..-1]) unless built?
        else
          build([1]) unless built?
        end
        forward(input_tensor1, input_tensor2)
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
