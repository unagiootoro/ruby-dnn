module DNN
  module Layers

    module SplitLayerNode
      def forward(input)
        x = input.data
        prev = (input.is_a?(Tensor) ? input.link : input)
        ys = forward_node(x)
        link = Link.new(prevs: [prev], layer_node: self, num_outputs: 2)
        prev.next = link if prev.is_a?(Link)
        ys.map { |y| Tensor.convert(y, link) }
      end

      def forward_node(x)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward_node'"
      end

      def backward_node(dy1, dy2)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'backward_node'"
      end
    end

    class Split < Layer
      include SplitLayerNode

      attr_reader :axis
      attr_reader :dim

      def initialize(axis: 1, dim: nil)
        super()
        raise DNNError, "dim is nil" if dim == nil
        @axis = axis
        @dim = dim
      end

      def forward_node(x)
        x1_dim = @dim
        x2_dim = x.shape[@axis] - @dim
        x.split([x1_dim, x1_dim + x2_dim], axis: @axis)
      end

      def backward_node(dy1, dy2)
        dy1.concatenate(dy2, axis: @axis)
      end

      def to_hash
        super(axis: @axis, dim: @dim)
      end

      def load_hash(hash)
        initialize(axis: hash[:axis], dim: hash[:dim])
      end
    end

  end
end
