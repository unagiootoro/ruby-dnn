module DNN
  module Layers

    class Split < Layer
      include LayerNode

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
        y1, y2others = x.split([x1_dim, x1_dim + x2_dim], axis: @axis)
        y2 = y2others.is_a?(Array) ? y2others[0].concatenate(y2others[1..-1], axis: @axis) : y2others
        [y1, y2]
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
