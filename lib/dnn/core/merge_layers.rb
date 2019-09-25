module DNN
  module MergeLayers

    class MergeLayer < Layers::Layer
      def self.call(x1, x2, *args)
        self.new(*args).call(x1, x2)
      end

      def call(input1, input2)
        x1, prev_link1 = *input1
        x2, prev_link2 = *input2
        build(x1.shape[1..-1]) unless built?
        y = forward(x1, x2)
        link = TwoInputLink.new(prev_link1, prev_link2, self)
        [y, link]
      end
    end


    class Add < MergeLayer
      def forward(x1, x2)
        x1 + x2
      end

      def backward(dy)
        [dy, dy]
      end
    end


    class Mul < MergeLayer
      def forward(x1, x2)
        @x1, @x2 = x1, x2
        x1 * x2
      end

      def backward(dy)
        [dy * @x2, dy * @x1]
      end
    end


    class Concatenate < MergeLayer
      attr_reader :axis

      def initialize(axis: 1)
        super()
        @axis = axis
      end

      def forward(x1, x2)
        @x1_dim = x1.shape[@axis]
        @x2_dim = x2.shape[@axis]
        x1.concatenate(x2, axis: @axis)
      end

      def backward(dy)
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
