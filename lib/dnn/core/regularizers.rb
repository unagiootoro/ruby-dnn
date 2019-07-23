module DNN
  module Regularizers

    class Regularizer
      attr_accessor :param

      def forward(x)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward'")
      end

      def backward
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'backward'")
      end

      def to_hash(merge_hash)
        hash = {class: self.class.name}
        hash.merge!(merge_hash)
        hash
      end
    end

    class L1 < Regularizer
      attr_accessor :l1_lambda

      def self.from_hash(hash)
        L1.new(hash[:l1_lambda])
      end

      def initialize(l1_lambda = 0.01)
        @l1_lambda = l1_lambda
      end

      def forward(x)
        x + @l1_lambda * @param.data.abs.sum
      end

      def backward
        dparam = Xumo::SFloat.ones(*@param.data.shape)
        dparam[@param.data < 0] = -1
        @param.grad += @l1_lambda * dparam
      end

      def to_hash
        super(l1_lambda: @l1_lambda)
      end
    end


    class L2 < Regularizer
      attr_accessor :l2_lambda

      def self.from_hash(hash)
        L2.new(hash[:l2_lambda])
      end

      def initialize(l2_lambda = 0.01)
        @l2_lambda = l2_lambda
      end

      def forward(x)
        x + 0.5 * @l2_lambda * (@param.data ** 2).sum
      end

      def backward
        @param.grad += @l2_lambda * @param.data
      end

      def to_hash
        super(l2_lambda: @l2_lambda)
      end
    end

    class L1L2 < Regularizer
      attr_accessor :l1_lambda
      attr_accessor :l2_lambda

      def self.from_hash(hash)
        L1L2.new(hash[:l1_lambda], hash[:l2_lambda])
      end

      def initialize(l1_lambda = 0.01, l2_lambda = 0.01)
        @l1_lambda = l1_lambda
        @l2_lambda = l2_lambda
      end

      def forward(x)
        l1 = @l1_lambda * @param.data.abs.sum
        l2 = 0.5 * @l2_lambda * (@param.data ** 2).sum
        x + l1 + l2
      end

      def backward
        dparam = Xumo::SFloat.ones(*@param.data.shape)
        dparam[@param.data < 0] = -1
        @param.grad += @l1_lambda * dparam
        @param.grad += @l2_lambda * @param.data
      end

      def to_hash
        super(l1_lambda: l1_lambda, l2_lambda: l2_lambda)
      end

    end

  end
end
