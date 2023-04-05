module DNN
  module Functions

    class SoftmaxCrossEntropy < FunctionNode
      class << self
        def softmax(y)
          Xumo::NMath.exp(y) / Xumo::NMath.exp(y).sum(1, keepdims: true)
        end
      end

      def initialize(eps: 1e-7)
        @eps = eps
      end

      def forward(y, t)
        @t = t
        @x = SoftmaxCrossEntropy.softmax(y)
        -(t * Xumo::NMath.log(@x + @eps)).mean(0).sum
      end

      def backward(d)
        d * (@x - @t) / @x.shape[0]
      end
    end

    class SigmoidCrossEntropy < FunctionNode
      class << self
        def sigmoid(y)
          Functions::Sigmoid.new.forward(y)
        end
      end

      def initialize(eps: 1e-7)
        @eps = eps
      end

      def forward(y, t)
        @t = t
        @x = SigmoidCrossEntropy.sigmoid(y)
        -(t * Xumo::NMath.log(@x + @eps) + (1 - t) * Xumo::NMath.log(1 - @x + @eps)).mean(0).sum
      end

      def backward(d)
        d * (@x - @t) / @x.shape[0]
      end
    end

    module FunctionSpace
      module_function

      def softmax_cross_entropy(y, t, eps: 1e-7)
        SoftmaxCrossEntropy.new(eps: eps).(y, t)
      end

      def sigmoid_cross_entropy(y, t, eps: 1e-7)
        SigmoidCrossEntropy.new(eps: eps).(y, t)
      end
    end

  end
end
