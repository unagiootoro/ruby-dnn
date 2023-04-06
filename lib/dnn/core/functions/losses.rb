module DNN
  module Functions

    class SoftmaxCrossEntropy < FunctionNode
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def forward(y, t)
        @t = t
        @x = Functions::Softmax.new.forward(y)
        -(t * Xumo::NMath.log(@x + @eps)).mean(0).sum
      end

      def backward(d)
        d * (@x - @t) / @x.shape[0]
      end
    end

    class SigmoidCrossEntropy < FunctionNode
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def forward(y, t)
        @t = t
        @x = Functions::Sigmoid.new.forward(y)
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
