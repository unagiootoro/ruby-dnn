module DNN
  module Functions
    class MeanSquaredError < Function
      def forward(y, t)
        @y = y
        @t = t
        0.5 * ((y - t)**2).mean(0).sum
      end

      def backward(d)
        d * (@y - @t) / @y.shape[0]
      end
    end

    class MeanAbsoluteError < Function
      def forward(y, t)
        @y = y
        @t = t
        (y - t).abs.mean(0).sum
      end

      def backward(d)
        dy = (@y - @t)
        dy[dy >= 0] = 1
        dy[dy < 0] = -1
        d * dy / @y.shape[0]
      end
    end

    class Hinge < Function
      def forward(y, t)
        @t = t
        @a = 1 - y * t
        Xumo::SFloat.maximum(0, @a).mean(0).sum
      end

      def backward(d)
        a = Xumo::SFloat.ones(*@a.shape)
        a[@a <= 0] = 0
        d * (a * -@t) / a.shape[0]
      end
    end

    class HuberLoss < Function
      def forward(y, t)
        @y = y
        @t = t
        loss_l1_value = (y - t).abs.mean(0).sum
        @loss_value = loss_l1_value > 1 ? loss_l1_value : 0.5 * ((y - t)**2).mean(0).sum
      end

      def backward(d)
        dy = (@y - @t)
        if @loss_value > 1
          dy[dy >= 0] = 1
          dy[dy < 0] = -1
        end
        d * dy / @y.shape[0]
      end
    end

    class SoftmaxCrossEntropy < Function
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

    class SigmoidCrossEntropy < Function
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

      def mean_squared_error(y, t)
        MeanSquaredError.new.(y, t)
      end

      def mean_absolute_error(y, t)
        MeanAbsoluteError.new.(y, t)
      end

      def hinge(y, t)
        Hinge.new.(y, t)
      end

      def huber_loss(y, t)
        HuberLoss.new.(y, t)
      end

      def softmax_cross_entropy(y, t, eps: 1e-7)
        SoftmaxCrossEntropy.new(eps: eps).(y, t)
      end

      def sigmoid_cross_entropy(y, t, eps: 1e-7)
        SigmoidCrossEntropy.new(eps: eps).(y, t)
      end
    end

  end
end
