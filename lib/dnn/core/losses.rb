module DNN
  module Losses

    class Loss
      def forward(x, y, layers)
        loss_value = loss(x, y)
        regularizers = layers.select { |layer| layer.is_a?(Connection) }
                             .map { |layer| layer.regularizers }.flatten
        
        regularizers.each do |regularizer|
          loss_value = regularizer.forward(loss_value)
        end
        loss_value
      end

      def backward(y)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'backward'")
      end

      def regularizes_backward(layers)
        layers.select { |layer| layer.is_a?(Connection) }.each do |layer|
          layer.regularizers.each do |regularizer|
            regularizer.backward
          end
        end
      end

      def to_hash
        {class: self.class.name}
      end

      private

      def loss(x, y)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'loss'")
      end
    end

    class MeanSquaredError < Loss
      def loss(x, y)
        @x = x
        batch_size = y.shape[0]
        0.5 * ((x - y)**2).sum / batch_size
      end

      def backward(y)
        @x - y
      end
    end


    class MeanAbsoluteError < Loss
      def loss(x, y)
        @x = x
        batch_size = y.shape[0]
        (x - y).abs.sum / batch_size
      end

      def backward(y)
        dy = @x - y
        dy[dy >= 0] = 1
        dy[dy < 0] = -1
        dy
      end
    end


    class HuberLoss < Loss
      def forward(x, y, layers)
        @loss_value = super(x, y, layers)
      end

      def loss(x, y)
        @x = x
        loss_value = loss_l1(y)
        loss_value > 1 ? loss_value : loss_l2(y)
      end

      def backward(y)
        dy = @x - y
        if @loss_value > 1
          dy[dy >= 0] = 1
          dy[dy < 0] = -1
        end
        dy
      end

      private

      def loss_l1(y)
        batch_size = y.shape[0]
        (@x - y).abs.sum / batch_size
      end

      def loss_l2(y)
        batch_size = y.shape[0]
        0.5 * ((@x - y)**2).sum / batch_size
      end
    end


    class SoftmaxCrossEntropy < Loss
      def self.softmax(x)
        NMath.exp(x) / NMath.exp(x).sum(1).reshape(x.shape[0], 1)
      end

      def loss(x, y)
        @x = SoftmaxCrossEntropy.softmax(x)
        batch_size = y.shape[0]
        -(y * NMath.log(@x + 1e-7)).sum / batch_size
      end

      def backward(y)
        @x - y
      end
    end


    class SigmoidCrossEntropy < Loss
      def initialize
        @sigmoid = Sigmoid.new
      end

      def loss(x, y)
        @x = @sigmoid.forward(x)
        batch_size = y.shape[0]
        -(y * NMath.log(@x + 1e-7) + (1 - y) * NMath.log(1 - @x + 1e-7))
      end

      def backward(y)
        @x - y
      end
    end

  end
end
