module DNN
  module Losses

    class Loss
      def forward(out, y)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward'")
      end

      def backward(y)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'backward'")
      end

      def regularize(layers)
        layers.select { |layer| layer.is_a?(Connection) }
              .reduce(0) { |sum, layer| sum + layer.lasso + layer.ridge }
      end

      def d_regularize(layers)
        layers.select { |layer| layer.is_a?(Connection) }.each do |layer|
          layer.d_lasso
          layer.d_ridge
        end
      end

      def to_hash
        {class: self.class.name}
      end
    end

    class MeanSquaredError < Loss
      def forward(out, y)
        @out = out
        batch_size = y.shape[0]
        0.5 * ((out - y)**2).sum / batch_size
      end

      def backward(y)
        @out - y
      end
    end


    class MeanAbsoluteError < Loss
      def forward(out, y)
        @out = out
        batch_size = y.shape[0]
        (out - y).abs.sum / batch_size
      end

      def backward(y)
        dout = @out - y
        dout[dout >= 0] = 1
        dout[dout < 0] = -1
        dout
      end
    end


    class HuberLoss < Loss
      def forward(out, y, layers)
        @out = out
        loss = loss_l1(y)
        loss = loss > 1 ? loss : loss_l2(y)
        #@loss = loss + regularize(layers)
        @loss = loss
      end

      def backward(y)
        dout = @out - y
        if @loss > 1
          dout[dout >= 0] = 1
          dout[dout < 0] = -1
        end
        dout
      end

      private

      def loss_l1(y)
        batch_size = y.shape[0]
        (@out - y).abs.sum / batch_size
      end

      def loss_l2(y)
        batch_size = y.shape[0]
        0.5 * ((@out - y)**2).sum / batch_size
      end
    end


    class SoftmaxCrossEntropy < Loss
      def forward(x, y)
        @out = Utils.softmax(x)
        batch_size = y.shape[0]
        -(y * NMath.log(@out + 1e-7)).sum / batch_size
      end

      def backward(y)
        @out - y
      end
    end


    class SigmoidCrossEntropy < Loss
      def forward(x, y)
        @out = Utils.sigmoid(x)
        batch_size = y.shape[0]
        -(y * NMath.log(@out + 1e-7) + (1 - y) * NMath.log(1 - @out + 1e-7)).sum / batch_size
      end

      def backward(y)
        @out - y
      end
    end

  end
end
