module DNN
  module Losses

    class Loss
      def forward(x, y, layers)
        loss_value = forward_loss(x, y)
        regularizers = layers.select { |layer| layer.is_a?(Connection) }
                             .map { |layer| layer.regularizers }.flatten
        
        regularizers.each do |regularizer|
          loss_value = regularizer.forward(loss_value)
        end
        loss_value
      end

      def backward(y, layers)
        layers.select { |layer| layer.is_a?(Connection) }.each do |layer|
          layer.regularizers.each do |regularizer|
            regularizer.backward
          end
        end
        backward_loss(y)
      end

      def to_hash(merge_hash = nil)
        hash = {class: self.class.name}
        hash.merge!(merge_hash) if merge_hash
        hash
      end

      private

      def forward_loss(x, y)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward_loss'")
      end

      def backward_loss(y)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'backward_loss'")
      end
    end

    class MeanSquaredError < Loss
      private

      def forward_loss(x, y)
        @x = x
        batch_size = y.shape[0]
        0.5 * ((x - y)**2).sum / batch_size
      end

      def backward_loss(y)
        @x - y
      end
    end


    class MeanAbsoluteError < Loss
      private

      def forward_loss(x, y)
        @x = x
        batch_size = y.shape[0]
        (x - y).abs.sum / batch_size
      end

      def backward_loss(y)
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

      private

      def forward_loss(x, y)
        @x = x
        loss_value = loss_l1(y)
        loss_value > 1 ? loss_value : loss_l2(y)
      end

      def backward_loss(y)
        dy = @x - y
        if @loss_value > 1
          dy[dy >= 0] = 1
          dy[dy < 0] = -1
        end
        dy
      end

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
      # @return [Float] Return the eps value.
      attr_accessor :eps

      def self.from_hash(hash)
        SoftmaxCrossEntropy.new(eps: hash[:eps])
      end

      def self.softmax(x)
        NMath.exp(x) / NMath.exp(x).sum(1).reshape(x.shape[0], 1)
      end

      # @param [Float] eps Value to avoid nan.
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def to_hash
        super(eps: @eps)
      end

      private

      def forward_loss(x, y)
        @x = SoftmaxCrossEntropy.softmax(x)
        batch_size = y.shape[0]
        -(y * NMath.log(@x + @eps)).sum / batch_size
      end

      def backward_loss(y)
        @x - y
      end
    end


    class SigmoidCrossEntropy < Loss
      # @return [Float] Return the eps value.
      attr_accessor :eps

      def self.from_hash(hash)
        SigmoidCrossEntropy.new(eps: hash[:eps])
      end

      # @param [Float] eps Value to avoid nan.
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def to_hash
        super(eps: @eps)
      end

      private

      def forward_loss(x, y)
        @x = Sigmoid.new.forward(x)
        batch_size = y.shape[0]
        -(y * NMath.log(@x) + (1 - y) * NMath.log(1 - @x))
      end

      def backward_loss(y)
        @x - y
      end
    end

  end
end
