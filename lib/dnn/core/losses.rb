module DNN
  module Losses

    class Loss
      def forward(y, t, layers)
        loss_value = forward_loss(y, t)
        regularizers = layers.select { |layer| layer.is_a?(Connection) }
                             .map { |layer| layer.regularizers }.flatten
        
        regularizers.each do |regularizer|
          loss_value = regularizer.forward(loss_value)
        end
        loss_value
      end

      def backward(t, layers)
        layers.select { |layer| layer.is_a?(Connection) }.each do |layer|
          layer.regularizers.each do |regularizer|
            regularizer.backward
          end
        end
        backward_loss(t)
      end

      def to_hash(merge_hash = nil)
        hash = {class: self.class.name}
        hash.merge!(merge_hash) if merge_hash
        hash
      end

      private

      def forward_loss(y, t)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'forward_loss'")
      end

      def backward_loss(t)
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'backward_loss'")
      end
    end

    class MeanSquaredError < Loss
      private

      def forward_loss(y, t)
        @y = y
        batch_size = t.shape[0]
        0.5 * ((y - t)**2).sum / batch_size
      end

      def backward_loss(t)
        @y - t
      end
    end


    class MeanAbsoluteError < Loss
      private

      def forward_loss(y, t)
        @y = y
        batch_size = t.shape[0]
        (y - t).abs.sum / batch_size
      end

      def backward_loss(t)
        dy = @y - t
        dy[dy >= 0] = 1
        dy[dy < 0] = -1
        dy
      end
    end


    class Hinge < Loss
      private
      
      def forward_loss(y, t)
        @y = y
        @a = 1 - y * t
        @a[@a < 0] = 0
        @a
      end

      def backward_loss(t)
        a = Xumo::SFloat.ones(*@a.shape)
        a[@a <= 0] = 0
        a * -t
      end
    end


    class HuberLoss < Loss
      def forward(y, t, layers)
        @loss_value = super(y, t, layers)
      end

      private

      def forward_loss(y, t)
        @y = y
        loss_value = loss_l1(t)
        loss_value > 1 ? loss_value : loss_l2(t)
      end

      def backward_loss(t)
        dy = @y - t
        if @loss_value > 1
          dy[dy >= 0] = 1
          dy[dy < 0] = -1
        end
        dy
      end

      def loss_l1(t)
        batch_size = t.shape[0]
        (@y - t).abs.sum / batch_size
      end

      def loss_l2(t)
        batch_size = t.shape[0]
        0.5 * ((@y - t)**2).sum / batch_size
      end
    end


    class SoftmaxCrossEntropy < Loss
      # @return [Float] Return the eps value.
      attr_accessor :eps

      def self.from_hash(hash)
        SoftmaxCrossEntropy.new(eps: hash[:eps])
      end

      def self.softmax(y)
        Xumo::NMath.exp(y) / Xumo::NMath.exp(y).sum(1).reshape(y.shape[0], 1)
      end

      # @param [Float] eps Value to avoid nan.
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def to_hash
        super(eps: @eps)
      end

      private

      def forward_loss(y, t)
        @y = SoftmaxCrossEntropy.softmax(y)
        batch_size = t.shape[0]
        -(t * Xumo::NMath.log(@y + @eps)).sum / batch_size
      end

      def backward_loss(t)
        @y - t
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

      def forward_loss(y, t)
        @y = Sigmoid.new.forward(y)
        -(t * Xumo::NMath.log(@y) + (1 - t) * Xumo::NMath.log(1 - @y))
      end

      def backward_loss(t)
        @y - t
      end
    end

  end
end
