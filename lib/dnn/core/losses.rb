module DNN
  module Losses

    class Loss
      def self.call(y, t, *args)
        new(*args).(y, t)
      end

      def self.from_hash(hash)
        return nil unless hash
        loss_class = DNN.const_get(hash[:class])
        loss = loss_class.allocate
        raise DNN_Error, "#{loss.class} is not an instance of #{self} class." unless loss.is_a?(self)
        loss.load_hash(hash)
        loss
      end

      def call(y, t)
        forward(y, t)
      end

      def loss(y, t, layers = nil)
        unless y.shape == t.shape
          raise DNN_ShapeError, "The shape of y does not match the t shape. y shape is #{y.shape}, but t shape is #{t.shape}."
        end
        loss = call(y, t)
        loss = regularizers_forward(loss, layers) if layers
        loss
      end

      def forward(y, t)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward'"
      end

      def regularizers_forward(loss, layers)
        regularizers = layers.select { |layer| layer.respond_to?(:regularizers) }
                             .map(&:regularizers).flatten
        regularizers.each do |regularizer|
          loss = regularizer.forward(loss)
        end
        loss
      end

      def regularizers_backward(layers)
        layers.select { |layer| layer.respond_to?(:regularizers) }.each do |layer|
          layer.regularizers.each(&:backward)
        end
      end

      def to_hash(merge_hash = nil)
        hash = { class: self.class.name }
        hash.merge!(merge_hash) if merge_hash
        hash
      end

      def load_hash(hash)
        initialize
      end

      def clean
        hash = to_hash
        instance_variables.each do |ivar|
          instance_variable_set(ivar, nil)
        end
        load_hash(hash)
      end
    end

    class MeanSquaredError < Loss
      include Layers::MergeLayerNode

      def forward_node(y, t)
        @y = y
        @t = t
        0.5 * ((y - t)**2).mean(0).sum
      end

      def backward_node(dy)
        @y - @t
      end
    end

    class MeanAbsoluteError < Loss
      include Layers::MergeLayerNode

      def forward_node(y, t)
        @y = y
        @t = t
        (y - t).abs.mean(0).sum
      end

      def backward_node(d)
        dy = @y - @t
        dy[dy >= 0] = 1
        dy[dy < 0] = -1
        dy
      end
    end

    class Hinge < Loss
      include Layers::MergeLayerNode

      def forward_node(y, t)
        @t = t
        @a = 1 - y * t
        Xumo::SFloat.maximum(0, @a).mean(0).sum
      end

      def backward_node(d)
        a = Xumo::SFloat.ones(*@a.shape)
        a[@a <= 0] = 0
        a * -@t
      end
    end

    class HuberLoss < Loss
      include Layers::MergeLayerNode

      def forward_node(y, t)
        @y = y
        @t = t
        loss_l1_value = loss_l1(y, t)
        @loss_value = loss_l1_value > 1 ? loss_l1_value : loss_l2(y, t)
      end

      def backward_node(d)
        dy = @y - @t
        if @loss_value > 1
          dy[dy >= 0] = 1
          dy[dy < 0] = -1
        end
        dy
      end

      private

      def loss_l1(y, t)
        (y - t).abs.mean(0).sum
      end

      def loss_l2(y, t)
        0.5 * ((y - t)**2).mean(0).sum
      end
    end

    class SoftmaxCrossEntropy < Loss
      include Layers::MergeLayerNode

      attr_accessor :eps

      class << self
        def softmax(y)
          Xumo::NMath.exp(y) / Xumo::NMath.exp(y).sum(1, keepdims: true)
        end

        alias activation softmax
      end

      # @param [Float] eps Value to avoid nan.
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def forward_node(y, t)
        @t = t
        @x = SoftmaxCrossEntropy.softmax(y)
        -(t * Xumo::NMath.log(@x + @eps)).mean(0).sum
      end

      def backward_node(d)
        @x - @t
      end

      def to_hash
        super(eps: @eps)
      end

      def load_hash(hash)
        initialize(eps: hash[:eps])
      end
    end

    class SigmoidCrossEntropy < Loss
      include Layers::MergeLayerNode
      
      attr_accessor :eps

      class << self
        def sigmoid(y)
          Layers::Sigmoid.new.forward_node(y)
        end

        alias activation sigmoid
      end

      # @param [Float] eps Value to avoid nan.
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def forward_node(y, t)
        @t = t
        @x = SigmoidCrossEntropy.sigmoid(y)
        -(t * Xumo::NMath.log(@x + @eps) + (1 - t) * Xumo::NMath.log(1 - @x + @eps)).mean(0).sum
      end

      def backward_node(d)
        @x - @t
      end

      def to_hash
        super(eps: @eps)
      end

      def load_hash(hash)
        initialize(eps: hash[:eps])
      end
    end

  end
end
