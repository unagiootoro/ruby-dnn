module DNN
  module Losses

    class Loss
      def self.call(y, t, *args, **kwargs)
        new(*args, **kwargs).(y, t)
      end

      def self.from_hash(hash)
        return nil unless hash
        loss_class = DNN.const_get(hash[:class])
        loss = loss_class.allocate
        raise DNNError, "#{loss.class} is not an instance of #{self} class." unless loss.is_a?(self)
        loss.load_hash(hash)
        loss
      end

      def call(y, t)
        forward(y, t)
      end

      def loss(y, t, layers: nil, loss_weight: nil)
        unless y.shape == t.shape
          raise DNNShapeError, "The shape of y does not match the t shape. y shape is #{y.shape}, but t shape is #{t.shape}."
        end
        loss = call(y, t)
        loss *= loss_weight if loss_weight
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
      include Layers::LayerNode

      def forward_node(y, t)
        @y = y
        @t = t
        0.5 * ((y - t)**2).mean(0).sum
      end

      def backward_node(d)
        d * (@y - @t) / @y.shape[0]
      end
    end

    class MeanAbsoluteError < Loss
      include Layers::LayerNode

      def forward_node(y, t)
        @y = y
        @t = t
        (y - t).abs.mean(0).sum
      end

      def backward_node(d)
        dy = (@y - @t)
        dy[dy >= 0] = 1
        dy[dy < 0] = -1
        d * dy / @y.shape[0]
      end
    end

    class Hinge < Loss
      include Layers::LayerNode

      def forward_node(y, t)
        @t = t
        @a = 1 - y * t
        Xumo::SFloat.maximum(0, @a).mean(0).sum
      end

      def backward_node(d)
        a = Xumo::SFloat.ones(*@a.shape)
        a[@a <= 0] = 0
        d * (a * -@t) / a.shape[0]
      end
    end

    class HuberLoss < Loss
      include Layers::LayerNode

      def forward_node(y, t)
        @y = y
        @t = t
        loss_l1_value = (y - t).abs.mean(0).sum
        @loss_value = loss_l1_value > 1 ? loss_l1_value : 0.5 * ((y - t)**2).mean(0).sum
      end

      def backward_node(d)
        dy = (@y - @t)
        if @loss_value > 1
          dy[dy >= 0] = 1
          dy[dy < 0] = -1
        end
        d * dy / @y.shape[0]
      end
    end

    class SoftmaxCrossEntropy < Loss
      attr_accessor :eps

      class << self
        def softmax(y)
          Functions::SoftmaxCrossEntropy.softmax(y)
        end

        alias activation softmax
      end

      # @param [Float] eps Value to avoid nan.
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def forward(y, t)
        Functions::FunctionSpace.softmax_cross_entropy(y, t, eps: @eps)
      end

      def to_hash
        super(eps: @eps)
      end

      def load_hash(hash)
        initialize(eps: hash[:eps])
      end
    end

    class SigmoidCrossEntropy < Loss
      attr_accessor :eps

      class << self
        def sigmoid(y)
          Functions::SigmoidCrossEntropy.sigmoid(y)
        end

        alias activation sigmoid
      end

      # @param [Float] eps Value to avoid nan.
      def initialize(eps: 1e-7)
        @eps = eps
      end

      def forward(y, t)
        Functions::FunctionSpace.sigmoid_cross_entropy(y, t, eps: @eps)
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
