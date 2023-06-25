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

      def forward(y, t)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward'"
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
      def forward(y, t)
        Functions::FunctionSpace.mean_squared_error(y, t)
      end
    end

    class MeanAbsoluteError < Loss
      def forward(y, t)
        Functions::FunctionSpace.mean_absolute_error(y, t)
      end
    end

    class Hinge < Loss
      def forward(y, t)
        Functions::FunctionSpace.hinge(y, t)
      end
    end

    class HuberLoss < Loss
      def forward(y, t)
        Functions::FunctionSpace.huber_loss(y, t)
      end
    end

    class SoftmaxCrossEntropy < Loss
      attr_accessor :eps

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
