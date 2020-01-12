module DNN
  module Regularizers

    class Regularizer
      attr_accessor :param

      def self.from_hash(hash)
        return nil unless hash
        regularizer_class = DNN.const_get(hash[:class])
        regularizer = regularizer_class.allocate
        raise DNNError, "#{regularizer.class} is not an instance of #{self} class." unless regularizer.is_a?(self)
        regularizer.load_hash(hash)
        regularizer
      end

      def forward(x)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward'"
      end

      def to_hash(merge_hash)
        hash = { class: self.class.name }
        hash.merge!(merge_hash)
        hash
      end

      def load_hash(hash)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'load_hash'"
      end
    end

    class L1 < Regularizer
      # @param [Float] l1_lambda L1 regularizer coefficient.
      def initialize(l1_lambda = 0.01)
        @l1 = Layers::Lasso.new(l1_lambda)
      end

      def forward(x)
        x + @l1.(@param)
      end

      def l1_lambda
        @l1.l1_lambda
      end

      def l1_lambda=(lam)
        @l1.l1_lambda = lam
      end

      def to_hash
        super(l1_lambda: l1_lambda)
      end

      def load_hash(hash)
        initialize(hash[:l1_lambda])
      end
    end

    class L2 < Regularizer
      # @param [Float] l2_lambda L2 regularizer coefficient.
      def initialize(l2_lambda = 0.01)
        @l2 = Layers::Ridge.new(l2_lambda)
      end

      def forward(x)
        x + @l2.(@param)
      end

      def l2_lambda
        @l2.l2_lambda
      end

      def l2_lambda=(lam)
        @l2.l2_lambda = lam
      end

      def to_hash
        super(l2_lambda: l2_lambda)
      end

      def load_hash(hash)
        initialize(hash[:l2_lambda])
      end
    end

    class L1L2 < Regularizer
      # @param [Float] l1_lambda L1 regularizer coefficient.
      # @param [Float] l2_lambda L2 regularizer coefficient.
      def initialize(l1_lambda = 0.01, l2_lambda = 0.01)
        @l1 = Layers::Lasso.new(l1_lambda)
        @l2 = Layers::Ridge.new(l2_lambda)
      end

      def forward(x)
        x + @l1.(@param) + @l2.(@param)
      end

      def l1_lambda
        @l1.l1_lambda
      end

      def l1_lambda=(lam)
        @l1.l1_lambda = lam
      end

      def l2_lambda
        @l2.l2_lambda
      end

      def l2_lambda=(lam)
        @l2.l2_lambda = lam
      end

      def to_hash
        super(l1_lambda: l1_lambda, l2_lambda: l2_lambda)
      end

      def load_hash(hash)
        initialize(hash[:l1_lambda], hash[:l2_lambda])
      end
    end

  end
end
