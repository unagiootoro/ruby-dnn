module DNN
  module Layers

    class BatchNormalization < TrainableLayer
      attr_reader :gamma
      attr_reader :beta
      attr_reader :running_mean
      attr_reader :running_var
      attr_reader :axis
      attr_accessor :momentum
      attr_accessor :eps

      # @param [Integer] axis The axis to normalization.
      # @param [Float] momentum Exponential moving average of mean and variance.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(axis: 0, momentum: 0.9, eps: 1e-7)
        super()
        @axis = axis
        @momentum = momentum
        @eps = eps
        @gamma = Param.new(nil, Xumo::SFloat[0])
        @beta = Param.new(nil, Xumo::SFloat[0])
        @running_mean = Param.new
        @running_var = Param.new
      end

      def build(input_shape)
        super
        @gamma.data = Xumo::SFloat.ones(*@output_shape)
        @beta.data = Xumo::SFloat.zeros(*@output_shape)
        @running_mean.data = Xumo::SFloat.zeros(*@output_shape)
        @running_var.data = Xumo::SFloat.zeros(*@output_shape)
      end

      def forward(x)
        batch_norm = Functions::BatchNormalization.new(running_mean.data, running_var.data, axis: @axis, momentum: @momentum, eps: @eps, learning_phase: DNN.learning_phase)
        y = batch_norm.(x, @gamma, @beta)
        @running_mean.data = batch_norm.running_mean
        @running_var.data = batch_norm.running_var
        y
      end

      def to_hash
        super(axis: @axis, momentum: @momentum, eps: @eps)
      end

      def load_hash(hash)
        initialize(axis: hash[:axis], momentum: hash[:momentum])
      end

      def get_params
        { gamma: @gamma, beta: @beta, running_mean: @running_mean, running_var: @running_var }
      end
    end

  end
end
