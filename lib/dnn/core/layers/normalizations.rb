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
        fs = Functions::FunctionSpace
        if DNN.learning_phase
          mean = x.mean(axis: @axis, keepdims: true)
          xc = x - mean
          var = (xc**2).mean(axis: @axis, keepdims: true)
          std = fs.sqrt(var + @eps)
          xn = xc / std
          @running_mean.data = @momentum * @running_mean.data + (1 - @momentum) * mean.data
          @running_var.data = @momentum * @running_var.data + (1 - @momentum) * var.data
        else
          xc = x - Tensor.new(@running_mean.data)
          xn = xc / fs.sqrt(Tensor.new(@running_var.data) + @eps)
        end
        gamma * xn + beta
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
