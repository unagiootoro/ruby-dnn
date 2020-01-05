module DNN
  module Layers

    class BatchNormalization < TrainableLayer
      include LayerNode

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
        @gamma.data = Xumo::SFloat.ones(*output_shape)
        @beta.data = Xumo::SFloat.zeros(*output_shape)
        @running_mean.data = Xumo::SFloat.zeros(*output_shape)
        @running_var.data = Xumo::SFloat.zeros(*output_shape)
      end

      def forward_node(x)
        if DNN.learning_phase
          mean = x.mean(axis: @axis, keepdims: true)
          @xc = x - mean
          var = (@xc**2).mean(axis: @axis, keepdims: true)
          @std = Xumo::NMath.sqrt(var + @eps)
          xn = @xc / @std
          @xn = xn
          @running_mean.data = @momentum * @running_mean.data + (1 - @momentum) * mean
          @running_var.data = @momentum * @running_var.data + (1 - @momentum) * var
        else
          xc = x - @running_mean.data
          xn = xc / Xumo::NMath.sqrt(@running_var.data + @eps)
        end
        @gamma.data * xn + @beta.data
      end

      def backward_node(dy)
        batch_size = dy.shape[@axis]
        if @trainable
          @beta.grad = dy.sum(axis: @axis, keepdims: true)
          @gamma.grad = (@xn * dy).sum(axis: @axis, keepdims: true)
        end
        dxn = @gamma.data * dy
        dxc = dxn / @std
        dstd = -((dxn * @xc) / (@std**2)).sum(axis: @axis, keepdims: true)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / batch_size) * @xc * dvar
        dmean = dxc.sum(axis: @axis, keepdims: true)
        dxc - dmean / batch_size
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
