module DNN
  module Layers

    class BatchNormalization < HasParamLayer
      # @return [Integer] The axis to normalization.
      attr_reader :axis
      # @return [Float] Exponential moving average of mean and variance.
      attr_accessor :momentum
      # @return [Float] Value to avoid division by zero.
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(axis: hash[:axis], momentum: hash[:momentum])
      end

      # @param [integer] axis The axis to normalization.
      # @param [Float] momentum Exponential moving average of mean and variance.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(axis: 0, momentum: 0.9, eps: 1e-7)
        super()
        @axis = axis
        @momentum = momentum
        @eps = eps
      end

      def build(input_shape)
        super
        @params[:gamma] = @gamma = Param.new(Xumo::SFloat.ones(*output_shape), 0)
        @params[:beta] = @beta = Param.new(Xumo::SFloat.zeros(*output_shape), 0)
        @params[:running_mean] = @running_mean = Param.new(Xumo::SFloat.zeros(*output_shape))
        @params[:running_var] = @running_var = Param.new(Xumo::SFloat.zeros(*output_shape))
      end

      def forward(x)
        if learning_phase
          mean = x.mean(axis: @axis, keepdims: true)
          @xc = x - mean
          var = (@xc**2).mean(axis: @axis, keepdims: true)
          @std = NMath.sqrt(var + @eps)
          xn = @xc / @std
          @xn = xn
          @running_mean.data = @momentum * @running_mean.data + (1 - @momentum) * mean
          @running_var.data = @momentum * @running_var.data + (1 - @momentum) * var
        else
          xc = x - @running_mean.data
          xn = xc / NMath.sqrt(@running_var.data + @eps)
        end
        @gamma.data * xn + @beta.data
      end

      def backward(dy)
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
        super({axis: @axis, momentum: @momentum, eps: @eps})
      end
    end

  end
end
