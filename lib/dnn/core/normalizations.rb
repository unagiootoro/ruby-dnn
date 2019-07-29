module DNN
  module Layers

    class BatchNormalization < HasParamLayer
      attr_reader :gamma
      attr_reader :beta
      attr_reader :running_mean
      attr_reader :running_var
      attr_reader :axis
      attr_accessor :momentum
      attr_accessor :eps

      def self.from_hash(hash)
        self.new(axis: hash[:axis], momentum: hash[:momentum])
      end

      # @param [Integer] axis The axis to normalization.
      # @param [Float] momentum Exponential moving average of mean and variance.
      # @param [Float] eps Value to avoid division by zero.
      def initialize(axis: 0, momentum: 0.9, eps: 1e-7)
        super()
        @axis = axis
        @momentum = momentum
        @eps = eps
      end

      def call(input)
        x, prev_link, model = *input
        build(x.shape[1..-1]) unless built?
        y = forward(x, model.learning_phase)
        link = Link.new(prev_link, self)
        prev_link.next = link
        [y, link, model]
      end

      def build(input_shape)
        super
        @gamma = Param.new(Xumo::SFloat.ones(*output_shape), 0)
        @beta = Param.new(Xumo::SFloat.zeros(*output_shape), 0)
        @running_mean = Param.new(Xumo::SFloat.zeros(*output_shape))
        @running_var = Param.new(Xumo::SFloat.zeros(*output_shape))
      end

      def forward(x, learning_phase)
        if learning_phase
          mean = x.mean(axis: @axis, keepdims: true)
          @xc = x - mean
          var = (@xc ** 2).mean(axis: @axis, keepdims: true)
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

      def backward(dy)
        batch_size = dy.shape[@axis]
        if @trainable
          @beta.grad = dy.sum(axis: @axis, keepdims: true)
          @gamma.grad = (@xn * dy).sum(axis: @axis, keepdims: true)
        end
        dxn = @gamma.data * dy
        dxc = dxn / @std
        dstd = -((dxn * @xc) / (@std ** 2)).sum(axis: @axis, keepdims: true)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / batch_size) * @xc * dvar
        dmean = dxc.sum(axis: @axis, keepdims: true)
        dxc - dmean / batch_size
      end

      def to_hash
        super(axis: @axis, momentum: @momentum, eps: @eps)
      end

      def get_params
        {gamma: @gamma, beta: @beta, running_mean: @running_mean, running_var: @running_var}
      end
    end

  end
end
