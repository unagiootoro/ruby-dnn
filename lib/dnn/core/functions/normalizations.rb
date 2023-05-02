module DNN
  module Functions

    class BatchNormalization < Function
      attr_reader :running_mean
      attr_reader :running_var

      def initialize(running_mean, running_var, axis: 0, momentum: 0.9, eps: 1e-7, learning_phase: false)
        super()
        @running_mean = running_mean
        @running_var = running_var
        @axis = axis
        @momentum = momentum
        @eps = eps
        @learning_phase = learning_phase
      end

      def forward(x, gamma, beta)
        if @learning_phase
          @gamma = gamma
          mean = x.mean(axis: @axis, keepdims: true)
          @xc = x - mean
          var = (@xc**2).mean(axis: @axis, keepdims: true)
          @std = Xumo::NMath.sqrt(var + @eps)
          xn = @xc / @std
          @xn = xn
          @running_mean = @momentum * @running_mean + (1 - @momentum) * mean
          @running_var = @momentum * @running_var + (1 - @momentum) * var
        else
          xc = x - @running_mean
          xn = xc / Xumo::NMath.sqrt(@running_var + @eps)
        end
        gamma * xn + beta
      end

      def backward(dy)
        batch_size = dy.shape[@axis]
        dbeta = dy.sum(axis: @axis, keepdims: true)
        dgamma = (@xn * dy).sum(axis: @axis, keepdims: true)
        dxn = @gamma * dy
        dxc = dxn / @std
        dstd = -((dxn * @xc) / (@std**2)).sum(axis: @axis, keepdims: true)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / batch_size) * @xc * dvar
        dmean = dxc.sum(axis: @axis, keepdims: true)
        dx = dxc - dmean / batch_size
        [dx, dgamma, dbeta]
      end
    end

  end
end
