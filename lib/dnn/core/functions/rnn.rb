module DNN
  module Functions
    class RNNTimeSplit < Function
      def forward(xs)
        @xs_shape = xs.shape
        y_outputs = []
        xs.shape[1].times do |t|
          y_outputs << xs[true, t, false]
        end
        y_outputs
      end

      def backward(*dy_inputs)
        dxs = Xumo::SFloat.zeros(@xs_shape)
        dy_inputs.each.with_index do |dy, t|
          dxs[true, t, false] = dy
        end
        dxs
      end
    end

    class RNNTimeConcatenate < Function
      def forward(*x_inputs)
        ys = Xumo::SFloat.zeros(x_inputs[0].shape[0], x_inputs.length, *x_inputs[0].shape[1..-1])
        x_inputs.each.with_index do |x, t|
          ys[true, t, false] = x
        end
        ys
      end

      def backward(dys)
        x_outputs = []
        dys.shape[1].times do |t|
          x_outputs << dys[true, t, false]
        end
        x_outputs
      end
    end

    class SimpleRNNCell < Function
      def initialize(requires_weight_grad: true)
        @requires_weight_grad = requires_weight_grad
      end

      def forward(x, h, weight, recurrent_weight, bias = nil)
        @x = x
        @h = h
        @weight = weight
        @recurrent_weight = recurrent_weight
        @bias = bias
        h2 = x.dot(@weight) + h.dot(@recurrent_weight)
        h2 += @bias if @bias
        h2
      end

      def backward(dh2)
        if @requires_weight_grad
          dweight = @x.transpose.dot(dh2)
          drecurrent_weight = @h.transpose.dot(dh2)
          dbias = dh2.sum(0) if @bias
        else
          dweight = Xumo:SFloat.zeros(*@weight.shape)
          drecurrent_weight = Xumo:SFloat.zeros(*@recurrent_weight.shape)
          dbias = Xumo:SFloat.zeros(*@bias.shape) if @bias
        end
        dx = dh2.dot(@weight.transpose)
        dh = dh2.dot(@recurrent_weight.transpose)
        if @bias
          [dx, dh, dweight, drecurrent_weight, dbias]
        else
          [dx, dh, dweight, drecurrent_weight]
        end
      end
    end

    class LSTMCell < Function
      def initialize(return_c: true, requires_weight_grad: true)
        @requires_weight_grad = requires_weight_grad
        @tanh = Functions::Tanh.new
        @g_tanh = Functions::Tanh.new
        @forget_sigmoid = Functions::Sigmoid.new
        @in_sigmoid = Functions::Sigmoid.new
        @out_sigmoid = Functions::Sigmoid.new
        @return_c = return_c
      end

      def forward(x, h, c, weight, recurrent_weight, bias = nil)
        @x = x
        @h = h
        @c = c
        @weight = weight
        @recurrent_weight = recurrent_weight
        @bias = bias
        num_units = h.shape[1]
        a = x.dot(weight) + h.dot(recurrent_weight)
        a += bias if bias

        @forget = @forget_sigmoid.forward(a[true, 0...num_units])
        @g = @g_tanh.forward(a[true, num_units...(num_units * 2)])
        @in = @in_sigmoid.forward(a[true, (num_units * 2)...(num_units * 3)])
        @out = @out_sigmoid.forward(a[true, (num_units * 3)..-1])

        c2 = @forget * c + @g * @in
        @tanh_c2 = @tanh.forward(c2)
        h2 = @out * @tanh_c2
        @return_c ? [h2, c2] : h2
      end

      def backward(dh2, dc2 = nil)
        dh2_tmp = @tanh_c2 * dh2
        dc2_tmp = @tanh.backward(@out * dh2)
        dc2_tmp += dc2 if dc2

        dout = @out_sigmoid.backward(dh2_tmp)
        din = @in_sigmoid.backward(dc2_tmp * @g)
        dg = @g_tanh.backward(dc2_tmp * @in)
        dforget = @forget_sigmoid.backward(dc2_tmp * @c)

        da = Xumo::SFloat.hstack([dforget, dg, din, dout])

        if @requires_weight_grad
          dweight = @x.transpose.dot(da)
          drecurrent_weight = @h.transpose.dot(da)
          dbias = da.sum(0) if @bias
        else
          dweight = Xumo:SFloat.zeros(*@weight.shape)
          drecurrent_weight = Xumo:SFloat.zeros(*@recurrent_weight.shape)
          dbias = Xumo:SFloat.zeros(*@bias.shape) if @bias
        end
        dx = da.dot(@weight.transpose)
        dh = da.dot(@recurrent_weight.transpose)
        dc = dc2_tmp * @forget
        if @requires_weight_grad
          [dx, dh, dc, dweight, drecurrent_weight, dbias]
        else
          [dx, dh, dc, dweight, drecurrent_weight]
        end
      end
    end

    class GRUCell < Function
      def initialize(requires_weight_grad: true)
        @requires_weight_grad = requires_weight_grad
        @update_sigmoid = Functions::Sigmoid.new
        @reset_sigmoid = Functions::Sigmoid.new
        @tanh = Functions::Tanh.new
      end

      def forward(x, h, weight, recurrent_weight, bias = nil)
        @x = x
        @h = h
        @weight = weight
        @recurrent_weight = recurrent_weight
        @bias = bias
        num_units = h.shape[1]
        @weight_a = weight[true, 0...(num_units * 2)]
        @weight2_a = recurrent_weight[true, 0...(num_units * 2)]
        a = x.dot(@weight_a) + h.dot(@weight2_a)
        a += bias[0...(num_units * 2)] if bias
        @update = @update_sigmoid.forward(a[true, 0...num_units])
        @reset = @reset_sigmoid.forward(a[true, num_units..-1])

        @weight_h = weight[true, (num_units * 2)..-1]
        @weight2_h = recurrent_weight[true, (num_units * 2)..-1]
        @tanh_h = if bias
                    bias_h = bias[(num_units * 2)..-1]
                    @tanh.forward(x.dot(@weight_h) + (h * @reset).dot(@weight2_h) + bias_h)
                  else
                    @tanh.forward(x.dot(@weight_h) + (h * @reset).dot(@weight2_h))
                  end
        h2 = (1 - @update) * @tanh_h + @update * h
        h2
      end

      def backward(dh2)
        dtanh_h = @tanh.backward(dh2 * (1 - @update))
        dh = dh2 * @update

        if @requires_weight_grad
          dweight_h = @x.transpose.dot(dtanh_h)
          dweight2_h = (@h * @reset).transpose.dot(dtanh_h)
          dbias_h = dtanh_h.sum(0) if @bias
        end
        dx = dtanh_h.dot(@weight_h.transpose)
        dh += dtanh_h.dot(@weight2_h.transpose) * @reset

        dreset = @reset_sigmoid.backward(dtanh_h.dot(@weight2_h.transpose) * @h)
        dupdate = @update_sigmoid.backward(dh2 * @h - dh2 * @tanh_h)
        da = Xumo::SFloat.hstack([dupdate, dreset])
        if @requires_weight_grad
          dweight_a = @x.transpose.dot(da)
          dweight2_a = @h.transpose.dot(da)
          dbias_a = da.sum(0) if @bias
        end
        dx += da.dot(@weight_a.transpose)
        dh += da.dot(@weight2_a.transpose)

        if @requires_weight_grad
          dweight = Xumo::SFloat.hstack([dweight_a, dweight_h])
          drecurrent_weight = Xumo::SFloat.hstack([dweight2_a, dweight2_h])
          dbias = Xumo::SFloat.hstack([dbias_a, dbias_h]) if @bias
        else
          dweight = Xumo:SFloat.zeros(*@weight.shape)
          drecurrent_weight = Xumo:SFloat.zeros(*@recurrent_weight.shape)
          dbias = Xumo:SFloat.zeros(*@bias.shape) if @bias
        end
        if @bias
          [dx, dh, dweight, drecurrent_weight, dbias]
        else
          [dx, dh, dweight, drecurrent_weight]
        end
      end
    end

  end
end
