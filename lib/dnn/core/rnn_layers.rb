module DNN
  module Layers

    # Super class of all RNN classes.
    class RNN < Connection
      include Initializers

      # @return [Integer] number of nodes.
      attr_reader :num_nodes
      # @return [Bool] Maintain state between batches.
      attr_reader :stateful
      # @return [Bool] Set the false, only the last of each cell of RNN is left.
      attr_reader :return_sequences

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: RandomNormal.new,
                     bias_initializer: Zeros.new,
                     l1_lambda: 0,
                     l2_lambda: 0,
                     use_bias: true)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              l1_lambda: l1_lambda, l2_lambda: l2_lambda, use_bias: use_bias)
        @num_nodes = num_nodes
        @stateful = stateful
        @return_sequences = return_sequences
        @layers = []
        @hidden = @params[:h] = Param.new
        # TODO
        # Change to a good name.
        @params[:weight2] = @weight2 = Param.new(nil, 0)
      end

      def build(input_shape)
        super
        @time_length = @input_shape[0]
      end

      def forward(xs)
        @xs_shape = xs.shape
        hs = Xumo::SFloat.zeros(xs.shape[0], @time_length, @num_nodes)
        h = (@stateful && @hidden.data) ? @hidden.data : Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        xs.shape[1].times do |t|
          x = xs[true, t, false]
          h = @layers[t].forward(x, h)
          hs[true, t, false] = h
        end
        @hidden.data = h
        @return_sequences ? hs : h
      end

      def backward(dh2s)
        @weight.grad += Xumo::SFloat.zeros(*@weight.data.shape)
        @weight2.grad += Xumo::SFloat.zeros(*@weight2.data.shape)
        @bias.grad += Xumo::SFloat.zeros(*@bias.data.shape) if @bias
        unless @return_sequences
          dh = dh2s
          dh2s = Xumo::SFloat.zeros(dh.shape[0], @time_length, dh.shape[1])
          dh2s[true, -1, false] = dh
        end
        dxs = Xumo::SFloat.zeros(@xs_shape)
        dh = 0
        (0...dh2s.shape[1]).to_a.reverse.each do |t|
          dh2 = dh2s[true, t, false]
          dx, dh = @layers[t].backward(dh2 + dh)
          dxs[true, t, false] = dx
        end
        dxs
      end

      def output_shape
        @return_sequences ? [@time_length, @num_nodes] : [@num_nodes]
      end

      def to_hash(merge_hash = nil)
        hash = {
          num_nodes: @num_nodes,
          stateful: @stateful,
          return_sequences: @return_sequences
        }
        hash.merge!(merge_hash) if merge_hash
        super(hash)
      end

      # Reset the state of RNN.
      def reset_state
        @hidden.data = @hidden.data.fill(0) if @hidden.data
      end

      def regularizers
        regularizers = []
        if @l1_lambda > 0
          regularizers << Lasso.new(@l1_lambda, @weight)
          regularizers << Lasso.new(@l1_lambda, @weight2)
        end
        if @l2_lambda > 0
          regularizers << Ridge.new(@l2_lambda, @weight)
          regularizers << Ridge.new(@l2_lambda, @weight2)
        end
        regularizers
      end
    end


    class SimpleRNN_Dense
      def initialize(weight, weight2, bias, activation)
        @weight = weight
        @weight2 = weight2
        @bias = bias
        @activation = activation.clone
      end

      def forward(x, h)
        @x = x
        @h = h
        h2 = x.dot(@weight.data) + h.dot(@weight2.data)
        h2 += @bias.data if @bias
        @activation.forward(h2)
      end

      def backward(dh2)
        dh2 = @activation.backward(dh2)
        @weight.grad += @x.transpose.dot(dh2)
        @weight2.grad += @h.transpose.dot(dh2)
        @bias.grad += dh2.sum(0) if @bias
        dx = dh2.dot(@weight.data.transpose)
        dh = dh2.dot(@weight2.data.transpose)
        [dx, dh]
      end
    end


    class SimpleRNN < RNN
      include Activations

      attr_reader :activation
      
      def self.from_hash(hash)
        simple_rnn = self.new(hash[:num_nodes],
                              stateful: hash[:stateful],
                              return_sequences: hash[:return_sequences],
                              activation: Utils.from_hash(hash[:activation]),
                              weight_initializer: Utils.from_hash(hash[:weight_initializer]),
                              bias_initializer: Utils.from_hash(hash[:bias_initializer]),
                              l1_lambda: hash[:l1_lambda],
                              l2_lambda: hash[:l2_lambda],
                              use_bias: hash[:use_bias])
        simple_rnn
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     activation: Tanh.new,
                     weight_initializer: RandomNormal.new,
                     bias_initializer: Zeros.new,
                     l1_lambda: 0,
                     l2_lambda: 0,
                     use_bias: true)
        super(num_nodes,
              stateful: stateful,
              return_sequences: return_sequences,
              weight_initializer: weight_initializer,
              bias_initializer: bias_initializer,
              l1_lambda: l1_lambda,
              l2_lambda: l2_lambda,
              use_bias: use_bias)
        @activation = activation
      end

      def build(input_shape)
        super
        num_prev_nodes = input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes)
        @weight_initializer.init_param(self, @weight)
        @weight2.data = Xumo::SFloat.new(@num_nodes, @num_nodes)
        @weight_initializer.init_param(self, @weight2)
        if @bias
          @bias.data = Xumo::SFloat.new(@num_nodes)
          @bias_initializer.init_param(self, @bias) 
        end
        @time_length.times do |t|
          @layers << SimpleRNN_Dense.new(@weight, @weight2, @bias, @activation)
        end
      end

      def to_hash
        super({activation: @activation.to_hash})
      end
    end


    class LSTM_Dense
      def initialize(weight, weight2, bias)
        @weight = weight
        @weight2 = weight2
        @bias = bias
        @tanh = Tanh.new
        @g_tanh = Tanh.new
        @forget_sigmoid = Sigmoid.new
        @in_sigmoid = Sigmoid.new
        @out_sigmoid = Sigmoid.new
      end

      def forward(x, h, c)
        @x = x
        @h = h
        @c = c
        num_nodes = h.shape[1]
        a = x.dot(@weight.data) + h.dot(@weight2.data)
        a += @bias.data if @bias

        @forget = @forget_sigmoid.forward(a[true, 0...num_nodes])
        @g = @g_tanh.forward(a[true, num_nodes...(num_nodes * 2)])
        @in = @in_sigmoid.forward(a[true, (num_nodes * 2)...(num_nodes * 3)])
        @out = @out_sigmoid.forward(a[true, (num_nodes * 3)..-1])

        c2 = @forget * c + @g * @in
        @tanh_c2 = @tanh.forward(c2)
        h2 = @out * @tanh_c2
        [h2, c2]
      end

      def backward(dh2, dc2)
        dh2_tmp = @tanh_c2 * dh2
        dc2_tmp = @tanh.backward(@out * dh2) + dc2

        dout = @out_sigmoid.backward(dh2_tmp)
        din = @in_sigmoid.backward(dc2_tmp * @g)
        dg = @g_tanh.backward(dc2_tmp * @in)
        dforget = @forget_sigmoid.backward(dc2_tmp * @c)

        da = Xumo::SFloat.hstack([dforget, dg, din, dout])

        @weight.grad += @x.transpose.dot(da)
        @weight2.grad += @h.transpose.dot(da)
        @bias.grad += da.sum(0) if @bias
        dx = da.dot(@weight.data.transpose)
        dh = da.dot(@weight2.data.transpose)
        dc = dc2_tmp * @forget
        [dx, dh, dc]
      end
    end


    class LSTM < RNN
      def self.from_hash(hash)
        lstm = self.new(hash[:num_nodes],
                        stateful: hash[:stateful],
                        return_sequences: hash[:return_sequences],
                        weight_initializer: Utils.from_hash(hash[:weight_initializer]),
                        bias_initializer: Utils.from_hash(hash[:bias_initializer]),
                        l1_lambda: hash[:l1_lambda],
                        l2_lambda: hash[:l2_lambda],
                        use_bias: hash[:use_bias])
        lstm
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: RandomNormal.new,
                     bias_initializer: Zeros.new,
                     l1_lambda: 0,
                     l2_lambda: 0,
                     use_bias: true)
        super
        @cell = @params[:c] = Param.new
      end

      def build(input_shape)
        super
        num_prev_nodes = input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes * 4)
        @weight_initializer.init_param(self, @weight)
        @weight2.data = Xumo::SFloat.new(@num_nodes, @num_nodes * 4)
        @weight_initializer.init_param(self, @weight2)
        if @bias
          @bias.data = Xumo::SFloat.new(@num_nodes * 4) 
          @bias_initializer.init_param(self, @bias) 
        end
        @time_length.times do |t|
          @layers << LSTM_Dense.new(@weight, @weight2, @bias)
        end
      end

      def forward(xs)
        @xs_shape = xs.shape
        hs = Xumo::SFloat.zeros(xs.shape[0], @time_length, @num_nodes)
        h = nil
        c = nil
        if @stateful
          h = @hidden.data if @hidden.data
          c = @cell.data if @cell.data
        end
        h ||= Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        c ||= Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        xs.shape[1].times do |t|
          x = xs[true, t, false]
          h, c = @layers[t].forward(x, h, c)
          hs[true, t, false] = h
        end
        @hidden.data = h
        @cell.data = c
        @return_sequences ? hs : h
      end

      def backward(dh2s)
        @weight.grad += Xumo::SFloat.zeros(*@weight.data.shape)
        @weight2.grad += Xumo::SFloat.zeros(*@weight2.data.shape)
        @bias.grad += Xumo::SFloat.zeros(*@bias.data.shape) if @bias
        unless @return_sequences
          dh = dh2s
          dh2s = Xumo::SFloat.zeros(dh.shape[0], @time_length, dh.shape[1])
          dh2s[true, -1, false] = dh
        end
        dxs = Xumo::SFloat.zeros(@xs_shape)
        dh = 0
        dc = 0
        (0...dh2s.shape[1]).to_a.reverse.each do |t|
          dh2 = dh2s[true, t, false]
          dx, dh, dc = @layers[t].backward(dh2 + dh, dc)
          dxs[true, t, false] = dx
        end
        dxs
      end

      def reset_state
        super()
        @cell.data = @cell.data.fill(0) if @cell.data
      end
    end


    class GRU_Dense
      def initialize(weight, weight2, bias)
        @weight = weight
        @weight2 = weight2
        @bias = bias
        @update_sigmoid = Sigmoid.new
        @reset_sigmoid = Sigmoid.new
        @tanh = Tanh.new
      end

      def forward(x, h)
        @x = x
        @h = h
        num_nodes = h.shape[1]
        @weight_a = @weight.data[true, 0...(num_nodes * 2)]
        @weight2_a = @weight2.data[true, 0...(num_nodes * 2)]
        a = x.dot(@weight_a) + h.dot(@weight2_a)
        a += @bias.data[0...(num_nodes * 2)] if @bias
        @update = @update_sigmoid.forward(a[true, 0...num_nodes])
        @reset = @reset_sigmoid.forward(a[true, num_nodes..-1])

        @weight_h = @weight.data[true, (num_nodes * 2)..-1]
        @weight2_h = @weight2.data[true, (num_nodes * 2)..-1]
        @tanh_h = if @bias
          bias_h = @bias.data[(num_nodes * 2)..-1]
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

        dweight_h = @x.transpose.dot(dtanh_h)
        dx = dtanh_h.dot(@weight_h.transpose)
        dweight2_h = (@h * @reset).transpose.dot(dtanh_h)
        dh += dtanh_h.dot(@weight2_h.transpose) * @reset
        dbias_h = dtanh_h.sum(0) if @bias

        dreset = @reset_sigmoid.backward(dtanh_h.dot(@weight2_h.transpose) * @h)
        dupdate = @update_sigmoid.backward(dh2 * @h - dh2 * @tanh_h)
        da = Xumo::SFloat.hstack([dupdate, dreset])
        dweight_a = @x.transpose.dot(da)
        dx += da.dot(@weight_a.transpose)
        dweight2_a = @h.transpose.dot(da)
        dh += da.dot(@weight2_a.transpose)
        dbias_a = da.sum(0) if @bias

        @weight.grad += Xumo::SFloat.hstack([dweight_a, dweight_h])
        @weight2.grad += Xumo::SFloat.hstack([dweight2_a, dweight2_h])
        @bias.grad += Xumo::SFloat.hstack([dbias_a, dbias_h]) if @bias
        [dx, dh]
      end
    end


    class GRU < RNN
      def self.from_hash(hash)
        gru = self.new(hash[:num_nodes],
                       stateful: hash[:stateful],
                       return_sequences: hash[:return_sequences],
                       weight_initializer: Utils.from_hash(hash[:weight_initializer]),
                       bias_initializer: Utils.from_hash(hash[:bias_initializer]),
                       l1_lambda: hash[:l1_lambda],
                       l2_lambda: hash[:l2_lambda],
                       use_bias: hash[:use_bias])
        gru
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: RandomNormal.new,
                     bias_initializer: Zeros.new,
                     l1_lambda: 0,
                     l2_lambda: 0,
                     use_bias: true)
        super
      end
    
      def build(input_shape)
        super
        num_prev_nodes = @input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes * 3)
        @weight_initializer.init_param(self, @weight)
        @weight2.data = Xumo::SFloat.new(@num_nodes, @num_nodes * 3)
        @weight_initializer.init_param(self, @weight2)
        if @bias
          @bias.data = Xumo::SFloat.new(@num_nodes * 3)
          @bias_initializer.init_param(self, @bias) 
        end
        @time_length.times do |t|
          @layers << GRU_Dense.new(@weight, @weight2, @bias)
        end
      end
    end

  end
end
