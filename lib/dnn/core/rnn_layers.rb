module DNN
  module Layers

    # Super class of all RNN classes.
    class RNN < Connection
      include Activations

      attr_accessor :h
      attr_reader :num_nodes
      attr_reader :stateful

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     l1_lambda: 0,
                     l2_lambda: 0)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              l1_lambda: l1_lambda, l2_lambda: l2_lambda)
        @num_nodes = num_nodes
        @stateful = stateful
        @return_sequences = return_sequences
        @layers = []
        @h = nil
      end

      def forward(xs)
        @xs_shape = xs.shape
        hs = Xumo::SFloat.zeros(xs.shape[0], @time_length, @num_nodes)
        h = (@stateful && @h) ? @h : Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        xs.shape[1].times do |t|
          x = xs[true, t, false]
          h = @layers[t].forward(x, h)
          hs[true, t, false] = h
        end
        @h = h
        @return_sequences ? hs : h
      end

      def backward(dh2s)
        @grads[:weight] = Xumo::SFloat.zeros(*@params[:weight].shape)
        @grads[:weight2] = Xumo::SFloat.zeros(*@params[:weight2].shape)
        @grads[:bias] = Xumo::SFloat.zeros(*@params[:bias].shape)
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

      def to_hash(merge_hash = nil)
        hash = {
          num_nodes: @num_nodes,
          stateful: @stateful,
          return_sequences: @return_sequences,
          h: @h.to_a
        }
        hash.merge!(merge_hash) if merge_hash
        super(hash)
      end

      def shape
        @return_sequences ? [@time_length, @num_nodes] : [@num_nodes]
      end

      def reset_state
        @h = @h.fill(0) if @h
      end

      def lasso
        if @l1_lambda > 0
          @l1_lambda * (@params[:weight].abs.sum + @params[:weight2].abs.sum)
        else
          0
        end
      end

      def ridge
        if @l2_lambda > 0
          0.5 * (@l2_lambda * ((@params[:weight]**2).sum + (@params[:weight2]**2).sum))
        else
          0
        end
      end

      def dlasso
        dlasso = Xumo::SFloat.ones(*@params[:weight].shape)
        dlasso[@params[:weight] < 0] = -1
        @l1_lambda * dlasso
      end

      def dridge
        @l2_lambda * @params[:weight]
      end

      def dlasso2
        dlasso = Xumo::SFloat.ones(*@params[:weight2].shape)
        dlasso[@params[:weight2] < 0] = -1
        @l1_lambda * dlasso
      end

      def dridge2
        @l2_lambda * @params[:weight2]
      end

      private

      def init_params
        @time_length = prev_layer.shape[0]
      end
    end


    class SimpleRNN_Dense
      def initialize(rnn)
        @rnn = rnn
        @activation = rnn.activation.clone
      end

      def forward(x, h)
        @x = x
        @h = h
        h2 = x.dot(@rnn.params[:weight]) + h.dot(@rnn.params[:weight2]) + @rnn.params[:bias]
        @activation.forward(h2)
      end

      def backward(dh2)
        dh2 = @activation.backward(dh2)
        @rnn.grads[:weight] += @x.transpose.dot(dh2)
        @rnn.grads[:weight2] += @h.transpose.dot(dh2)
        if @rnn.l1_lambda > 0
          @rnn.grads[:weight] += dlasso
          @rnn.grads[:weight2] += dlasso2
        elsif @rnn.l2_lambda > 0
          @rnn.grads[:weight] += dridge
          @grads[:weight2] += dridge2
        end
        @rnn.grads[:bias] += dh2.sum(0)
        dx = dh2.dot(@rnn.params[:weight].transpose)
        dh = dh2.dot(@rnn.params[:weight2].transpose)
        [dx, dh]
      end
    end


    class SimpleRNN < RNN
      def self.load_hash(hash)
        simple_rnn = self.new(hash[:num_nodes],
                              stateful: hash[:stateful],
                              return_sequences: hash[:return_sequences],
                              activation: Util.load_hash(hash[:activation]),
                              weight_initializer: Util.load_hash(hash[:weight_initializer]),
                              bias_initializer: Util.load_hash(hash[:bias_initializer]),
                              l1_lambda: hash[:l1_lambda],
                              l2_lambda: hash[:l2_lambda])
        simple_rnn.h = Xumo::SFloat.cast(hash[:h])
        simple_rnn
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     activation: nil,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     l1_lambda: 0,
                     l2_lambda: 0)
        super(num_nodes,
              stateful: stateful,
              return_sequences: return_sequences,
              weight_initializer: weight_initializer,
              bias_initializer: bias_initializer,
              l1_lambda: 0,
              l2_lambda: 0)
        @activation = (activation || Tanh.new)
      end

      def to_hash
        super({activation: @activation.to_hash})
      end

      private
    
      def init_params
        super()
        num_prev_nodes = prev_layer.shape[1]
        @params[:weight] = Xumo::SFloat.new(num_prev_nodes, @num_nodes)
        @params[:weight2] = Xumo::SFloat.new(@num_nodes, @num_nodes)
        @params[:bias] = Xumo::SFloat.new(@num_nodes)
        @weight_initializer.init_param(self, :weight)
        @weight_initializer.init_param(self, :weight2)
        @bias_initializer.init_param(self, :bias)
        @time_length.times do |t|
          @layers << SimpleRNN_Dense.new(self)
        end
      end
    end


    class LSTM_Dense
      def initialize(rnn)
        @rnn = rnn
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
        a = x.dot(@rnn.params[:weight]) + h.dot(@rnn.params[:weight2]) + @rnn.params[:bias]

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

        @rnn.grads[:weight] += @x.transpose.dot(da)
        @rnn.grads[:weight2] += @h.transpose.dot(da)
        if @rnn.l1_lambda > 0
          @rnn.grads[:weight] += dlasso
          @rnn.grads[:weight2] += dlasso2
        elsif @rnn.l2_lambda > 0
          @rnn.grads[:weight] += dridge
          @rnn.grads[:weight2] += dridge2
        end
        @rnn.grads[:bias] += da.sum(0)
        dx = da.dot(@rnn.params[:weight].transpose)
        dh = da.dot(@rnn.params[:weight2].transpose)
        dc = dc2_tmp * @forget
        [dx, dh, dc]
      end
    end


    class LSTM < RNN
      attr_accessor :c

      def self.load_hash(hash)
        lstm = self.new(hash[:num_nodes],
                        stateful: hash[:stateful],
                        return_sequences: hash[:return_sequences],
                        weight_initializer: Util.load_hash(hash[:weight_initializer]),
                        bias_initializer: Util.load_hash(hash[:bias_initializer]),
                        l1_lambda: hash[:l1_lambda],
                        l2_lambda: hash[:l2_lambda])
        lstm.h = Xumo::SFloat.cast(hash[:h])
        lstm.c = Xumo::SFloat.cast(hash[:c])
        lstm
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     l1_lambda: 0,
                     l2_lambda: 0)
        super
        @c = nil
      end

      def forward(xs)
        @xs_shape = xs.shape
        hs = Xumo::SFloat.zeros(xs.shape[0], @time_length, @num_nodes)
        h = nil
        c = nil
        if @stateful
          h = @h if @h
          c = @c if @c
        end
        h ||= Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        c ||= Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        xs.shape[1].times do |t|
          x = xs[true, t, false]
          h, c = @layers[t].forward(x, h, c)
          hs[true, t, false] = h
        end
        @h = h
        @c = c
        @return_sequences ? hs : h
      end

      def backward(dh2s)
        @grads[:weight] = Xumo::SFloat.zeros(*@params[:weight].shape)
        @grads[:weight2] = Xumo::SFloat.zeros(*@params[:weight2].shape)
        @grads[:bias] = Xumo::SFloat.zeros(*@params[:bias].shape)
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
        @c = @c.fill(0) if @c
      end

      def to_hash
        super({c: @c.to_a})
      end

      private
    
      def init_params
        super()
        num_prev_nodes = prev_layer.shape[1]
        @params[:weight] = Xumo::SFloat.new(num_prev_nodes, @num_nodes * 4)
        @params[:weight2] = Xumo::SFloat.new(@num_nodes, @num_nodes * 4)
        @params[:bias] = Xumo::SFloat.new(@num_nodes * 4)
        @weight_initializer.init_param(self, :weight)
        @weight_initializer.init_param(self, :weight2)
        @bias_initializer.init_param(self, :bias)
        @time_length.times do |t|
          @layers << LSTM_Dense.new(self)
        end
      end
    end


    class GRU_Dense
      def initialize(rnn)
        @rnn = rnn
        @update_sigmoid = Sigmoid.new
        @reset_sigmoid = Sigmoid.new
        @tanh = Tanh.new
      end

      def forward(x, h)
        @x = x
        @h = h
        num_nodes = h.shape[1]
        @weight_a = @rnn.params[:weight][true, 0...(num_nodes * 2)]
        @weight2_a = @rnn.params[:weight2][true, 0...(num_nodes * 2)]
        bias_a = @rnn.params[:bias][0...(num_nodes * 2)]
        a = x.dot(@weight_a) + h.dot(@weight2_a) + bias_a
        @update = @update_sigmoid.forward(a[true, 0...num_nodes])
        @reset = @reset_sigmoid.forward(a[true, num_nodes..-1])

        @weight_h = @rnn.params[:weight][true, (num_nodes * 2)..-1]
        @weight2_h = @rnn.params[:weight2][true, (num_nodes * 2)..-1]
        bias_h = @rnn.params[:bias][(num_nodes * 2)..-1]
        @tanh_h = @tanh.forward(x.dot(@weight_h) + (h * @reset).dot(@weight2_h) + bias_h)
        h2 = (1 - @update) * h + @update * @tanh_h
        h2
      end

      def backward(dh2)
        dtanh_h = @tanh.backward(dh2 * @update)
        dh = dh2 * (1 - @update)

        dweight_h = @x.transpose.dot(dtanh_h)
        dx = dtanh_h.dot(@weight_h.transpose)
        dweight2_h = (@h * @reset).transpose.dot(dtanh_h)
        dh += dtanh_h.dot(@weight2_h.transpose) * @reset
        dbias_h = dtanh_h.sum(0)

        dreset = @reset_sigmoid.backward(dtanh_h.dot(@weight2_h.transpose) * @h)
        dupdate = @update_sigmoid.backward(dh2 * @tanh_h - dh2 * @h)
        da = Xumo::SFloat.hstack([dupdate, dreset])
        dweight_a = @x.transpose.dot(da)
        dx += da.dot(@weight_a.transpose)
        dweight2_a = @h.transpose.dot(da)
        dh += da.dot(@weight2_a.transpose)
        dbias_a = da.sum(0)

        @rnn.grads[:weight] += Xumo::SFloat.hstack([dweight_a, dweight_h])
        @rnn.grads[:weight2] += Xumo::SFloat.hstack([dweight2_a, dweight2_h])
        if @rnn.l1_lambda > 0
          @rnn.grads[:weight] += dlasso
          @rnn.grads[:weight2] += dlasso2
        elsif @rnn.l2_lambda > 0
          @rnn.grads[:weight] += dridge
          @rnn.grads[:weight2] += dridge2
        end
        @rnn.grads[:bias] += Xumo::SFloat.hstack([dbias_a, dbias_h])
        [dx, dh]
      end
    end


    class GRU < RNN
      def self.load_hash(hash)
        gru = self.new(hash[:num_nodes],
                       stateful: hash[:stateful],
                       return_sequences: hash[:return_sequences],
                       weight_initializer: Util.load_hash(hash[:weight_initializer]),
                       bias_initializer: Util.load_hash(hash[:bias_initializer]),
                       l1_lambda: hash[:l1_lambda],
                       l2_lambda: hash[:l2_lambda])
        gru.h = Xumo::SFloat.cast(hash[:h])
        gru
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     l1_lambda: 0,
                     l2_lambda: 0)
        super
      end

      private
    
      def init_params
        super()
        num_prev_nodes = prev_layer.shape[1]
        @params[:weight] = Xumo::SFloat.new(num_prev_nodes, @num_nodes * 3)
        @params[:weight2] = Xumo::SFloat.new(@num_nodes, @num_nodes * 3)
        @params[:bias] = Xumo::SFloat.new(@num_nodes * 3)
        @weight_initializer.init_param(self, :weight)
        @weight_initializer.init_param(self, :weight2)
        @bias_initializer.init_param(self, :bias)
        @time_length.times do |t|
          @layers << GRU_Dense.new(self)
        end
      end
    end

  end
end
