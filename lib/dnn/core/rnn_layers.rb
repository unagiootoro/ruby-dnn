module DNN
  module Layers

    # Super class of all RNN classes.
    class RNN < Connection      
      attr_reader :num_nodes
      # @return [DNN::Param] Recurrent weight parameter.
      attr_reader :recurrent_weight
      # @return [DNN::Param] Hidden parameter that Stateful RNN has.
      attr_reader :hidden
      attr_reader :stateful
      attr_reader :return_sequences
      attr_reader :recurrent_weight_initializer
      attr_reader :recurrent_weight_regularizer

      # @param [Integer] num_nodes Number of nodes.
      # @param [Boolean] stateful maintain state between batches.
      # @param [Boolean] return_sequences Set the false, only the last of each cell of RNN is left.
      # @return [DNN::Initializers::Initializer] recurrent_weight_initializer Recurrent weight initializer.
      # @return [DNN::Regularizers::Regularizer] recurrent_weight_regularizer Recurrent weight regularization.
      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: Initializers::RandomNormal.new,
                     recurrent_weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     recurrent_weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              weight_regularizer: weight_regularizer, bias_regularizer: bias_regularizer, use_bias: use_bias)
        @num_nodes = num_nodes
        @stateful = stateful
        @return_sequences = return_sequences
        @layers = []
        @hidden = Param.new
        @recurrent_weight = Param.new(nil, 0)
        @recurrent_weight_initializer = recurrent_weight_initializer
        @recurrent_weight_regularizer = recurrent_weight_regularizer
      end

      def build(input_shape)
        unless input_shape.length == 2
          raise DNN_ShapeError.new("Input shape is #{input_shape}. But input shape must be 2 dimensional.")
        end
        super
        @time_length = @input_shape[0]
      end

      def forward(xs)
        @xs_shape = xs.shape
        hs = Xumo::SFloat.zeros(xs.shape[0], @time_length, @num_nodes)
        h = (@stateful && @hidden.data) ? @hidden.data : Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        xs.shape[1].times do |t|
          x = xs[true, t, false]
          @layers[t].trainable = @trainable
          h = @layers[t].forward(x, h)
          hs[true, t, false] = h
        end
        @hidden.data = h
        @return_sequences ? hs : h
      end

      def backward(dh2s)
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
          return_sequences: @return_sequences,
          recurrent_weight_initializer: @recurrent_weight_initializer.to_hash,
          recurrent_weight_regularizer: @recurrent_weight_regularizer&.to_hash,
        }
        hash.merge!(merge_hash) if merge_hash
        super(hash)
      end

      def get_params
        {weight: @weight, recurrent_weight: @recurrent_weight, bias: @bias, hidden: @hidden}
      end

      # Reset the state of RNN.
      def reset_state
        @hidden.data = @hidden.data.fill(0) if @hidden.data
      end

      def regularizers
        regularizers = []
        regularizers << @weight_regularizer if @weight_regularizer
        regularizers << @recurrent_weight_regularizer if @recurrent_weight_regularizer
        regularizers << @bias_regularizer if @bias_regularizer
        regularizers
      end

      private def init_weight_and_bias
        super
        @recurrent_weight_initializer.init_param(self, @recurrent_weight)
        @recurrent_weight_regularizer.param = @recurrent_weight if @recurrent_weight_regularizer
      end
    end


    class SimpleRNN_Dense
      attr_accessor :trainable

      def initialize(weight, recurrent_weight, bias, activation)
        @weight = weight
        @recurrent_weight = recurrent_weight
        @bias = bias
        @activation = activation.clone
        @trainable = true
      end

      def forward(x, h)
        @x = x
        @h = h
        h2 = x.dot(@weight.data) + h.dot(@recurrent_weight.data)
        h2 += @bias.data if @bias
        @activation.forward(h2)
      end

      def backward(dh2)
        dh2 = @activation.backward(dh2)
        if @trainable
          @weight.grad += @x.transpose.dot(dh2)
          @recurrent_weight.grad += @h.transpose.dot(dh2)
          @bias.grad += dh2.sum(0) if @bias
        end
        dx = dh2.dot(@weight.data.transpose)
        dh = dh2.dot(@recurrent_weight.data.transpose)
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
                              recurrent_weight_initializer: Utils.from_hash(hash[:recurrent_weight_initializer]),
                              bias_initializer: Utils.from_hash(hash[:bias_initializer]),
                              weight_regularizer: Utils.from_hash(hash[:weight_regularizer]),
                              recurrent_weight_regularizer: Utils.from_hash(hash[:recurrent_weight_regularizer]),
                              bias_regularizer: Utils.from_hash(hash[:bias_regularizer]),
                              use_bias: hash[:use_bias])
        simple_rnn
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     activation: Tanh.new,
                     weight_initializer: Initializers::RandomNormal.new,
                     recurrent_weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     recurrent_weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true)
        super(num_nodes,
              stateful: stateful,
              return_sequences: return_sequences,
              weight_initializer: weight_initializer,
              recurrent_weight_initializer: recurrent_weight_initializer,
              bias_initializer: bias_initializer,
              weight_regularizer: weight_regularizer,
              recurrent_weight_regularizer: recurrent_weight_regularizer,
              bias_regularizer: bias_regularizer,
              use_bias: use_bias)
        @activation = activation
      end

      def build(input_shape)
        super
        num_prev_nodes = input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes)
        @recurrent_weight.data = Xumo::SFloat.new(@num_nodes, @num_nodes)
        @bias.data = Xumo::SFloat.new(@num_nodes) if @bias
        init_weight_and_bias
        @time_length.times do |t|
          @layers << SimpleRNN_Dense.new(@weight, @recurrent_weight, @bias, @activation)
        end
      end

      def to_hash
        super({activation: @activation.to_hash})
      end
    end


    class LSTM_Dense
      attr_accessor :trainable

      def initialize(weight, recurrent_weight, bias)
        @weight = weight
        @recurrent_weight = recurrent_weight
        @bias = bias
        @tanh = Tanh.new
        @g_tanh = Tanh.new
        @forget_sigmoid = Sigmoid.new
        @in_sigmoid = Sigmoid.new
        @out_sigmoid = Sigmoid.new
        @trainable = true
      end

      def forward(x, h, c)
        @x = x
        @h = h
        @c = c
        num_nodes = h.shape[1]
        a = x.dot(@weight.data) + h.dot(@recurrent_weight.data)
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

        if @trainable
          @weight.grad += @x.transpose.dot(da)
          @recurrent_weight.grad += @h.transpose.dot(da)
          @bias.grad += da.sum(0) if @bias
        end
        dx = da.dot(@weight.data.transpose)
        dh = da.dot(@recurrent_weight.data.transpose)
        dc = dc2_tmp * @forget
        [dx, dh, dc]
      end
    end


    class LSTM < RNN
      # @return [DNN::Param] Hidden parameter that Stateful RNN has.
      attr_reader :cell

      def self.from_hash(hash)
        lstm = self.new(hash[:num_nodes],
                        stateful: hash[:stateful],
                        return_sequences: hash[:return_sequences],
                        weight_initializer: Utils.from_hash(hash[:weight_initializer]),
                        recurrent_weight_initializer: Utils.from_hash(hash[:recurrent_weight_initializer]),
                        bias_initializer: Utils.from_hash(hash[:bias_initializer]),
                        weight_regularizer: Utils.from_hash(hash[:weight_regularizer]),
                        recurrent_weight_regularizer: Utils.from_hash(hash[:recurrent_weight_regularizer]),
                        bias_regularizer: Utils.from_hash(hash[:bias_regularizer]),
                        use_bias: hash[:use_bias])
        lstm
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: Initializers::RandomNormal.new,
                     recurrent_weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     recurrent_weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true)
        super
        @cell = Param.new
      end

      def build(input_shape)
        super
        num_prev_nodes = input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes * 4)
        @recurrent_weight.data = Xumo::SFloat.new(@num_nodes, @num_nodes * 4)
        @bias.data = Xumo::SFloat.new(@num_nodes * 4) if @bias
        init_weight_and_bias
        @time_length.times do |t|
          @layers << LSTM_Dense.new(@weight, @recurrent_weight, @bias)
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
          @layers[t].trainable = @trainable
          h, c = @layers[t].forward(x, h, c)
          hs[true, t, false] = h
        end
        @hidden.data = h
        @cell.data = c
        @return_sequences ? hs : h
      end

      def backward(dh2s)
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

      def get_params
        {weight: @weight, recurrent_weight: @recurrent_weight, bias: @bias, hidden: @hidden, cell: @cell}
      end
    end


    class GRU_Dense
      attr_accessor :trainable

      def initialize(weight, recurrent_weight, bias)
        @weight = weight
        @recurrent_weight = recurrent_weight
        @bias = bias
        @update_sigmoid = Sigmoid.new
        @reset_sigmoid = Sigmoid.new
        @tanh = Tanh.new
        @trainable = true
      end

      def forward(x, h)
        @x = x
        @h = h
        num_nodes = h.shape[1]
        @weight_a = @weight.data[true, 0...(num_nodes * 2)]
        @weight2_a = @recurrent_weight.data[true, 0...(num_nodes * 2)]
        a = x.dot(@weight_a) + h.dot(@weight2_a)
        a += @bias.data[0...(num_nodes * 2)] if @bias
        @update = @update_sigmoid.forward(a[true, 0...num_nodes])
        @reset = @reset_sigmoid.forward(a[true, num_nodes..-1])

        @weight_h = @weight.data[true, (num_nodes * 2)..-1]
        @weight2_h = @recurrent_weight.data[true, (num_nodes * 2)..-1]
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

        if @trainable
          dweight_h = @x.transpose.dot(dtanh_h)
          dweight2_h = (@h * @reset).transpose.dot(dtanh_h)
          dbias_h = dtanh_h.sum(0) if @bias
        end
        dx = dtanh_h.dot(@weight_h.transpose)
        dh += dtanh_h.dot(@weight2_h.transpose) * @reset

        dreset = @reset_sigmoid.backward(dtanh_h.dot(@weight2_h.transpose) * @h)
        dupdate = @update_sigmoid.backward(dh2 * @h - dh2 * @tanh_h)
        da = Xumo::SFloat.hstack([dupdate, dreset])
        if @trainable
          dweight_a = @x.transpose.dot(da)
          dweight2_a = @h.transpose.dot(da)
          dbias_a = da.sum(0) if @bias
        end
        dx += da.dot(@weight_a.transpose)
        dh += da.dot(@weight2_a.transpose)

        if @trainable
          @weight.grad += Xumo::SFloat.hstack([dweight_a, dweight_h])
          @recurrent_weight.grad += Xumo::SFloat.hstack([dweight2_a, dweight2_h])
          @bias.grad += Xumo::SFloat.hstack([dbias_a, dbias_h]) if @bias
        end
        [dx, dh]
      end
    end


    class GRU < RNN
      def self.from_hash(hash)
        gru = self.new(hash[:num_nodes],
                       stateful: hash[:stateful],
                       return_sequences: hash[:return_sequences],
                       weight_initializer: Utils.from_hash(hash[:weight_initializer]),
                       recurrent_weight_initializer: Utils.from_hash(hash[:recurrent_weight_initializer]),
                       bias_initializer: Utils.from_hash(hash[:bias_initializer]),
                       weight_regularizer: Utils.from_hash(hash[:weight_regularizer]),
                       recurrent_weight_regularizer: Utils.from_hash(hash[:recurrent_weight_regularizer]),
                       bias_regularizer: Utils.from_hash(hash[:bias_regularizer]),
                       use_bias: hash[:use_bias])
        gru
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: Initializers::RandomNormal.new,
                     recurrent_weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     recurrent_weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true)
        super
      end
    
      def build(input_shape)
        super
        num_prev_nodes = @input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_nodes, @num_nodes * 3)
        @recurrent_weight.data = Xumo::SFloat.new(@num_nodes, @num_nodes * 3)
        @bias.data = Xumo::SFloat.new(@num_nodes * 3) if @bias
        init_weight_and_bias
        @time_length.times do |t|
          @layers << GRU_Dense.new(@weight, @recurrent_weight, @bias)
        end
      end
    end

  end
end
