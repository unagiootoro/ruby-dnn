module DNN
  module Layers

    class SimpleRNN_Dense
      def initialize(params, grads, activation)
        @params = params
        @grads = grads
        @activation = activation
      end

      def forward(x, h)
        @x = x
        @h = h
        h2 = x.dot(@params[:weight]) + h.dot(@params[:weight2]) + @params[:bias]
        @activation.forward(h2)
      end

      def backward(dh2)
        dh2 = @activation.backward(dh2)
        @grads[:weight] += @x.transpose.dot(dh2)
        @grads[:weight2] += @h.transpose.dot(dh2)
        @grads[:bias] += dh2.sum(0)
        dx = dh2.dot(@params[:weight].transpose)
        dh = dh2.dot(@params[:weight2].transpose)
        [dx, dh]
      end
    end


    class SimpleRNN < HasParamLayer
      include Initializers
      include Activations

      attr_reader :num_nodes
      attr_reader :stateful
      attr_reader :weight_decay

      def self.load_hash(hash)
        self.new(hash[:num_nodes],
                 stateful: hash[:stateful],
                 return_sequences: hash[:return_sequences],
                 activation: Util.load_hash(hash[:activation]),
                 weight_initializer: Util.load_hash(hash[:weight_initializer]),
                 bias_initializer: Util.load_hash(hash[:bias_initializer]),
                 weight_decay: hash[:weight_decay])
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     activation: nil,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     weight_decay: 0)
        super()
        @num_nodes = num_nodes
        @stateful = stateful
        @return_sequences = return_sequences
        @activation = (activation || Tanh.new)
        @weight_initializer = (weight_initializer || RandomNormal.new)
        @bias_initializer = (bias_initializer || Zeros.new)
        @weight_decay = weight_decay
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

      def shape
        @return_sequences ? [@time_length, @num_nodes] : [@num_nodes]
      end

      def ridge
        if @weight_decay > 0
          0.5 * (@weight_decay * (@params[:weight]**2).sum + @weight_decay * (@params[:weight]**2).sum)
        else
          0
        end
      end

      def to_hash
        super({num_nodes: @num_nodes,
               stateful: @stateful,
               return_sequences: @return_sequences,
               activation: @activation.to_hash,
               weight_initializer: @weight_initializer.to_hash,
               bias_initializer: @bias_initializer.to_hash,
               weight_decay: @weight_decay})
      end

      private
    
      def init_params
        @time_length = prev_layer.shape[0]
        num_prev_nodes = prev_layer.shape[1]
        @params[:weight] = Xumo::SFloat.new(num_prev_nodes, @num_nodes)
        @params[:weight2] = Xumo::SFloat.new(@num_nodes, @num_nodes)
        @params[:bias] = Xumo::SFloat.new(@num_nodes)
        @weight_initializer.init_param(self, :weight)
        @weight_initializer.init_param(self, :weight2)
        @bias_initializer.init_param(self, :bias)
        @time_length.times do |t|
          @layers << SimpleRNN_Dense.new(@params, @grads, @activation.clone)
        end
      end
    end


    class LSTM_Dense
      def initialize(params, grads)
        @params = params
        @grads = grads
        @tanh = Tanh.new
        @g_tanh = Tanh.new
        @forget_sigmoid = Sigmoid.new
        @in_sigmoid = Sigmoid.new
        @out_sigmoid = Sigmoid.new
      end

      def forward(x, h, cell)
        @x = x
        @h = h
        @cell = cell
        num_nodes = h.shape[1]
        a = x.dot(@params[:weight]) + h.dot(@params[:weight2]) + @params[:bias]

        @forget = @forget_sigmoid.forward(a[true, 0...num_nodes])
        @g = @g_tanh.forward(a[true, num_nodes...(num_nodes * 2)])
        @in = @in_sigmoid.forward(a[true, (num_nodes * 2)...(num_nodes * 3)])
        @out = @out_sigmoid.forward(a[true, (num_nodes * 3)..-1])

        @cell2 = @forget * cell + @g * @in
        @tanh_cell2 = @tanh.forward(@cell2)
        @h2 = @out * @tanh_cell2
        [@h2, @cell2]
      end

      def backward(dh2, dcell2)
        dh2_tmp = @tanh_cell2 * dh2
        dcell2_tmp = @tanh.backward(@out * dh2) + dcell2

        dout = @out_sigmoid.backward(dh2_tmp)
        din = @in_sigmoid.backward(dcell2_tmp * @g)
        dg = @g_tanh.backward(dcell2_tmp * @in)
        dforget = @forget_sigmoid.backward(dcell2_tmp * @cell)

        da = Xumo::SFloat.hstack([dforget, dg, din, dout])

        @grads[:weight] += @x.transpose.dot(da)
        @grads[:weight2] += @h.transpose.dot(da)
        @grads[:bias] += da.sum(0)
        dx = da.dot(@params[:weight].transpose)
        dh = da.dot(@params[:weight2].transpose)
        dcell = dcell2_tmp * @forget
        [dx, dh, dcell]
      end
    end


    # In development
    class LSTM < HasParamLayer
      include Initializers
      include Activations

      attr_reader :num_nodes
      attr_reader :stateful
      attr_reader :weight_decay

      def self.load_hash(hash)
        self.new(hash[:num_nodes],
                 stateful: hash[:stateful],
                 return_sequences: hash[:return_sequences],
                 weight_initializer: Util.load_hash(hash[:weight_initializer]),
                 bias_initializer: Util.load_hash(hash[:bias_initializer]),
                 weight_decay: hash[:weight_decay])
      end

      def initialize(num_nodes,
                     stateful: false,
                     return_sequences: true,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     weight_decay: 0)
        super()
        @num_nodes = num_nodes
        @stateful = stateful
        @return_sequences = return_sequences
        @weight_initializer = (weight_initializer || RandomNormal.new)
        @bias_initializer = (bias_initializer || Zeros.new)
        @weight_decay = weight_decay
        @layers = []
        @h = nil
        @cell = nil
      end

      def forward(xs)
        @xs_shape = xs.shape
        hs = Xumo::SFloat.zeros(xs.shape[0], @time_length, @num_nodes)
        h = nil
        cell = nil
        if @stateful
          h = @h if @h
          cell = @cell if @cell
        end
        h ||= Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        cell ||= Xumo::SFloat.zeros(xs.shape[0], @num_nodes)
        xs.shape[1].times do |t|
          x = xs[true, t, false]
          h, cell = @layers[t].forward(x, h, cell)
          hs[true, t, false] = h
        end
        @h = h
        @cell = cell
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
        dcell = 0
        (0...dh2s.shape[1]).to_a.reverse.each do |t|
          dh2 = dh2s[true, t, false]
          dx, dh, dcell = @layers[t].backward(dh2 + dh, dcell)
          dxs[true, t, false] = dx
        end
        dxs
      end

      def shape
        @return_sequences ? [@time_length, @num_nodes] : [@num_nodes]
      end

      def ridge
        if @weight_decay > 0
          0.5 * (@weight_decay * (@params[:weight]**2).sum + @weight_decay * (@params[:weight]**2).sum)
        else
          0
        end
      end

      def to_hash
        super({num_nodes: @num_nodes,
               stateful: @stateful,
               return_sequences: @return_sequences,
               weight_initializer: @weight_initializer.to_hash,
               bias_initializer: @bias_initializer.to_hash,
               weight_decay: @weight_decay})
      end

      private
    
      def init_params
        @time_length = prev_layer.shape[0] 
        num_prev_nodes = prev_layer.shape[1]
        @params[:weight] = Xumo::SFloat.new(num_prev_nodes, @num_nodes * 4)
        @params[:weight2] = Xumo::SFloat.new(@num_nodes, @num_nodes * 4)
        @params[:bias] = Xumo::SFloat.new(@num_nodes * 4)
        @weight_initializer.init_param(self, :weight)
        @weight_initializer.init_param(self, :weight2)
        @bias_initializer.init_param(self, :bias)
        @time_length.times do |t|
          @layers << LSTM_Dense.new(@params, @grads)
        end
      end
    end

  end
end
