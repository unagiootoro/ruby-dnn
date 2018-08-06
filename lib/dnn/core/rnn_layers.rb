module DNN
  module Layers

    class SimpleRNN < HasParamLayer
      include Initializers
      include Activations

      attr_reader :num_nodes
      attr_reader :stateful
      attr_reader :weight_decay

      def self.load_hash(hash)
        self.new(hash[:num_nodes],
                 stateful: hash[:stateful],
                 activation: Util.load_hash(hash[:activation]),
                 weight_initializer: Util.load_hash(hash[:weight_initializer]),
                 bias_initializer: Util.load_hash(hash[:bias_initializer]),
                 weight_decay: hash[:weight_decay])
      end

      def initialize(num_nodes,
                     stateful: false,
                     activation: nil,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     weight_decay: 0)
        super()
        @num_nodes = num_nodes
        @stateful = stateful
        @activation = (activation || Tanh.new)
        @weight_initializer = (weight_initializer || RandomNormal.new)
        @bias_initializer = (bias_initializer || Zeros.new)
        @weight_decay = weight_decay
        @h = nil
      end

      def forward(xs)
        @xs = xs
        @hs = SFloat.zeros(xs.shape[0], *shape)
        h = (@stateful && @h) ? @h : SFloat.zeros(xs.shape[0], @num_nodes)
        xs.shape[1].times do |t|
          x = xs[true, t, false]
          h = x.dot(@params[:weight]) + h.dot(@params[:weight2]) + @params[:bias]
          h = @activation.forward(h)
          @hs[true, t, false] = h
        end
        @h = h
        @hs
      end

      def backward(douts)
        @grads[:weight] = SFloat.zeros(*@params[:weight].shape)
        @grads[:weight2] = SFloat.zeros(*@params[:weight2].shape)
        dxs = SFloat.zeros(@xs.shape)
        (0...douts.shape[1]).to_a.reverse.each do |t|
          dout = douts[true, t, false]
          x = @xs[true, t, false]
          h = @hs[true, t, false]
          dout = @activation.backward(dout)
          @grads[:weight] += x.transpose.dot(dout)
          @grads[:weight2] += h.transpose.dot(dout)
          dxs[true, t, false] = dout.dot(@params[:weight].transpose)
        end
        @grads[:bias] = douts.sum(0).sum(0)
        dxs
      end

      def shape
        [@time_length, @num_nodes]
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
               activation: @activation.to_hash,
               weight_initializer: @weight_initializer.to_hash,
               bias_initializer: @bias_initializer.to_hash,
               weight_decay: @weight_decay})
      end

      private
    
      def init_params
        @time_length = prev_layer.shape[0] 
        num_prev_nodes = prev_layer.shape[1]
        @params[:weight] = SFloat.new(num_prev_nodes, @num_nodes)
        @params[:weight2] = SFloat.new(@num_nodes, @num_nodes)
        @params[:bias] = SFloat.new(@num_nodes)
        @weight_initializer.init_param(self, :weight)
        @weight_initializer.init_param(self, :weight2)
        @bias_initializer.init_param(self, :bias)
      end
    end

  end
end
