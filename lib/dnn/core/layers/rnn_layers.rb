module DNN
  module Layers
    class SimpleRNNCell < Layer
      def forward(x, h, weight, recurrent_weight, bias = nil)
        h2 = x.dot(weight) + h.dot(recurrent_weight)
        h2 += bias if bias
        h2
      end
    end

    class LSTMCell < Layer
      def forward(x, h, c, weight, recurrent_weight, bias = nil)
        fs = Functions::FunctionSpace

        a = x.dot(weight) + h.dot(recurrent_weight)
        a += bias if bias

        a1, a2, a3, a4 = fs.split(a, 4, axis: 1)

        f = fs.sigmoid(a1)
        g = fs.tanh(a2)
        i = fs.sigmoid(a3)
        o = fs.sigmoid(a4)

        c2 = f * c + g * i
        h2 = o * fs.tanh(c2)
        [h2, c2]
      end
    end

    class GRUCell < Layer
      def forward(x, h, weight, recurrent_weight, bias = nil)
        fs = Functions::FunctionSpace

        num_units = h.shape[1]
        weight_a, weight_h = fs.split(weight, [num_units * 2], axis: 1)
        weight2_a, weight2_h = fs.split(recurrent_weight, [num_units * 2], axis: 1)
        bias_a, bias_h = fs.split(bias, [num_units * 2], axis: 0) if bias

        a = x.dot(weight_a) + h.dot(weight2_a)
        a += bias_a if bias
        a1, a2 = fs.split(a, [num_units], axis: 1)
        u = fs.sigmoid(a1)
        r = fs.sigmoid(a2)

        h2 = if bias
               fs.tanh(x.dot(weight_h) + (h * r).dot(weight2_h) + bias_h)
             else
               fs.tanh(x.dot(weight_h) + (h * r).dot(weight2_h))
             end
        h2 = (1 - u) * h2 + u * h
        h2
      end
    end

    # Super class of all RNN classes.
    class RNN < Connection
      attr_reader :num_units
      attr_reader :recurrent_weight
      attr_reader :stateful
      attr_reader :return_sequences
      attr_reader :recurrent_weight_initializer
      attr_reader :recurrent_weight_regularizer

      # @param [Integer] num_units Number of nodes.
      # @param [Boolean] stateful Maintain state between batches.
      # @param [Boolean] return_sequences Set the false, only the last of each cell of RNN is left.
      # @param [DNN::Initializers::Initializer] recurrent_weight_initializer Recurrent weight initializer.
      # @param [DNN::Regularizers::Regularizer | NilClass] recurrent_weight_regularizer Recurrent weight regularizer.
      def initialize(num_units,
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
        @num_units = num_units
        @stateful = stateful
        @return_sequences = return_sequences
        @recurrent_weight = Variable.new(nil, Xumo::SFloat[0])
        @recurrent_weight_initializer = recurrent_weight_initializer
        @recurrent_weight_regularizer = recurrent_weight_regularizer
      end

      def build(input_shape)
        unless input_shape.length == 2
          raise DNNShapeError, "Input shape is #{input_shape}. But input shape must be 2 dimensional."
        end
        super
      end

      def to_hash(merge_hash = nil)
        hash = {
          num_units: @num_units,
          stateful: @stateful,
          return_sequences: @return_sequences,
          recurrent_weight_initializer: @recurrent_weight_initializer.to_hash,
          recurrent_weight_regularizer: @recurrent_weight_regularizer&.to_hash,
        }
        hash.merge!(merge_hash) if merge_hash
        super(hash)
      end

      def load_hash(hash)
        initialize(hash[:num_units],
                   stateful: hash[:stateful],
                   return_sequences: hash[:return_sequences],
                   weight_initializer: Initializers::Initializer.from_hash(hash[:weight_initializer]),
                   recurrent_weight_initializer: Initializers::Initializer.from_hash(hash[:recurrent_weight_initializer]),
                   bias_initializer: Initializers::Initializer.from_hash(hash[:bias_initializer]),
                   weight_regularizer: Regularizers::Regularizer.from_hash(hash[:weight_regularizer]),
                   recurrent_weight_regularizer: Regularizers::Regularizer.from_hash(hash[:recurrent_weight_regularizer]),
                   bias_regularizer: Regularizers::Regularizer.from_hash(hash[:bias_regularizer]),
                   use_bias: hash[:use_bias])
      end

      def get_variables
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'get_variables'"
      end

      def get_trainable_variables
        { weight: @weight, recurrent_weight: @recurrent_weight, bias: @bias }
      end

      # Reset the state of RNN.
      def reset_state
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'reset_state'"
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
        @recurrent_weight_initializer.init_param(@recurrent_weight, @input_shapes)
        @recurrent_weight_regularizer.param = @recurrent_weight if @recurrent_weight_regularizer
      end
    end

    class SimpleRNN < RNN
      attr_reader :activation

      # @param [Symbol] activation Activation function to use in a recurrent network.
      def initialize(num_units,
                     stateful: false,
                     return_sequences: true,
                     activation: :tanh,
                     weight_initializer: Initializers::RandomNormal.new,
                     recurrent_weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     recurrent_weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true)
        super(num_units,
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
        num_prev_units = input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_units, @num_units)
        @recurrent_weight.data = Xumo::SFloat.new(@num_units, @num_units)
        @bias.data = Xumo::SFloat.new(@num_units) if @bias
        init_weight_and_bias
      end

      def forward(xs)
        fs = Functions::FunctionSpace
        h_array = [] if @return_sequences
        x_array = fs.split(xs, xs.shape[1], axis: 1).map do |x|
          x.reshape(x.shape[0], x.shape[2])
        end
        h = (@stateful && @h) ? @h : Tensor.new(Xumo::SFloat.zeros(xs.shape[0], @num_units))
        x_array.each.with_index do |x, t|
          h = SimpleRNNCell.new.(x, h, @weight, @recurrent_weight, @bias)
          h = Functions::FunctionSpace.send(@activation, h)
          h_array << h if @return_sequences
        end
        @h = Tensor.new(h.data) if @stateful
        if @return_sequences
          fs.concatenate(*h_array.map { |_h| _h.reshape(_h.shape[0], 1, _h.shape[1]) }, axis: 1)
        else
          h
        end
      end

      def reset_state
        @h = Tensor.new(Xumo::SFloat.zeros(*@h.shape)) if @h
      end

      def get_variables
        { weight: @weight, recurrent_weight: @recurrent_weight, bias: @bias, h: @h }
      end

      def to_hash
        super(activation: @activation)
      end

      def load_hash(hash)
        initialize(hash[:num_units],
                   stateful: hash[:stateful],
                   return_sequences: hash[:return_sequences],
                   activation: hash[:activation],
                   weight_initializer: Initializers::Initializer.from_hash(hash[:weight_initializer]),
                   recurrent_weight_initializer: Initializers::Initializer.from_hash(hash[:recurrent_weight_initializer]),
                   bias_initializer: Initializers::Initializer.from_hash(hash[:bias_initializer]),
                   weight_regularizer: Regularizers::Regularizer.from_hash(hash[:weight_regularizer]),
                   recurrent_weight_regularizer: Regularizers::Regularizer.from_hash(hash[:recurrent_weight_regularizer]),
                   bias_regularizer: Regularizers::Regularizer.from_hash(hash[:bias_regularizer]),
                   use_bias: hash[:use_bias])
      end
    end

    class LSTM < RNN
      attr_reader :cell

      def build(input_shape)
        super
        num_prev_units = input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_units, @num_units * 4)
        @recurrent_weight.data = Xumo::SFloat.new(@num_units, @num_units * 4)
        @bias.data = Xumo::SFloat.new(@num_units * 4) if @bias
        init_weight_and_bias
      end

      def forward(xs)
        fs = Functions::FunctionSpace
        h_array = [] if @return_sequences
        x_array = fs.split(xs, xs.shape[1], axis: 1).map do |x|
          x.reshape(x.shape[0], x.shape[2])
        end
        h = (@stateful && @h) ? @h : Tensor.new(Xumo::SFloat.zeros(xs.shape[0], @num_units))
        c = (@stateful && @c) ? @c : Tensor.new(Xumo::SFloat.zeros(xs.shape[0], @num_units))
        x_array.each.with_index do |x, t|
          h, c = LSTMCell.new.(x, h, c, @weight, @recurrent_weight, @bias)
          h_array << h if @return_sequences
        end
        if @stateful
          @h = Tensor.new(h.data)
          @c = Tensor.new(c.data)
        end
        if @return_sequences
          fs.concatenate(*h_array.map { |_h| _h.reshape(_h.shape[0], 1, _h.shape[1]) }, axis: 1)
        else
          h
        end
      end

      def reset_state
        @h = Tensor.new(Xumo::SFloat.zeros(*@h.shape)) if @h
        @c = Tensor.new(Xumo::SFloat.zeros(*@c.shape)) if @c
      end

      def get_variables
        { weight: @weight, recurrent_weight: @recurrent_weight, bias: @bias, h: @h, c: @c }
      end
    end

    class GRU < RNN
      def initialize(num_units,
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
        num_prev_units = input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_units, @num_units * 3)
        @recurrent_weight.data = Xumo::SFloat.new(@num_units, @num_units * 3)
        @bias.data = Xumo::SFloat.new(@num_units * 3) if @bias
        init_weight_and_bias
      end

      def forward(xs)
        fs = Functions::FunctionSpace
        h_array = [] if @return_sequences
        x_array = fs.split(xs, xs.shape[1], axis: 1).map do |x|
          x.reshape(x.shape[0], x.shape[2])
        end
        h = (@stateful && @h) ? @h : Tensor.new(Xumo::SFloat.zeros(xs.shape[0], @num_units))
        x_array.each.with_index do |x, t|
          h = GRUCell.new.(x, h, @weight, @recurrent_weight, @bias)
          h_array << h if @return_sequences
        end
        @h = Tensor.new(h.data) if @stateful
        if @return_sequences
          fs.concatenate(*h_array.map { |_h| _h.reshape(_h.shape[0], 1, _h.shape[1]) }, axis: 1)
        else
          h
        end
      end

      def reset_state
        @h = Tensor.new(Xumo::SFloat.zeros(*@h.shape)) if @h
      end

      def get_variables
        { weight: @weight, recurrent_weight: @recurrent_weight, bias: @bias, h: @h }
      end

    end

  end
end
