module DNN
  module Layers
    # Super class of all RNN classes.
    class RNN < Connection
      attr_reader :num_units
      attr_reader :recurrent_weight
      attr_reader :hidden
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
        @hidden = Param.new
        @recurrent_weight = Param.new(nil, Xumo::SFloat[0])
        @recurrent_weight_initializer = recurrent_weight_initializer
        @recurrent_weight_regularizer = recurrent_weight_regularizer
      end

      def build(input_shape)
        unless input_shape.length == 2
          raise DNNShapeError, "Input shape is #{input_shape}. But input shape must be 2 dimensional."
        end
        super
      end

      private def create_hidden_layer
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'create_hidden_layer'"
      end

      def compute_output_shape
        @time_length = @input_shape[0]
        @return_sequences ? [@time_length, @num_units] : [@num_units]
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

      def get_params
        { weight: @weight, recurrent_weight: @recurrent_weight, bias: @bias, hidden: @hidden }
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
        h_array = [] if @return_sequences
        x_array = Functions::RNNTimeSplit.new.(xs)
        h = Tensor.new(Xumo::SFloat.zeros(xs.shape[0], @num_units))
        x_array.each.with_index do |x, t|
          h = Functions::SimpleRNNCell.new.(x, h, @weight, @recurrent_weight, @bias)
          h = Functions::FunctionSpace.send(@activation, h)
          h_array << h if @return_sequences
        end
        @return_sequences ? Functions::RNNTimeConcatenate.new.(*h_array) : h
      end

      def to_hash
        super(activation: @activation.to_hash)
      end

      def load_hash(hash)
        initialize(hash[:num_units],
                   stateful: hash[:stateful],
                   return_sequences: hash[:return_sequences],
                   activation: Layers::Layer.from_hash(hash[:activation]),
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
        @cell = Param.new
      end

      def build(input_shape)
        super
        num_prev_units = input_shape[1]
        @weight.data = Xumo::SFloat.new(num_prev_units, @num_units * 4)
        @recurrent_weight.data = Xumo::SFloat.new(@num_units, @num_units * 4)
        @bias.data = Xumo::SFloat.new(@num_units * 4) if @bias
        init_weight_and_bias
      end

      def forward(xs)
        h_array = [] if @return_sequences
        x_array = Functions::RNNTimeSplit.new.(xs)
        h = Tensor.new(Xumo::SFloat.zeros(xs.shape[0], @num_units))
        c = Tensor.new(Xumo::SFloat.zeros(xs.shape[0], @num_units))
        x_array.each.with_index do |x, t|
          if t == x_array.length - 1
            h = Functions::LSTMCell.new(return_c: false).(x, h, c, @weight, @recurrent_weight, @bias)
          else
            h, c = Functions::LSTMCell.new.(x, h, c, @weight, @recurrent_weight, @bias)
          end
          h_array << h if @return_sequences
        end
        @return_sequences ? Functions::RNNTimeConcatenate.new.(*h_array) : h
      end

      def reset_state
        super()
        @cell.data = @cell.data.fill(0) if @cell.data
      end

      def get_params
        { weight: @weight, recurrent_weight: @recurrent_weight, bias: @bias, hidden: @hidden, cell: @cell }
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
        h_array = [] if @return_sequences
        x_array = Functions::RNNTimeSplit.new.(xs)
        h = Tensor.new(Xumo::SFloat.zeros(xs.shape[0], @num_units))
        x_array.each.with_index do |x, t|
          h = Functions::GRUCell.new.(x, h, @weight, @recurrent_weight, @bias)
          h_array << h if @return_sequences
        end
        @return_sequences ? Functions::RNNTimeConcatenate.new.(*h_array) : h
      end

    end

  end
end
