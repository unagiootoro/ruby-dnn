module DNN
  module Layers

    class Embedding < HasParamLayer
      attr_reader :input_length
      attr_reader :weight
      attr_reader :weight_initializer
      attr_reader :weight_regularizer

      def self.from_hash(hash)
        self.new(hash[:input_shape], hash[:input_length],
                 weight_initializer: DNN::Utils.hash_to_obj(hash[:weight_initializer]),
                 weight_regularizer: DNN::Utils.hash_to_obj(hash[:weight_regularizer]))
      end

      # @param [Integer | Array] input_dim_or_shape Set input data dimension or shape.
      # @param [Integer] input_length Set the time series length of input data.
      # @param [DNN::Initializers::Initializer] weight_initializer Weight initializer.
      # @param [DNN::Regularizers::Regularizer | NilClass] weight_regularizer Weight regularizer.
      def initialize(input_dim_or_shape, input_length,
                     weight_initializer: Initializers::RandomUniform.new,
                     weight_regularizer: nil)
        super()
        @input_shape = input_dim_or_shape.is_a?(Array) ? input_dim_or_shape : [input_dim_or_shape]
        @input_length = input_length
        @weight_initializer = weight_initializer
        @weight_regularizer = weight_regularizer
      end

      def call(input)
        x, *, learning_phase = *input
        build unless built?
        [forward(x), Link.new(nil, self), learning_phase]
      end

      def build
        @built = true
        @weight = Param.new(Xumo::SFloat.new(@input_length), 0)
        @weight_initializer.init_param(self, @weight)
        @weight_regularizer.param = @weight if @weight_regularizer
        @input_shape
      end

      def forward(x)
        @x = x
        y = Xumo::SFloat.zeros(*x.shape)
        x.shape[0].times do |i|
          y[i, false] = @weight.data[x[i, false]]
        end
        y
      end

      def backward(dy)
        @weight.grad += Xumo::SFloat.zeros(*@weight.data.shape)
        @x.shape[0].times do |i|
          @x.shape[1].times do |j|
            @weight.grad[@x[i, j]] += dy[i, j]
          end
        end
        nil
      end

      def regularizers
        @weight_regularizer ? [@weight_regularizer] : []
      end

      def to_hash
        super(input_shape: @input_shape, input_length: @input_length,
              weight_initializer: @weight_initializer.to_hash, weight_regularizer: @weight_regularizer&.to_hash)
      end

      def get_params
        { weight: @weight }
      end
    end

  end
end
