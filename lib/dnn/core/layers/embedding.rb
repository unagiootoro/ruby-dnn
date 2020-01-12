module DNN
  module Layers

    class Embedding < TrainableLayer
      include LayerNode

      attr_reader :input_length
      attr_reader :weight
      attr_reader :weight_initializer
      attr_reader :weight_regularizer

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
        @weight = Param.new(nil, Xumo::SFloat[0])
      end

      def build(input_shape)
        super(@input_shape)
        @weight.data = Xumo::SFloat.new(@input_length)
        @weight_initializer.init_param(self, @weight)
        @weight_regularizer.param = @weight if @weight_regularizer
      end

      def forward_node(x)
        @x = x
        y = Xumo::SFloat.zeros(*x.shape)
        x.shape[0].times do |i|
          y[i, false] = @weight.data[x[i, false]]
        end
        y
      end

      def backward_node(dy)
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

      def load_hash(hash)
        initialize(hash[:input_shape], hash[:input_length],
                   weight_initializer: Initializers::Initializer.from_hash(hash[:weight_initializer]),
                   weight_regularizer: Regularizers::Regularizer.from_hash(hash[:weight_regularizer]))
      end

      def get_params
        { weight: @weight }
      end
    end

  end
end
