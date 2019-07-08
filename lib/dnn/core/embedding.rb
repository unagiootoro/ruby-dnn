module DNN
  module Layers

    class Embedding < HasParamLayer
      # @return [Integer] Return the input length.
      attr_reader :input_length
      # @return [Initializers::Initializer] Return the weight initializer.
      attr_reader :weight_initializer

      def self.from_hash(hash)
        self.new(hash[:input_shape], hash[:input_length],
                 weight_initializer: DNN::Utils.from_hash(hash[:weight_initializer]))
      end
      
      # @param [Integer | Array] input_dim_or_shape Set input data dimension or shape.
      # @param [Integer] input_length input Set the time series length of input data.
      def initialize(input_dim_or_shape, input_length, weight_initializer: Initializers::RandomUniform.new)
        super()
        @input_shape = input_dim_or_shape.is_a?(Array) ? input_dim_or_shape : [input_dim_or_shape]
        @input_length = input_length
        @weight_initializer = weight_initializer
      end

      def call(x)
        build unless built?
        [forward(x), Link.new(nil, self)]
      end

      def build
        @built = true
        @params[:weight] = @weight = Param.new(Xumo::SFloat.new(@input_length), 0)
        @weight_initializer.init_param(self, @weight)
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

      def to_hash
        super(input_shape: @input_shape, input_length: @input_length, weight_initializer: @weight_initializer.to_hash)
      end
    end

  end
end
