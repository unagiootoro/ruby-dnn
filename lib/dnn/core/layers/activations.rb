module DNN
  module Layers

    class Sigmoid < Layer
      def forward(x)
        Functions::FunctionSpace.sigmoid(x)
      end
    end

    class Tanh < Layer
      def forward(x)
        Functions::FunctionSpace.tanh(x)
      end
    end

    class Softsign < Layer
      def forward(x)
        Functions::FunctionSpace.softsign(x)
      end
    end

    class Softplus < Layer
      def forward(x)
        Functions::FunctionSpace.softplus(x)
      end
    end

    class Swish < Layer
      def forward(x)
        Functions::FunctionSpace.swish(x)
      end
    end

    class ReLU < Layer
      def forward(x)
        Functions::FunctionSpace.relu(x)
      end
    end

    class LeakyReLU < Layer
      attr_reader :alpha

      # @param [Float] alpha The slope when the output value is negative.
      def initialize(alpha = 0.3)
        super()
        @alpha = alpha
      end

      def forward(x)
        Functions::FunctionSpace.leaky_relu(x, alpha: @alpha)
      end

      def to_hash
        super(alpha: @alpha)
      end

      def load_hash(hash)
        initialize(hash[:alpha])
      end
    end

    class ELU < Layer
      attr_reader :alpha

      # @param [Float] alpha The slope when the output value is negative.
      def initialize(alpha = 1.0)
        super()
        @alpha = alpha
      end

      def forward(x)
        Functions::FunctionSpace.elu(x, alpha: @alpha)
      end

      def to_hash
        super(alpha: @alpha)
      end

      def load_hash(hash)
        initialize(hash[:alpha])
      end
    end

    class Mish < Layer
      def forward(x)
        Functions::FunctionSpace.mish(x)
      end
    end

  end
end
