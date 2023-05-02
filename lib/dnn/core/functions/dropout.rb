module DNN
  module Functions
    class Dropout < Function
      def initialize(dropout_ratio = 0.5, seed: rand(1 << 31), use_scale: true, learning_phase: false)
        super()
        @dropout_ratio = dropout_ratio
        @seed = seed
        @use_scale = use_scale
        @learning_phase = learning_phase
        @mask = nil
        @rnd = Random.new(@seed)
      end
    
      def forward(x)
        if @learning_phase
          Xumo::SFloat.srand(@rnd.rand(1 << 31))
          @mask = Xumo::SFloat.cast(Xumo::SFloat.new(*x.shape).rand >= @dropout_ratio)
          x = x * @mask
        elsif @use_scale
          x *= (1 - @dropout_ratio)
        end
        x
      end
    
      def backward(dy)
        dy * @mask
      end
    end    
  end
end
