module DNN
  module Functions

    class Embedding < Function
      def initialize(mask_zero)
        @mask_zero = mask_zero
      end

      def forward(x, weight)
        @x = x
        @weight = weight
        y = Xumo::SFloat.zeros(*x.shape)
        x.shape[0].times do |i|
          if @mask_zero
            x.shape[1].times do |j|
              index = x[i, j]
              y[i, j] = index == 0 ? 0 : weight[index]
            end
          else
            y[i, false] = weight[x[i, false]]
          end
        end
        y
      end

      def backward(dy)
        dweight = Xumo::SFloat.zeros(*@weight.shape)
        @x.shape[0].times do |i|
          @x.shape[1].times do |j|
            index = @x[i, j]
            if @mask_zero
              dweight[index] += dy[i, j] unless index == 0
            else
              dweight[index] += dy[i, j]
            end
          end
        end
        [nil, dweight]
      end
    end

  end
end
