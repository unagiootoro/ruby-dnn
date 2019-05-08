module DNN

  class Lasso
    def initialize(l1_lambda, param)
      @l1_lambda = l1_lambda
      @param = param
    end

    def forward(x)
      x + @l1_lambda * @param.data.abs.sum
    end

    def backward
      dlasso = Xumo::SFloat.ones(*@param.data.shape)
      dlasso[@param.data < 0] = -1
      @param.grad += @l1_lambda * dlasso
    end
  end


  class Ridge
    def initialize(l2_lambda, param)
      @l2_lambda = l2_lambda
      @param = param
    end

    def forward(x)
      x + 0.5 * @l2_lambda * (@param.data**2).sum
    end

    def backward
      @param.grad += @l2_lambda * @param.data
    end
  end

end
