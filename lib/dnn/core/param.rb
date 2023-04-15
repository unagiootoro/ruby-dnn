module DNN
  class Param < BaseTensor
    attr_accessor :requires_grad
    attr_accessor :grad
    attr_writer :data

    def initialize(data = nil, grad = nil)
      super(data)
      @grad = grad
      @requires_grad = true
    end

    private def backward_internal(grad)
      if @requires_grad
        @grad ||= Xumo::SFloat[0]
        if @data.shape == grad.shape || (grad.shape[-1] == 1 && @data.shape == grad.shape[0...-1])
          @grad += grad
        elsif @data.shape == grad.shape[1..-1]
          @grad += grad.sum(0)
        else
          p [@data.shape, grad.shape]
          raise DNNError, "Shape is missmatch."
        end
      else
        @grad = Xumo::SFloat[0]
      end
    end
  end
end
