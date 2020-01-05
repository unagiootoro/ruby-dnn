module DNN
  class Param
    attr_accessor :trainable
    attr_accessor :data
    attr_accessor :grad

    def initialize(data = nil, grad = nil)
      @data = data
      @grad = grad
      @trainable = true
    end

    def backward(grad)
      if @trainable
        @grad ||= Xumo::SFloat[0]
        if @data.shape == grad.shape
          @grad += grad
        elsif @data.shape == grad.shape[1..-1]
          @grad += grad.sum(0)
        else
          raise DNN_Error, "Shape is missmatch."
        end
      else
        @grad = Xumo::SFloat[0]
      end
    end
  end
end
