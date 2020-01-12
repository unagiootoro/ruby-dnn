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

    def shape
      @data.shape
    end

    def +@
      self
    end

    def -@
      Neg.(self)
    end

    def +(other)
      Layers::Add.(self, other)
    end

    def -(other)
      Layers::Sub.(self, other)
    end

    def *(other)
      Layers::Mul.(self, other)
    end

    def /(other)
      Layers::Div.(self, other)
    end

    def **(index)
      Layers::Pow.new(index).(self)
    end
  end
end
