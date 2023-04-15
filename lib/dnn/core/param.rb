module DNN
  class Param
    attr_accessor :requires_grad
    attr_accessor :data
    attr_accessor :grad

    def initialize(data = nil, grad = nil)
      if data.is_a?(Integer)
        @data = Xumo::Int32[data]
      elsif data.is_a?(Float)
        @data = Xumo::SFloat[data]
      else
        @data = data
      end
      @grad = grad
      @requires_grad = true
    end

    def backward(grad)
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
      other = Tensor.new(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Functions::Add.(self, other)
    end

    def -(other)
      other = Tensor.new(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Functions::Sub.(self, other)
    end

    def *(other)
      other = Tensor.new(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Functions::Mul.(self, other)
    end

    def /(other)
      other = Tensor.new(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Functions::Div.(self, other)
    end

    def **(index)
      Functions::Pow.new(index).(self)
    end

    def dot(other)
      other = Tensor.new(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Functions::Dot.(self, other)
    end

    def flatten
      Functions::Flatten.new.(self)
    end

    def reshape(*shape)
      Functions::Reshape.new(shape).(self)
    end

    def transpose(*axes)
      Functions::Transpose.new(*axes).(self)
    end
  end
end
