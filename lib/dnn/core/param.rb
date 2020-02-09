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
      other = Tensor.convert(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Layers::Add.(self, other)
    end

    def -(other)
      other = Tensor.convert(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Layers::Sub.(self, other)
    end

    def *(other)
      other = Tensor.convert(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Layers::Mul.(self, other)
    end

    def /(other)
      other = Tensor.convert(other) unless other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      Layers::Div.(self, other)
    end

    def **(index)
      Layers::Pow.new(index).(self)
    end
  end
end
