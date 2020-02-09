module DNN
  class Tensor
    attr_reader :data
    attr_accessor :link

    def self.convert(inputs, link = nil)
      if inputs.is_a?(Array)
        inputs.map { |input| Tensor.new(input, link) }
      elsif inputs.is_a?(Integer) || inputs.is_a?(Float)
        Tensor.new(Xumo::SFloat[inputs], link)
      else
        Tensor.new(inputs, link)
      end
    end

    def initialize(data, link = nil)
      @data = data
      @link = link
    end

    def >>(layer)
      layer.(self)
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
