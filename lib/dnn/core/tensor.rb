module DNN
  class Tensor
    attr_reader :data
    attr_accessor :link

    def self.convert(inputs)
      if inputs.is_a?(Array)
        inputs.map { |input| Tensor.new(input) }
      else
        Tensor.new(inputs)
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
