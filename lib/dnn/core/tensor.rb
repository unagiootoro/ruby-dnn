module DNN
  class Tensor
    attr_reader :data
    attr_accessor :link

    def initialize(data, link = nil)
      @data = data
      @link = link
    end

    def shape
      @data.shape
    end

    def +@
      self
    end

    def -@
      self * -1
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
