module DNN
  class Tensor
    attr_reader :data
    attr_reader :backward_index
    attr_reader :prev_link
    attr_reader :next_links

    def initialize(data, prev_link: nil, backward_index: nil)
      if data.is_a?(Integer)
        @data = Xumo::Int32[data]
      elsif data.is_a?(Float)
        @data = Xumo::SFloat[data]
      elsif data.is_a?(Tensor)
        @data = data.data
      else
        @data = data
      end
      @prev_link = prev_link
      @next_links = []
      @backward_index = backward_index
      @hold_datas = []
      @held_flags = []
    end

    def backward(grad = Xumo::SFloat[1], index = 0)
      @hold_datas[index] = grad
      @held_flags[index] = true
      return if @held_flags.compact.length < @next_links.length
      return unless requires_grad
      @prev_link.backward(@hold_datas.reduce(&:+), @backward_index) if @prev_link
      @hold_datas = []
      @held_flags = []
    end

    def requires_grad
      @prev_link ? @prev_link.requires_grad : false
    end

    def add_next_link(link)
      @next_links << link unless @next_links.include?(link)
    end

    def next_link_index(link)
      @next_links.index(link)
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
      Functions::Neg.(self)
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

    def transpose(*axes)
      Functions::Transpose.new(*axes).(self)
    end
  end
end
