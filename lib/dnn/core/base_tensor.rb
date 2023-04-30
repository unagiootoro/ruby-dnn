module DNN
  class BaseTensor
    attr_reader :data
    attr_reader :next_links

    def initialize(data)
      if data.is_a?(Integer)
        @data = Xumo::Int32[data]
      elsif data.is_a?(Float)
        @data = Xumo::SFloat[data]
      elsif data.is_a?(Tensor)
        @data = data.data
      else
        @data = data
      end
      @next_links = []
      @hold_datas = []
      @held_flags = []
    end

    def backward(grad = Xumo::SFloat[1], index = 0)
      @hold_datas[index] = grad
      @held_flags[index] = true
      return if @held_flags.compact.length < @next_links.length
      return unless requires_grad
      backward_internal(@hold_datas.reduce(&:+))
      @hold_datas = []
      @held_flags = []
    end

    private def backward_internal(grad)
      raise NotImplementedError
    end

    def requires_grad
      raise NotImplementedError
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

    def size
      @data.size
    end

    def +@
      self
    end

    def -@
      Functions::Neg.(self)
    end

    def +(other)
      other = Tensor.new(other) unless other.is_a?(DNN::BaseTensor)
      Functions::Add.(self, other)
    end

    def -(other)
      other = Tensor.new(other) unless other.is_a?(DNN::BaseTensor)
      Functions::Sub.(self, other)
    end

    def *(other)
      other = Tensor.new(other) unless other.is_a?(DNN::BaseTensor)
      Functions::Mul.(self, other)
    end

    def /(other)
      other = Tensor.new(other) unless other.is_a?(DNN::BaseTensor)
      Functions::Div.(self, other)
    end

    def **(index)
      Functions::Pow.new(index).(self)
    end

    def dot(other)
      other = Tensor.new(other) unless other.is_a?(DNN::BaseTensor)
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

    def sum(axis: nil, keepdims: true)
      Functions::Sum.new(axis: axis, keepdims: keepdims).(self)
    end

    def mean(axis: nil, keepdims: true)
      Functions::Mean.new(axis: axis, keepdims: keepdims).(self)
    end

    def abs
      Functions::Abs.new.(self)
    end

    def max(axis: nil, keepdims: true)
      Functions::Max.new(axis: axis, keepdims: keepdims).(self)
    end
  end
end
