class Integer
  alias dnn__add +
  def +(other)
    if other.is_a?(DNN::BaseTensor)
      DNN::Functions::Add.(DNN::Tensor.new(self), other)
    else
      dnn__add(other)
    end
  end

  alias dnn__sub -
  def -(other)
    if other.is_a?(DNN::BaseTensor)
      DNN::Functions::Sub.(DNN::Tensor.new(self), other)
    else
      dnn__sub(other)
    end
  end

  alias dnn__mul *
  def *(other)
    if other.is_a?(DNN::BaseTensor)
      DNN::Functions::Mul.(DNN::Tensor.new(self), other)
    else
      dnn__mul(other)
    end
  end

  alias dnn__div /
  def /(other)
    if other.is_a?(DNN::BaseTensor)
      DNN::Functions::Div.(DNN::Tensor.new(self), other)
    else
      dnn__div(other)
    end
  end
end

class Float
  alias dnn__add +
  def +(other)
    if other.is_a?(DNN::BaseTensor)
      DNN::Functions::Add.(DNN::Tensor.new(self), other)
    else
      dnn__add(other)
    end
  end

  alias dnn__sub -
  def -(other)
    if other.is_a?(DNN::BaseTensor)
      DNN::Functions::Sub.(DNN::Tensor.new(self), other)
    else
      dnn__sub(other)
    end
  end

  alias dnn__mul *
  def *(other)
    if other.is_a?(DNN::BaseTensor)
      DNN::Functions::Mul.(DNN::Tensor.new(self), other)
    else
      dnn__mul(other)
    end
  end

  alias dnn__div /
  def /(other)
    if other.is_a?(DNN::BaseTensor)
      DNN::Functions::Div.(DNN::Tensor.new(self), other)
    else
      dnn__div(other)
    end
  end
end

if RUBY_VERSION < "2.6.0"
  class Hash
    alias dnn__to_h to_h
    def to_h(&block)
      dnn__to_h unless block
      map(&block).to_h
    end
  end
end
