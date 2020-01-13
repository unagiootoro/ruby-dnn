class Integer
  alias dnn__add +
  def +(other)
    if other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      DNN::Layers::Add.(self, other)
    else
      dnn__add(other)
    end
  end

  alias dnn__sub -
  def -(other)
    if other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      DNN::Layers::Sub.(self, other)
    else
      dnn__sub(other)
    end
  end

  alias dnn__mul *
  def *(other)
    if other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      DNN::Layers::Mul.(self, other)
    else
      dnn__mul(other)
    end
  end

  alias dnn__div /
  def /(other)
    if other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      DNN::Layers::Div.(self, other)
    else
      dnn__div(other)
    end
  end
end

class Float
  alias dnn__add +
  def +(other)
    if other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      DNN::Layers::Add.(self, other)
    else
      dnn__add(other)
    end
  end

  alias dnn__sub -
  def -(other)
    if other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      DNN::Layers::Sub.(self, other)
    else
      dnn__sub(other)
    end
  end

  alias dnn__mul *
  def *(other)
    if other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      DNN::Layers::Mul.(self, other)
    else
      dnn__mul(other)
    end
  end

  alias dnn__div /
  def /(other)
    if other.is_a?(DNN::Tensor) || other.is_a?(DNN::Param)
      DNN::Layers::Div.(self, other)
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
