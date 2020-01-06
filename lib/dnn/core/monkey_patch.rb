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
