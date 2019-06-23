class DNN::Param
  attr_accessor :data
  attr_accessor :grad

  def initialize(data = nil, grad = nil)
    @data = data
    @grad = grad
  end
end
