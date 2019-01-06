class DNN::LearningParam
  attr_accessor :data
  attr_accessor :grad
  attr_reader :layer

  def initialize(layer)
    @layer = layer
  end
end
