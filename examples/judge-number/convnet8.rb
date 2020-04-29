require "dnn"
require "numo/linalg/autoloader"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses

class ConvNet < Model
  def self.create(input_shape)
    convnet = ConvNet.new(input_shape, 32)
    convnet.setup(Adam.new, SoftmaxCrossEntropy.new)
    convnet
  end

  def initialize(input_shape, base_filter_size)
    super()
    @input_shape = input_shape
    @cv1 = Conv2D.new(base_filter_size, 3, padding: true)
    @cv2 = Conv2D.new(base_filter_size, 3, padding: true)
    @cv3 = Conv2D.new(base_filter_size * 2, 3, padding: true)
    @cv4 = Conv2D.new(base_filter_size * 2, 3, padding: true)
    @cv5 = Conv2D.new(base_filter_size * 4, 3, padding: true)
    @cv6 = Conv2D.new(base_filter_size * 4, 3, padding: true)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
    @bn3 = BatchNormalization.new
    @bn4 = BatchNormalization.new
    @d1 = Dense.new(512)
    @d2 = Dense.new(10)
  end

  def forward(x)
    x = InputLayer.new(@input_shape).(x)

    x = @cv1.(x)
    x = ReLU.(x)
    x = Dropout.(x, 0.25)

    x = @cv2.(x)
    x = @bn1.(x)
    x = ReLU.(x)
    x = MaxPool2D.(x, 2)

    x = @cv3.(x)
    x = ReLU.(x)
    x = Dropout.(x, 0.25)

    x = @cv4.(x)
    x = @bn2.(x)
    x = ReLU.(x)
    x = MaxPool2D.(x, 2)

    x = @cv5.(x)
    x = ReLU.(x)
    x = Dropout.(x, 0.25)

    x = @cv6.(x)
    x = @bn3.(x)
    x = ReLU.(x)
    x = MaxPool2D.(x, 2)

    x = Flatten.(x)
    x = @d1.(x)
    x = @bn4.(x)
    x = ReLU.(x)
    x = @d2.(x)
    x
  end
end
