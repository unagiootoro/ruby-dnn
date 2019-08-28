include DNN::Layers
include DNN::Activations
include DNN::Optimizers
include DNN::Losses
include DNN::Models

class Generator < Model
  def initialize
    super
    @l1 = Dense.new(1024)
    @l2 = Dense.new(7 * 7 * 64)
    @l3 = Conv2DTranspose.new(64, 4, strides: 2, padding: true)
    @l4 = Conv2D.new(64, 4, padding: true)
    @l5 = Conv2DTranspose.new(32, 4, strides: 2, padding: true)
    @l6 = Conv2D.new(32, 4, padding: true)
    @l7 = Conv2D.new(1, 4, padding: true)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
    @bn3 = BatchNormalization.new
    @bn4 = BatchNormalization.new
    @bn5 = BatchNormalization.new
    @bn6 = BatchNormalization.new
  end

  def call(x)
    x = InputLayer.new(20).(x)
    x = @l1.(x)
    x = @bn1.(x)
    x = ReLU.(x)

    x = @l2.(x)
    x = @bn2.(x)
    x = ReLU.(x)

    x = Reshape.(x, [7, 7, 64])
    x = @l3.(x)
    x = @bn3.(x)
    x = ReLU.(x)

    x = @l4.(x)
    x = @bn4.(x)
    x = ReLU.(x)

    x = @l5.(x)
    x = @bn5.(x)
    x = ReLU.(x)

    x = @l6.(x)
    x = @bn6.(x)
    x = ReLU.(x)

    x = @l7.(x)
    x = Tanh.(x)
    x
  end
end


class Discriminator < Model
  def initialize
    super
    @l1 = Conv2D.new(32, 4, strides: 2, padding: true)
    @l2 = Conv2D.new(32, 4, padding: true)
    @l3 = Conv2D.new(64, 4, strides: 2, padding: true)
    @l4 = Conv2D.new(64, 4, padding: true)
    @l5 = Dense.new(1024)
    @l6 = Dense.new(1)
  end

  def call(x, trainable = true)
    has_param_layers.each do |layer|
      layer.trainable = trainable
    end
    x = InputLayer.new([28, 28, 1]).(x)
    x = @l1.(x)
    x = LeakyReLU.(x, 0.2)

    x = @l2.(x)
    x = LeakyReLU.(x, 0.2)

    x = @l3.(x)
    x = LeakyReLU.(x, 0.2)

    x = @l4.(x)
    x = LeakyReLU.(x, 0.2)

    x = Flatten.(x)
    x = @l5.(x)
    x = LeakyReLU.(x, 0.2)

    x = @l6.(x)
    x
  end
end


class DCGAN < Model
  attr_reader :gen
  attr_reader :dis

  def initialize(gen, dis)
    super()
    @gen = gen
    @dis = dis
  end

  def call(x)
    x = @gen.(x)
    x = @dis.(x, false)
    x
  end
end