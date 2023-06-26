include DNN::Models
include DNN::Layers

class Generator < Model
  def initialize
    super
    @d1 = Dense.new(1024)
    @d2 = Dense.new(7 * 7 * 64)
    @cv1 = Conv2D.new(64, 4, padding: true)
    @cvt1 = Conv2DTranspose.new(64, 4, strides: 2, padding: true)
    @cvt2 = Conv2DTranspose.new(32, 4, strides: 2, padding: true)
    @cv2 = Conv2D.new(32, 4, padding: true)
    @cv3 = Conv2D.new(1, 4, padding: true)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
    @bn3 = BatchNormalization.new
    @bn4 = BatchNormalization.new
    @bn5 = BatchNormalization.new
    @bn6 = BatchNormalization.new
  end

  def forward(x)
    x = InputLayer.new(20).(x)
    x = @d1.(x)
    x = @bn1.(x)
    x = ReLU.(x)

    x = @d2.(x)
    x = @bn2.(x)
    x = ReLU.(x)

    x = Reshape.(x, [7, 7, 64])
    x = @cvt1.(x)
    x = @bn3.(x)
    x = ReLU.(x)

    x = @cv1.(x)
    x = @bn4.(x)
    x = ReLU.(x)

    x = @cvt2.(x)
    x = @bn5.(x)
    x = ReLU.(x)

    x = @cv2.(x)
    x = @bn6.(x)
    x = ReLU.(x)

    x = @cv3.(x)
    x = Tanh.(x)
    x
  end
end

class Discriminator < Model
  def initialize
    super
    @cv1 = Conv2D.new(32, 4, strides: 2, padding: true)
    @cv2 = Conv2D.new(32, 4, padding: true)
    @cv3 = Conv2D.new(64, 4, strides: 2, padding: true)
    @cv4 = Conv2D.new(64, 4, padding: true)
    @d1 = Dense.new(1024)
    @d2 = Dense.new(1)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
    @bn3 = BatchNormalization.new
  end

  def forward(x)
    x = InputLayer.new([28, 28, 1]).(x)
    x = @cv1.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv2.(x)
    x = @bn1.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv3.(x)
    x = @bn2.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv4.(x)
    x = @bn3.(x)
    x = LeakyReLU.(x, 0.2)

    x = Flatten.(x)
    x = @d1.(x)
    x = LeakyReLU.(x, 0.2)

    x = @d2.(x)
    x
  end

  def enable_training
    trainable_layers.each do |layer|
      layer.trainable = true
    end
  end
  
  def disable_training
    trainable_layers.each do |layer|
      layer.trainable = false
    end
  end
end

class DCGAN < Model
  attr_accessor :gen
  attr_accessor :dis

  def initialize(gen = nil, dis = nil)
    super()
    @gen = gen
    @dis = dis
  end

  def forward(x)
    x = @gen.(x)
    @dis.disable_training
    x = @dis.(x)
    x
  end
end
