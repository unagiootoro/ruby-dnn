include DNN::Models
include DNN::Layers

class Generator < Model
  def initialize(input_shape)
    super()
    @input_shape = input_shape
    @l1 = Conv2D.new(32, 4, padding: true)
    @l2 = Conv2D.new(32, 4, strides: 2, padding: true)
    @l3 = Conv2D.new(64, 4, padding: true)
    @l4 = Conv2D.new(64, 4, strides: 2, padding: true)
    @l5 = Conv2D.new(128, 4, padding: true)
    @l6 = Conv2DTranspose.new(64, 4, strides: 2, padding: true)
    @l7 = Conv2D.new(64, 4, padding: true)
    @l8 = Conv2DTranspose.new(32, 4, strides: 2, padding: true)
    @l9 = Conv2D.new(32, 4, padding: true)
    @l10 = Conv2D.new(32, 4, padding: true)
    @l11 = Conv2D.new(3, 4, padding: true)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
    @bn3 = BatchNormalization.new
    @bn4 = BatchNormalization.new
    @bn5 = BatchNormalization.new
    @bn6 = BatchNormalization.new
    @bn7 = BatchNormalization.new
    @bn8 = BatchNormalization.new
    @bn9 = BatchNormalization.new
  end

  def forward(x)
    input = InputLayer.new(@input_shape).(x)
    x = @l1.(input)
    x = @bn1.(x)
    h1 = ReLU.(x)

    x = @l2.(h1)
    x = @bn2.(x)
    x = ReLU.(x)

    x = @l3.(x)
    x = @bn3.(x)
    h2 = ReLU.(x)

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
    x = @bn7.(x)
    x = ReLU.(x)
    x = Concatenate.(x, h2, axis: 3)

    x = @l8.(x)
    x = @bn8.(x)
    x = ReLU.(x)

    x = @l9.(x)
    x = @bn9.(x)
    x = ReLU.(x)
    x = Concatenate.(x, h1, axis: 3)

    x = @l10.(x)
    x = ReLU.(x)

    x = @l11.(x)
    x = Tanh.(x)
    x
  end
end

class Discriminator < Model
  def initialize(gen_input_shape, gen_output_shape)
    super()
    @gen_input_shape = gen_input_shape
    @gen_output_shape = gen_output_shape
    @l1_1 = Conv2D.new(32, 4, padding: true)
    @l1_2 = Conv2D.new(32, 4, padding: true)
    @l2 = Conv2D.new(32, 4, strides: 2, padding: true)
    @l3 = Conv2D.new(32, 4, padding: true)
    @l4 = Conv2D.new(64, 4, strides: 2, padding: true)
    @l5 = Conv2D.new(64, 4, padding: true)
    @l6 = Dense.new(1024)
    @l7 = Dense.new(1)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
    @bn3 = BatchNormalization.new
    @bn4 = BatchNormalization.new
    @bn5 = BatchNormalization.new
    @bn6 = BatchNormalization.new
  end

  def forward(inputs)
    input, images = *inputs
    x = InputLayer.new(@gen_input_shape).(input)
    x = @l1_1.(x)
    x = @bn1.(x)
    x1 = LeakyReLU.(x, 0.2)

    x = InputLayer.new(@gen_output_shape).(images)
    x = @l1_2.(x)
    x = @bn2.(x)
    x2 = LeakyReLU.(x, 0.2)

    x = Concatenate.(x1, x2)
    x = @l2.(x)
    x = @bn3.(x)
    x = LeakyReLU.(x, 0.2)

    x = @l3.(x)
    x = @bn4.(x)
    x = LeakyReLU.(x, 0.2)

    x = @l4.(x)
    x = @bn5.(x)
    x = LeakyReLU.(x, 0.2)

    x = @l5.(x)
    x = @bn6.(x)
    x = LeakyReLU.(x, 0.2)

    x = Flatten.(x)
    x = @l6.(x)
    x = LeakyReLU.(x, 0.2)

    x = @l7.(x)
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
  attr_reader :gen
  attr_reader :dis

  def initialize(gen, dis)
    super()
    @gen = gen
    @dis = dis
  end

  def forward(input)
    images = @gen.(input)
    @dis.disable_training
    out = @dis.([input, images])
    [images, out]
  end
end
