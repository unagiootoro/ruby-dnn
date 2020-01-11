require "dnn"
require "dnn/datasets/mnist"
require "dnn/image"
require "numo/linalg/autoloader"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses

x_train, y_train = DNN::MNIST.load_train
x_test, y_test = DNN::MNIST.load_test

x_train = Numo::SFloat.cast(x_train).reshape(x_train.shape[0], 784)
x_test = Numo::SFloat.cast(x_test).reshape(x_test.shape[0], 784)

x_train /= 255
x_test /= 255

y_train = DNN::Utils.to_categorical(y_train, 10, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 10, Numo::SFloat)

$z_dim = 2
$z_mean = nil
$z_sigma = nil

class Sampling < MergeLayer
  def forward(z_mean, z_sigma)
    epsilon = DNN::Tensor.new(Numo::SFloat.new($z_dim).rand_norm(0, 1))
    Tanh.(z_mean + z_sigma * epsilon)
  end
end

class Encoder < Model
  def initialize
    super
    @l1 = Dense.new(196)
    @l2 = Dense.new(49)
    @l3_1 = Dense.new($z_dim)
    @l3_2 = Dense.new($z_dim)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
  end

  def forward(x)
    x = InputLayer.new(784).(x)
    x = @l1.(x)
    x = @bn1.(x)
    x = ReLU.(x)
    x = @l2.(x)
    x = @bn2.(x)
    x = ReLU.(x)
    z_mean = @l3_1.(x)
    z_sigma = @l3_2.(x)
    [z_mean, z_sigma]
  end
end

class Decoder < Model
  def initialize
    super
    @l3 = Dense.new(196)
    @l4 = Dense.new(784)
    @bn1 = BatchNormalization.new
  end

  def forward(z)
    x = @l3.(z)
    x = @bn1.(x)
    x = ReLU.(x)
    x = @l4.(x)
    x
  end
end

class VAE < Model
  attr_accessor :enc
  attr_accessor :dec

  def initialize(enc = nil, dec = nil)
    super()
    @enc = enc || Encoder.new
    @dec = dec || Decoder.new
  end

  def forward(x)
    z_mean, z_sigma = @enc.(x)
    $z_mean, $z_sigma = z_mean, z_sigma
    z = Sampling.(z_mean, z_sigma)
    x = @dec.(z)
    x
  end
end

class VAELoss < Loss
  def forward(y, t)
    kl = -0.5 * Mean.(Sum.(1 + Log.($z_sigma**2) - $z_mean**2 - $z_sigma**2, axis: 1), axis: 0)
    SigmoidCrossEntropy.(y, t) + kl
  end
end

model = VAE.new
dec = model.dec
model.setup(Adam.new, VAELoss.new)

model.train(x_train, x_train, 10, batch_size: 100)

images = []
10.times do |i|
  10.times do |j|
    z1 = (i / 4.5) - 1
    z2 = (j / 4.5) - 1
    z = Numo::SFloat[z1, z2]
    out = DNN::Utils.sigmoid(dec.predict1(z))
    img = Numo::UInt8.cast(out * 255).reshape(28, 28, 1)
    DNN::Image.write("img/img_#{i}_#{j}.png", img)
  end
end
