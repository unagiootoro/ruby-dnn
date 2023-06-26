require "dnn"
require "dnn/datasets/mnist"
require "dnn/image"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses

x_train, * = DNN::MNIST.load_train

x_train = Numo::SFloat.cast(x_train).reshape(x_train.shape[0], 784)

x_train /= 255

$z_dim = 2
$z_mean = nil
$z_sigma = nil

class Encoder < Model
  def initialize
    super
    @d1 = Dense.new(196)
    @d2 = Dense.new(49)
    @d3_1 = Dense.new($z_dim)
    @d3_2 = Dense.new($z_dim)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
  end

  def forward(x)
    x = InputLayer.new(784).(x)
    x = @d1.(x)
    x = @bn1.(x)
    x = ReLU.(x)
    x = @d2.(x)
    x = @bn2.(x)
    x = ReLU.(x)
    z_mean = @d3_1.(x)
    z_sigma = @d3_2.(x)
    [z_mean, z_sigma]
  end
end

class Decoder < Model
  def initialize
    super
    @d1 = Dense.new(196)
    @d2 = Dense.new(784)
    @bn1 = BatchNormalization.new
  end

  def forward(z)
    x = @d1.(z)
    x = @bn1.(x)
    x = ReLU.(x)
    x = @d2.(x)
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
    z = sampling(z_mean, z_sigma)
    x = @dec.(z)
    x
  end

  def sampling(z_mean, z_sigma)
    fs = DNN::Functions::FunctionSpace
    epsilon = DNN::Tensor.new(Numo::SFloat.new($z_dim).rand_norm(0, 1))
    fs.tanh(z_mean + z_sigma * epsilon)
  end
end

class VAELoss < Loss
  def forward(y, t)
    fs = DNN::Functions::FunctionSpace
    kl = -0.5 * ((1 + fs.log($z_sigma**2) - $z_mean**2 - $z_sigma**2).sum(axis: 1)).mean(axis: 0)
    fs.sigmoid_cross_entropy(y, t) + kl
  end
end

model = VAE.new
dec = model.dec
model.setup(Adam.new, VAELoss.new)

trainer = DNN::Trainer.new(model)
trainer.fit(x_train, x_train, 10, batch_size: 128, need_accuracy: false)

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
