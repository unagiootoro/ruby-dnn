require "dnn"
require "dnn/image"
require "numo/linalg/autoloader"
require_relative "dcgan"

include DNN::Loaders
Image = DNN::Image

batch_size = 100

gen = Generator.new
dis = Discriminator.new
dcgan = DCGAN.new(gen, dis)
dcgan.predict1(Numo::SFloat.zeros(20))

loader = MarshalLoader.new(dcgan)
loader.load("trained/dcgan_model_epoch20.marshal")

Numo::SFloat.srand(rand(1 << 31))
noise = Numo::SFloat.new(batch_size, 20).rand(-1, 1)

images = gen.predict(noise)

batch_size.times do |i|
  img = Numo::UInt8.cast(((images[i, false] + 1) * 127.5).round)
  Image.write("img/img_#{i}.jpg", img)
end
