require "dnn"
require "dnn/image"
require_relative "dcgan"

include DNN::Loaders
Image = DNN::Image

batch_size = 100

dcgan = DCGAN.load("trained/dcgan_model_epoch20.marshal")
gen = dcgan.gen

Numo::SFloat.srand(rand(1 << 31))
noise = Numo::SFloat.new(batch_size, 20).rand(-1, 1)

images = gen.predict(noise)

batch_size.times do |i|
  img = Numo::UInt8.cast(((images[i, false] + 1) * 127.5).round)
  Image.write("img/img_#{i}.jpg", img)
end
