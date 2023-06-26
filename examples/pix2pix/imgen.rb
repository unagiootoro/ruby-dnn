require "dnn"
require "dnn/image"
require "dnn/datasets/cifar10"
require_relative "dcgan"

def load_dataset
  x, y = DNN::CIFAR10.load_test
  x_out = Numo::SFloat.cast(x)
  x_in = x_out.mean(axis: 3, keepdims: true)
  x_in = (x_in / 127.5) - 1
  x_out = (x_out / 127.5) - 1
  [x_in, x_out]
end

batch_size = 100

dcgan = DCGAN.load("trained/dcgan_model_epoch20.marshal")
gen = dcgan.gen

x_in, x_out = load_dataset
images = gen.predict(x_in[0...batch_size, false])

batch_size.times do |i|
  img = Numo::UInt8.cast(((images[i, false] + 1) * 127.5).round)
  DNN::Image.write("img/img_#{i}.jpg", img)
end
