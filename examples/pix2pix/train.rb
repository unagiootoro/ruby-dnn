# This example converts a CIFAR10 grayscale image to a color image.

require "dnn"
require "dnn/datasets/cifar10"
require "numo/linalg/autoloader"
require_relative "dcgan"

include DNN::Optimizers
include DNN::Losses

def load_dataset
  x, y = DNN::CIFAR10.load_train
  x_out = Numo::SFloat.cast(x)
  x_in = x_out.mean(axis: 3, keepdims: true)
  x_in = (x_in / 127.5) - 1
  x_out = (x_out / 127.5) - 1
  [x_in, x_out]
end

epochs = 20
batch_size = 128

gen = Generator.new([32, 32, 1])
dis = Discriminator.new([32, 32, 1], [32, 32, 3])
dcgan = DCGAN.new(gen, dis)

gen.setup(Adam.new(alpha: 0.0002, beta1: 0.5), MeanAbsoluteError.new)
dis.setup(Adam.new(alpha: 0.00001, beta1: 0.1), SigmoidCrossEntropy.new)
dcgan.setup(Adam.new(alpha: 0.0002, beta1: 0.5), SigmoidCrossEntropy.new)

x_in, x_out = load_dataset

iter1 = DNN::Iterator.new(x_in, x_out)
iter2 = DNN::Iterator.new(x_in, x_out)
num_batchs = x_in.shape[0] / batch_size
(1..epochs).each do |epoch|
  num_batchs.times do |index|
    x_in, x_out = iter1.next_batch(batch_size)
    gen_loss = gen.train_on_batch(x_in, x_out)

    images = gen.generate_images
    y_real = Numo::SFloat.ones(batch_size, 1)
    y_fake = Numo::SFloat.zeros(batch_size, 1)
    dis.enable_training
    dis_loss = dis.train_on_batch([x_in, x_out], y_real)
    dis_loss += dis.train_on_batch([x_in, images], y_fake)

    x_in, x_out = iter2.next_batch(batch_size)
    dcgan_loss = dcgan.train_on_batch(x_in, y_real)

    puts "epoch: #{epoch}, index: #{index}, gen_loss: #{gen_loss}, dis_loss: #{dis_loss}, dcgan_loss: #{dcgan_loss}"
  end
  iter1.reset
  iter2.reset
  dcgan.save("trained/dcgan_model_epoch#{epoch}.marshal")
end
