require "dnn"
require "dnn/datasets/mnist"
require "numo/linalg/autoloader"
require_relative "dcgan"

include DNN::Optimizers
include DNN::Losses
MNIST = DNN::MNIST

Numo::SFloat.srand(rand(1 << 31))

epochs = 20
batch_size = 128

gen = Generator.new
dis = Discriminator.new
dcgan = DCGAN.new(gen, dis)

dis.setup(Adam.new(alpha: 0.00001, beta1: 0.1), SigmoidCrossEntropy.new)
dcgan.setup(Adam.new(alpha: 0.0002, beta1: 0.5), SigmoidCrossEntropy.new)

x_train, y_train = MNIST.load_train
x_train = Numo::SFloat.cast(x_train)
x_train = x_train / 127.5 - 1

iter = DNN::Iterator.new(x_train, y_train)
num_batchs = x_train.shape[0] / batch_size
(1..epochs).each do |epoch|
  puts "epoch: #{epoch}"
  num_batchs.times do |index|
    x_batch, y_batch = iter.next_batch(batch_size)
    noise = Numo::SFloat.new(batch_size, 20).rand(-1, 1)
    images = gen.predict(noise)
    x = x_batch.concatenate(images)
    y = Numo::SFloat.cast([1] * batch_size + [0] * batch_size).reshape(batch_size * 2, 1)
    dis_loss = dis.train_on_batch(x, y)

    noise = Numo::SFloat.new(batch_size, 20).rand(-1, 1)
    label = Numo::SFloat.cast([1] * batch_size).reshape(batch_size, 1)
    dcgan_loss = dcgan.train_on_batch(noise, label)

    puts "index: #{index}, dis_loss: #{dis_loss.mean}, dcgan_loss: #{dcgan_loss.mean}"
  end
  iter.reset
  dcgan.save("trained/dcgan_model_epoch#{epoch}.marshal")
end
