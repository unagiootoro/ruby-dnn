require "dnn"
require "dnn/mnist"
require "numo/linalg/autoloader"
require_relative "dcgan"

MNIST = DNN::MNIST

gen = Generator.new
dis = Discriminator.new
dcgan = DCGAN.new(gen, dis)

dis.setup(Adam.new(alpha: 0.00001, beta1: 0.1), SigmoidCrossEntropy.new)
dcgan.setup(Adam.new(alpha: 0.0002, beta1: 0.5), SigmoidCrossEntropy.new)


epochs = 20
batch_size = 128

x_train, y_train = MNIST.load_train

x_train = Numo::SFloat.cast(x_train)

x_train = x_train / 127.5 - 1

num_batchs = (x_train.shape[0] / batch_size)
iter = DNN::Iterator.new(x_train, y_train)
(1..epochs).each do |epoch|
  puts "epoch: #{epoch}"
  num_batchs.times do |index|
    Numo::SFloat.srand(rand(1 << 31))
    noise = Numo::SFloat.new(batch_size, 20).rand(-1, 1)
    images = gen.predict(noise)

    x_batch, y_batch = iter.next_batch(batch_size)
    x = x_batch.concatenate(images)
    y = Numo::SFloat.cast([1] * batch_size + [0] * batch_size).reshape(batch_size * 2, 1)
    dis_loss = dis.train_on_batch(x, y)

    Numo::SFloat.srand(rand(1 << 31))
    noise = Numo::SFloat.new(batch_size, 20).rand(-1, 1)
    label = Numo::SFloat.cast([1] * batch_size).reshape(batch_size, 1)
    dcgan_loss = dcgan.train_on_batch(noise, label)

    puts "index: #{index}, dis_loss: #{dis_loss.mean}, dcgan_loss: #{dcgan_loss.mean}"
  end
  dcgan.save("trained/dcgan_model_epoch#{epoch}.marshal")
end
