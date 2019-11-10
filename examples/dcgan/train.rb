require "dnn"
require "dnn/datasets/mnist"
require "numo/linalg/autoloader"
require_relative "dcgan"

include DNN::Optimizers
include DNN::Losses
include DNN::Callbacks
MNIST = DNN::MNIST

Numo::SFloat.srand(rand(1 << 31))

epochs = 20
batch_size = 128

gen = Generator.new
dis = Discriminator.new
dcgan = DCGAN.new(gen, dis)

dis.setup(Adam.new(alpha: 0.00001, beta1: 0.1), SigmoidCrossEntropy.new)
dcgan.setup(Adam.new(alpha: 0.0002, beta1: 0.5), SigmoidCrossEntropy.new)
dcgan.add_callback(CheckPoint.new("trained/dcgan_model"))

x_train, * = MNIST.load_train
x_train = Numo::SFloat.cast(x_train)
x_train = x_train / 127.5 - 1

iter = DNN::Iterator.new(x_train, x_train, last_round_down: true)
dcgan.fit_by_iterator(iter, epochs, batch_size: batch_size)
