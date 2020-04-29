require "dnn"
require "dnn/datasets/mnist"
require_relative "convnet8"

include DNN::Callbacks

x_train, y_train = DNN::MNIST.load_train
x_test, y_test = DNN::MNIST.load_test

x_train = Numo::SFloat.cast(x_train) / 255
x_test = Numo::SFloat.cast(x_test) / 255

y_train = DNN::Utils.to_categorical(y_train, 10, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 10, Numo::SFloat)

model = ConvNet.create([28, 28, 1])
model.add_callback(CheckPoint.new("trained/trained_mnist", interval: 5))

model.train(x_train, y_train, 20, batch_size: 128, test: [x_test, y_test])
