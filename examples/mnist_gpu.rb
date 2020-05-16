require "cumo/narray"
require "dnn"
require "dnn/datasets/mnist"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses

x_train, y_train = DNN::MNIST.load_train
x_test, y_test = DNN::MNIST.load_test

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

x_train = Numo::SFloat.cast(x_train) / 255
x_test = Numo::SFloat.cast(x_test) / 255

y_train = DNN::Utils.to_categorical(y_train, 10, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 10, Numo::SFloat)

if DNN.use_cumo?
  x_train = DNN::Utils.numo2cumo(x_train)
  y_train = DNN::Utils.numo2cumo(y_train)
  x_test = DNN::Utils.numo2cumo(x_test)
  y_test = DNN::Utils.numo2cumo(y_test)
end

model = Sequential.new

model << InputLayer.new(784)

model << Dense.new(256)
model << ReLU.new

model << Dense.new(256)
model << ReLU.new

model << Dense.new(10)

model.setup(Adam.new, SoftmaxCrossEntropy.new)

model.train(x_train, y_train, 10, batch_size: 128, test: [x_test, y_test])

accuracy, loss = model.evaluate(x_test, y_test)
puts "accuracy: #{accuracy}"
puts "loss: #{loss}"
