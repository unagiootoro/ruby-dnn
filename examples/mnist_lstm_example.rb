require "dnn"
require "dnn/lib/mnist"
# If you use numo/linalg then please uncomment out.
# require "numo/linalg/autoloader"

include DNN::Layers
include DNN::Activations
include DNN::Optimizers
include DNN::Losses
Model = DNN::Model
MNIST = DNN::MNIST

x_train, y_train = MNIST.load_train
x_test, y_test = MNIST.load_test

x_train = Numo::SFloat.cast(x_train).reshape(x_train.shape[0], 28, 28)
x_test = Numo::SFloat.cast(x_test).reshape(x_test.shape[0], 28, 28)

x_train /= 255
x_test /= 255

y_train = DNN::Utils.to_categorical(y_train, 10, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 10, Numo::SFloat)

model = Model.new

model << InputLayer.new([28, 28])

model << LSTM.new(200)
model << LSTM.new(200, return_sequences: false)

model << Dense.new(10)

model.compile(Adam.new, SoftmaxCrossEntropy.new)

model.train(x_train, y_train, 10, batch_size: 100, test: [x_test, y_test])
