require "dnn"
require "dnn/mnist"
# If you use numo/linalg then please uncomment out.
# require "numo/linalg/autoloader"

include DNN::Layers
include DNN::Activations
include DNN::Optimizers
include DNN::Losses
include DNN::Models
MNIST = DNN::MNIST

x_train, y_train = MNIST.load_train
x_test, y_test = MNIST.load_test

x_train = Numo::SFloat.cast(x_train).reshape(x_train.shape[0], 784)
x_test = Numo::SFloat.cast(x_test).reshape(x_test.shape[0], 784)

x_train /= 255
x_test /= 255

y_train = DNN::Utils.to_categorical(y_train, 10, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 10, Numo::SFloat)

model = Sequential.new

model << InputLayer.new(784)

model << Dense.new(256)
model << ReLU.new

model << Dense.new(256)
model << ReLU.new

model << Dense.new(10)

model.setup(RMSProp.new, SoftmaxCrossEntropy.new)

model.train(x_train, y_train, 10, batch_size: 100, test: [x_test, y_test])
