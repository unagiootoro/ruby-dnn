require "dnn"
require "dnn/datasets/iris"
# If you use numo/linalg then please uncomment out.
# require "numo/linalg/autoloader"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses

x, y = DNN::Iris.load(true)
x_train, y_train = x[0...100, true], y[0...100]
x_test, y_test = x[100...150, true], y[100...150]

y_train = DNN::Utils.to_categorical(y_train, 3, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 3, Numo::SFloat)

model = Sequential.new

model << InputLayer.new(4)

model << Dense.new(64)
model << ReLU.new

model << Dense.new(3)

model.setup(Adam.new, SoftmaxCrossEntropy.new)

model.train(x_train, y_train, 500, batch_size: 32, test: [x_test, y_test])

accuracy, loss = model.evaluate(x_test, y_test)
puts "accuracy: #{accuracy}"
puts "loss: #{loss}"
