require "dnn"
require "dnn/datasets/cifar100"
# If you use numo/linalg then please uncomment out.
# require "numo/linalg/autoloader"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses

x_train, y_train = DNN::CIFAR100.load_train
x_test, y_test = DNN::CIFAR100.load_test

x_train = Numo::SFloat.cast(x_train) / 255
x_test = Numo::SFloat.cast(x_test) / 255

y_train = DNN::Utils.to_categorical(y_train, 100, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 100, Numo::SFloat)

model = Sequential.new

model << InputLayer.new([32, 32, 3])

model << Conv2D.new(32, 3, padding: true)
model << Dropout.new(0.25)
model << ReLU.new

model << Conv2D.new(32, 3, padding: true)
model << BatchNormalization.new
model << ReLU.new
model << MaxPool2D.new(2)

model << Conv2D.new(64, 3, padding: true)
model << Dropout.new(0.25)
model << ReLU.new

model << Conv2D.new(64, 3, padding: true)
model << BatchNormalization.new
model << ReLU.new
model << MaxPool2D.new(2)

model << Conv2D.new(128, 3, padding: true)
model << Dropout.new(0.25)
model << ReLU.new

model << Conv2D.new(128, 3, padding: true)
model << BatchNormalization.new
model << ReLU.new

model << Flatten.new

model << Dense.new(512)
model << BatchNormalization.new
model << ReLU.new

model << Dense.new(100)

model.setup(Adam.new, SoftmaxCrossEntropy.new)

model.train(x_train, y_train, 10, batch_size: 128, test: [x_test, y_test])

accuracy, loss = model.evaluate(x_test, y_test)
puts "accuracy: #{accuracy}"
puts "loss: #{loss}"
