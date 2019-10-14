require "dnn"
require "dnn/datasets/cifar100"
# If you use numo/linalg then please uncomment out.
# require "numo/linalg/autoloader"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses
CIFAR100 = DNN::CIFAR100

x_train, y_train = CIFAR100.load_train
x_test, y_test = CIFAR100.load_test

x_train = Numo::SFloat.cast(x_train)
x_test = Numo::SFloat.cast(x_test)

x_train /= 255
x_test /= 255

y_train = y_train[true, 1]
y_test = y_test[true, 1]

y_train = DNN::Utils.to_categorical(y_train, 100, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 100, Numo::SFloat)

model = Sequential.new

model << InputLayer.new([32, 32, 3])

model << Conv2D.new(16, 5, padding: true)
model << BatchNormalization.new
model << ReLU.new

model << Conv2D.new(16, 5, padding: true)
model << BatchNormalization.new
model << ReLU.new

model << MaxPool2D.new(2)

model << Conv2D.new(32, 5, padding: true)
model << BatchNormalization.new
model << ReLU.new

model << Conv2D.new(32, 5, padding: true)
model << BatchNormalization.new
model << ReLU.new

model << MaxPool2D.new(2)

model << Conv2D.new(64, 5, padding: true)
model << BatchNormalization.new
model << ReLU.new

model << Conv2D.new(64, 5, padding: true)
model << BatchNormalization.new
model << ReLU.new

model << Flatten.new

model << Dense.new(1024)
model << BatchNormalization.new
model << ReLU.new
model << Dropout.new(0.5)

model << Dense.new(100)

model.setup(Adam.new, SoftmaxCrossEntropy.new)

model.train(x_train, y_train, 10, batch_size: 100, test: [x_test, y_test])
