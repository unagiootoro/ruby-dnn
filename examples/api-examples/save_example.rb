require "dnn"
require "dnn/datasets/mnist"
# If you use numo/linalg then please uncomment out.
# require "numo/linalg/autoloader"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses
include DNN::Savers
include DNN::Loaders
MNIST = DNN::MNIST

USE_MARSHAL = 0
USE_JSON = 1

EPOCHS = 3
BATCH_SIZE = 128

# Select save style from USE_MARSHAL or USE_JSON.
SAVE_STYLE = USE_MARSHAL

# When set a true, save data included optimizer status.
# This setting is enabled when SAVE_STYLE is USE_MARSHAL.
INCLUDE_OPTIMIZER = false

x_train, y_train = MNIST.load_train
x_test, y_test = MNIST.load_test

x_train = Numo::SFloat.cast(x_train).reshape(x_train.shape[0], 784)
x_test = Numo::SFloat.cast(x_test).reshape(x_test.shape[0], 784)

x_train /= 255
x_test /= 255

y_train = DNN::Utils.to_categorical(y_train, 10, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 10, Numo::SFloat)

class MLP < Model
  def initialize
    super
    @l1 = Dense.new(256)
    @l2 = Dense.new(256)
    @l3 = Dense.new(10)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
  end

  def call(x)
    x = InputLayer.(x)
    x = @l1.(x)
    x = @bn1.(x)
    x = ReLU.(x)
    x = @l2.(x)
    x = @bn2.(x)
    x = ReLU.(x)
    x = @l3.(x)
    x
  end
end

model = MLP.new
model.setup(Adam.new, SoftmaxCrossEntropy.new)
model.train(x_train, y_train, EPOCHS, batch_size: BATCH_SIZE, test: [x_test, y_test])

if SAVE_STYLE == USE_MARSHAL
  saver = MarshalSaver.new(model, include_optimizer: INCLUDE_OPTIMIZER)
  saver.save("trained_mnist.marshal")
  # model.save("trained_mnist.marshal") # This code is equivalent to the code above.
elsif SAVE_STYLE == USE_JSON
  saver = JSONSaver.new(model)
  saver.save("trained_mnist.json")
end

model2 = MLP.new
if SAVE_STYLE == USE_MARSHAL
  loader = MarshalLoader.new(model2)
  loader.load("trained_mnist.marshal")
  # MLP.load("trained_mnist.marshal") # This code is equivalent to the code above.
elsif SAVE_STYLE == USE_JSON
  loader = JSONLoader.new(model2)
  loader.load("trained_mnist.json")
end

puts model2.accuracy(x_test, y_test)
