require "dnn"
require "dnn/datasets/mnist"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses
include DNN::Savers
include DNN::Loaders

USE_MARSHAL = 0
USE_JSON = 1

EPOCHS = 3
BATCH_SIZE = 128

# Select save style from USE_MARSHAL or USE_JSON.
SAVE_STYLE = USE_MARSHAL

# When set a true, save data included model structure.
# This setting is enabled when SAVE_STYLE is USE_MARSHAL.
INCLUDE_MODEL = true

x_train, y_train = DNN::MNIST.load_train
x_test, y_test = DNN::MNIST.load_test

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

x_train = Numo::SFloat.cast(x_train) / 255
x_test = Numo::SFloat.cast(x_test) / 255

y_train = DNN::Utils.to_categorical(y_train, 10, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 10, Numo::SFloat)

class MLP < Model
  def initialize
    super
    @d1 = Dense.new(256)
    @d2 = Dense.new(256)
    @d3 = Dense.new(10)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
  end

  def forward(x)
    x = InputLayer.new(784).(x)
    x = @d1.(x)
    x = @bn1.(x)
    x = ReLU.(x)
    x = @d2.(x)
    x = @bn2.(x)
    x = ReLU.(x)
    x = @d3.(x)
    x
  end
end

model = MLP.new
model.setup(Adam.new, SoftmaxCrossEntropy.new)

trainer = DNN::Trainer.new(model)
trainer.train(x_train, y_train, EPOCHS, batch_size: BATCH_SIZE, test: [x_test, y_test])

if SAVE_STYLE == USE_MARSHAL
  saver = MarshalSaver.new(model, include_model: INCLUDE_MODEL)
  saver.save("trained_mnist.marshal")
  # model.save("trained_mnist.marshal") # This code is equivalent to the code above.
elsif SAVE_STYLE == USE_JSON
  saver = JSONSaver.new(model)
  saver.save("trained_mnist.json")
end

model2 = MLP.new
model2.setup(Adam.new, SoftmaxCrossEntropy.new)
model2.predict1(Numo::SFloat.zeros(784))
if SAVE_STYLE == USE_MARSHAL
  loader = MarshalLoader.new(model2)
  loader.load("trained_mnist.marshal")
  # model2 = MLP.load("trained_mnist.marshal") # This code is equivalent to the code above.
elsif SAVE_STYLE == USE_JSON
  loader = JSONLoader.new(model2)
  loader.load("trained_mnist.json")
end

evaluator = DNN::Evaluator.new(model2)
puts evaluator.evaluate(x_test, y_test)
