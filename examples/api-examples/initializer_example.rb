require "dnn"
require "dnn/datasets/mnist"
# If you use numo/linalg then please uncomment out.
# require "numo/linalg/autoloader"

include DNN::Models
include DNN::Layers
include DNN::Initializers
include DNN::Optimizers
include DNN::Losses
MNIST = DNN::MNIST

EPOCHS = 3
BATCH_SIZE = 128

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
    # Set the initial values of weight and bias to the initial values of He.
    @l1 = Dense.new(256, weight_initializer: He.new, bias_initializer: He.new)
    @l2 = Dense.new(256, weight_initializer: He.new, bias_initializer: He.new)
    @l3 = Dense.new(10, weight_initializer: He.new, bias_initializer: He.new)
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
