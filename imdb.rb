$:.unshift "#{__dir__}/lib"

require "dnn"
require "numo/linalg/autoloader"

include DNN::Layers
include DNN::Activations
include DNN::Optimizers
include DNN::Losses
include DNN::Models

x_train, y_train, x_test, y_test = Marshal.load(File.binread("imdb.marshal"))

class IMDB < Model
  def initialize
    super
    @l0 = Embedding.new(200, 10000)
    @l1 = LSTM.new(64)
    @l2 = LSTM.new(64, return_sequences: false)
    @l3 = Dense.new(1)
  end

  def call(x)
    x = @l0.(x)
    x = Reshape.(x, [10, 20])
    x = @l1.(x)
    x = @l2.(x)
    x = @l3.(x)
    x
  end
end

model = IMDB.new

model.setup(AdaBound.new, SigmoidCrossEntropy.new)

model.train(x_train, y_train, 10, batch_size: 100, test: [x_train, y_train])
