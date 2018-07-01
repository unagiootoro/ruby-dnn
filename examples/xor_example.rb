require "dnn"

include Numo
include DNN::Layers
include DNN::Activations
include DNN::Optimizers
Model = DNN::Model

x = SFloat[[0, 0], [1, 0], [0, 1], [1, 1]]
y = SFloat[[0], [1], [1], [0]]

model = Model.new

model << InputLayer.new(2)
model << Dense.new(16)
model << ReLU.new
model << Dense.new(1)
model << SigmoidWithLoss.new

model.compile(SGD.new)

model.train(x, y, 20000, batch_size: 4, verbose: false)

p model.predict(x)
