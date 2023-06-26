require "dnn"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses

x = Numo::SFloat[[0, 0], [1, 0], [0, 1], [1, 1]]
y = Numo::SFloat[[0], [1], [1], [0]]

model = Sequential.new

model << InputLayer.new(2)
model << Dense.new(16)
model << ReLU.new
model << Dense.new(1)

model.setup(SGD.new, SigmoidCrossEntropy.new)

trainer = DNN::Trainer.new(model)
trainer.fit(x, y, 20000, batch_size: 4, verbose: false)

p model.predict(x)
