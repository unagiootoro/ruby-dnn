require "dnn"
require_relative "convnet8"

model = ConvNet.load("trained_mnist_epoch20.marshal")
model.save_params("trained_mnist_params.marshal")
