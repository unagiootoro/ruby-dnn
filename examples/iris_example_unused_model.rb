require "dnn"
require "dnn/datasets/iris"
# If you use numo/linalg then please uncomment out.
# require "numo/linalg/autoloader"

include DNN::Layers
include DNN::Optimizers
include DNN::Losses

x, y = DNN::Iris.load(true)
x_train, y_train = x[0...100, true], y[0...100]
x_test, y_test = x[100...150, true], y[100...150]

y_train = DNN::Utils.to_categorical(y_train, 3, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 3, Numo::SFloat)

epochs = 1000
batch_size = 32

opt = Adam.new
lf = SoftmaxCrossEntropy.new

train_iter = DNN::Iterator.new(x_train, y_train)
test_iter = DNN::Iterator.new(x_test, y_test, random: false)

w1 = DNN::Variable.new(Numo::SFloat.new(4, 16).rand_norm)
b1 = DNN::Variable.new(Numo::SFloat.zeros(16))
w2 = DNN::Variable.new(Numo::SFloat.new(16, 3).rand_norm)
b2 = DNN::Variable.new(Numo::SFloat.zeros(3))

net = -> x, y do
  h = Dot.(x, w1) + b1
  h = Sigmoid.(h)
  out = Dot.(h, w2) + b2
  out
end

(1..epochs).each do |epoch|
  train_iter.foreach(batch_size) do |x_batch, y_batch, step|
    x = DNN::Tensor.convert(x_batch)
    y = DNN::Tensor.convert(y_batch)
    out = net.(x, y)
    loss = lf.(out, y)
    loss.link.backward
    puts "epoch: #{epoch}, step: #{step}, loss = #{loss.data.to_f}"
    opt.update([w1, b1, w2, b2])
  end
end

correct = 0
test_iter.foreach(batch_size) do |x_batch, y_batch, step|
  x = DNN::Tensor.convert(x_batch)
  y = DNN::Tensor.convert(y_batch)
  out = net.(x, y)
  correct += out.data.max_index(axis: 1).eq(y_batch.max_index(axis: 1)).count
end
puts "correct = #{correct}"
