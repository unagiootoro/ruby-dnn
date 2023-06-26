require "dnn"
require "dnn/datasets/iris"

include DNN::Layers
include DNN::Optimizers

Fs = DNN::Functions::FunctionSpace

x, y = DNN::Iris.load(true)
x_train, y_train = x[0...100, true], y[0...100]
x_test, y_test = x[100...150, true], y[100...150]

y_train = DNN::Utils.to_categorical(y_train, 3, Numo::SFloat)
y_test = DNN::Utils.to_categorical(y_test, 3, Numo::SFloat)

epochs = 1000
batch_size = 32

opt = Adam.new

train_iter = DNN::Iterator.new(x_train, y_train)
test_iter = DNN::Iterator.new(x_test, y_test, random: false)

w1 = DNN::Variable.new(Numo::SFloat.new(4, 16).rand_norm)
b1 = DNN::Variable.new(Numo::SFloat.zeros(16))
w2 = DNN::Variable.new(Numo::SFloat.new(16, 3).rand_norm)
b2 = DNN::Variable.new(Numo::SFloat.zeros(3))

net = -> x, y do
  h = x.dot(w1) + b1
  h = Fs.sigmoid(h)
  out = h.dot(w2) + b2
  out
end

(1..epochs).each do |epoch|
  train_iter.foreach(batch_size) do |x_batch, y_batch, step|
    x = DNN::Tensor.new(x_batch)
    y = DNN::Tensor.new(y_batch)
    out = net.(x, y)
    loss = Fs.softmax_cross_entropy(out, y)
    loss.link.backward
    puts "epoch: #{epoch}, step: #{step}, loss = #{loss.data.to_f}"
    opt.update([w1, b1, w2, b2])
  end
end

correct = 0
test_iter.foreach(batch_size) do |x_batch, y_batch, step|
  x = DNN::Tensor.new(x_batch)
  y = DNN::Tensor.new(y_batch)
  out = net.(x, y)
  correct += out.data.max_index(axis: 1).eq(y_batch.max_index(axis: 1)).count
end
puts "correct = #{correct}"
