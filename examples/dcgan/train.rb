require "dnn"
require "dnn/datasets/mnist"
require_relative "dcgan"

include DNN::Optimizers
include DNN::Losses
include DNN::Callbacks

class DCGANTrainer < DNN::Trainer
  def on_train_step(x_batch, y_batch, need_accuracy: false)
    batch_size = x_batch.shape[0]
    noise = Numo::SFloat.new(batch_size, 20).rand(-1, 1)
    images = @model.gen.predict(noise)
    y_real = Numo::SFloat.ones(batch_size, 1)
    y_fake = Numo::SFloat.zeros(batch_size, 1)
    @model.dis.enable_training
    dis_loss = @model.dis.train_on_batch(x_batch, y_real)
    dis_loss += @model.dis.train_on_batch(images, y_fake)

    noise = Numo::SFloat.new(batch_size, 20).rand(-1, 1)
    label = Numo::SFloat.cast([1] * batch_size).reshape(batch_size, 1)
    dcgan_loss = @model.train_on_batch(noise, label)

    { dis_loss: dis_loss, dcgan_loss: dcgan_loss }
  end
end

Numo::SFloat.srand(rand(1 << 31))

epochs = 20
batch_size = 128

gen = Generator.new
dis = Discriminator.new
dcgan = DCGAN.new(gen, dis)

dis.setup(Adam.new(alpha: 0.00001, beta1: 0.1), SigmoidCrossEntropy.new)
dcgan.setup(Adam.new(alpha: 0.0002, beta1: 0.5), SigmoidCrossEntropy.new)

x_train, * = DNN::MNIST.load_train
x_train = Numo::SFloat.cast(x_train)
x_train = x_train / 127.5 - 1

iter = DNN::Iterator.new(x_train, x_train, last_round_down: true)

trainer = DCGANTrainer.new(dcgan)
trainer.add_callback(CheckPoint.new(dcgan, "trained/dcgan_model"))
trainer.train_by_iterator(iter, epochs, batch_size: batch_size)
