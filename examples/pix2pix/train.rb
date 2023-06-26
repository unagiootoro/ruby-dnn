# This example converts a CIFAR10 grayscale image to a color image.

require "dnn"
require "dnn/datasets/cifar10"
require_relative "dcgan"

include DNN::Optimizers
include DNN::Losses

class Pix2pixIterator < DNN::BaseIterator
  def initialize(x, y)
    super(last_round_down: true)
    @iter1 = DNN::Iterator.new(x, y, last_round_down: true)
    @iter2 = DNN::Iterator.new(x, y, last_round_down: true)
    @num_datas = x.shape[0]
  end

  def next_batch(batch_size)
    x1, y1 = @iter1.next_batch(batch_size)
    x2, y2 = @iter2.next_batch(batch_size)
    [x1, y1, x2, y2]
  end

  def reset
    @iter1.reset
    @iter2.reset
  end
end

class Pix2pixTrainer < DNN::Trainer
  def on_train_step(x_batch1, y_batch1, x_batch2, y_batch2)
    gen = @model.gen
    dis = @model.dis

    images = gen.predict(x_batch1)
    y_real = Numo::SFloat.ones(@train_batch_size, 1)
    y_fake = Numo::SFloat.zeros(@train_batch_size, 1)
    dis.enable_training
    dis_loss = dis.train_on_batch([x_batch1, y_batch1], y_real)
    dis_loss += dis.train_on_batch([x_batch1, images], y_fake)

    dcgan_loss = @model.train_on_batch(x_batch2, [y_batch2, y_real])

    { dis_loss: dis_loss, dcgan_loss: dcgan_loss}
  end
end

def load_dataset
  x, y = DNN::CIFAR10.load_train
  x_out = Numo::SFloat.cast(x)
  x_in = x_out.mean(axis: 3, keepdims: true)
  x_in = (x_in / 127.5) - 1
  x_out = (x_out / 127.5) - 1
  [x_in, x_out]
end

initial_epoch = 1

epochs = 20
batch_size = 128

if initial_epoch == 1
  gen = Generator.new([32, 32, 1], 32)
  dis = Discriminator.new([32, 32, 1], [32, 32, 3], 32)
  dcgan = DCGAN.new(gen, dis)
  gen.setup(Adam.new(alpha: 0.0002, beta1: 0.5), MeanAbsoluteError.new)
  dis.setup(Adam.new(alpha: 0.00001, beta1: 0.1), SigmoidCrossEntropy.new)
  dcgan.setup(Adam.new(alpha: 0.0002, beta1: 0.5),
              [MeanAbsoluteError.new, SigmoidCrossEntropy.new], loss_weights: [10, 1])
else
  dcgan = DCGAN.load("trained/dcgan_model_epoch#{initial_epoch - 1}.marshal")
  gen = dcgan.gen
  dis = dcgan.dis
end

x_in, x_out = load_dataset

trainer = Pix2pixTrainer.new(dcgan)
trainer.fit_by_iterator(
  Pix2pixIterator.new(x_in, x_out),
  epochs,
  initial_epoch: initial_epoch,
  batch_size: batch_size,
  need_accuracy: false
)
