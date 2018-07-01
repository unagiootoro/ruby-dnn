require "dnn"
require "dnn/ext/cifar10/cifar10_ext"

module DNN
  module CIFAR10
    private_class_method :_cifar10_load

    def self.load_train
      bin = ""
      (1..5).each do |i|
        bin << File.binread("#{dir}/data_batch_#{i}.bin")
      end
      _cifar10_load(bin, 50000)
    end

    def self.load_test
      bin = File.binread("#{dir}/test_batch.bin")
      _cifar10_load(bin, 10000)
    end

    def self.dir
      "cifar-10-batches-bin"
    end
  end
end
##30730000