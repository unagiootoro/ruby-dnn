require "zlib"
require "archive/tar/minitar"
require_relative "../../ext/cifar_loader/cifar_loader"
require_relative "downloader"

URL_CIFAR10 = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
DIR_CIFAR10 = "cifar-10-batches-bin"

module DNN
  module CIFAR10
    class DNN_CIFAR10_LoadError < DNN_Error; end

    def self.downloads
      return if Dir.exist?(__dir__ + "/downloads/" + DIR_CIFAR10)
      Downloader.download(URL_CIFAR10)
      cifar10_binary_file_name = __dir__ + "/downloads/" + URL_CIFAR10.match(%r`.+/(.+)`)[1]
      begin
        Zlib::GzipReader.open(cifar10_binary_file_name) do |gz|
          Archive::Tar::Minitar.unpack(gz, __dir__ + "/downloads")
        end
      ensure
        File.unlink(cifar10_binary_file_name)
      end
    end

    def self.load_train
      downloads
      bin = ""
      (1..5).each do |i|
        fname = __dir__ + "/downloads/#{DIR_CIFAR10}/data_batch_#{i}.bin"
        raise DNN_CIFAR10_LoadError.new(%`file "#{fname}" is not found.`) unless File.exist?(fname)
        bin << File.binread(fname)
      end
      x_bin, y_bin = CIFAR10.load_binary(bin, 50000)
      x_train = Numo::UInt8.from_binary(x_bin).reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).clone
      y_train = Numo::UInt8.from_binary(y_bin)
      [x_train, y_train]
    end

    def self.load_test
      downloads
      fname = __dir__ + "/downloads/#{DIR_CIFAR10}/test_batch.bin"
      raise DNN_CIFAR10_LoadError.new(%`file "#{fname}" is not found.`) unless File.exist?(fname)
      bin = File.binread(fname)
      x_bin, y_bin = CIFAR10.load_binary(bin, 10000)
      x_test = Numo::UInt8.from_binary(x_bin).reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).clone
      y_test = Numo::UInt8.from_binary(y_bin)
      [x_test, y_test]
    end
  end
end
