require "zlib"
require "archive/tar/minitar"
require_relative "../../../ext/cifar_loader/cifar_loader"
require_relative "downloader"

URL_CIFAR100 = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
DIR_CIFAR100 = "cifar-100-binary"

module DNN
  module CIFAR100
    class DNN_CIFAR100_LoadError < DNN_Error; end

    def self.downloads
      return if Dir.exist?(__dir__ + "/downloads/" + DIR_CIFAR100)
      Downloader.download(URL_CIFAR100)
      cifar100_binary_file_name = __dir__ + "/downloads/" + URL_CIFAR100.match(%r`.+/(.+)`)[1]
      begin
        Zlib::GzipReader.open(cifar100_binary_file_name) do |gz|
          Archive::Tar::Minitar.unpack(gz, __dir__ + "/downloads")
        end
      ensure
        File.unlink(cifar100_binary_file_name)
      end
    end

    def self.load_train
      downloads
      bin = ""
      fname = __dir__ + "/downloads/#{DIR_CIFAR100}/train.bin"
      raise DNN_CIFAR100_LoadError.new(%`file "#{fname}" is not found.`) unless File.exist?(fname)
      bin << File.binread(fname)
      x_bin, y_bin = CIFAR100.load_binary(bin, 50000)
      x_train = Numo::UInt8.from_binary(x_bin).reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).clone
      y_train = Numo::UInt8.from_binary(y_bin).reshape(50000, 2)
      [x_train, y_train]
    end

    def self.load_test
      downloads
      fname = __dir__ + "/downloads/#{DIR_CIFAR100}/test.bin"
      raise DNN_CIFAR100_LoadError.new(%`file "#{fname}" is not found.`) unless File.exist?(fname)
      bin = File.binread(fname)
      x_bin, y_bin = CIFAR100.load_binary(bin, 10000)
      x_test = Numo::UInt8.from_binary(x_bin).reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).clone
      y_test = Numo::UInt8.from_binary(y_bin).reshape(10000, 2)
      [x_test, y_test]
    end
  end
end
