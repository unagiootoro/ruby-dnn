require "zlib"
require "archive/tar/minitar"
require_relative "downloader"

URL_CIFAR100 = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
DIR_CIFAR100 = "cifar-100-binary"

module DNN
  module CIFAR100
    class DNN_CIFAR100_LoadError < DNNError; end

    def self.downloads
      return if Dir.exist?(DOWNLOADS_PATH + "/downloads/" + DIR_CIFAR100)
      Downloader.download(URL_CIFAR100)
      cifar100_binary_file_name = DOWNLOADS_PATH + "/downloads/" + URL_CIFAR100.match(%r`.+/(.+)`)[1]
      begin
        Zlib::GzipReader.open(cifar100_binary_file_name) do |gz|
          Archive::Tar::Minitar.unpack(gz, DOWNLOADS_PATH + "/downloads")
        end
      ensure
        File.unlink(cifar100_binary_file_name)
      end
    end

    def self.load_train
      downloads
      bin = ""
      fname = DOWNLOADS_PATH + "/downloads/#{DIR_CIFAR100}/train.bin"
      raise DNN_CIFAR100_LoadError, %`file "#{fname}" is not found.` unless File.exist?(fname)
      bin << File.binread(fname)
      datas = Numo::UInt8.from_binary(bin).reshape(50000, 3074)
      x_train = datas[true, 2...3074]
      x_train = x_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).clone
      y_train = datas[true, 0...2]
      [x_train, y_train]
    end

    def self.load_test
      downloads
      fname = DOWNLOADS_PATH + "/downloads/#{DIR_CIFAR100}/test.bin"
      raise DNN_CIFAR100_LoadError, %`file "#{fname}" is not found.` unless File.exist?(fname)
      bin = File.binread(fname)
      datas = Numo::UInt8.from_binary(bin).reshape(10000, 3074)
      x_test = datas[true, 2...3074]
      x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).clone
      y_test = datas[true, 0...2]
      [x_test, y_test]
    end
  end
end
