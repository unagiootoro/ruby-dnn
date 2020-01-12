require "zlib"
require "archive/tar/minitar"
require_relative "downloader"

URL_CIFAR10 = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
DIR_CIFAR10 = "cifar-10-batches-bin"

module DNN
  module CIFAR10
    class DNN_CIFAR10_LoadError < DNNError; end

    def self.downloads
      return if Dir.exist?(DOWNLOADS_PATH + "/downloads/" + DIR_CIFAR10)
      Downloader.download(URL_CIFAR10)
      cifar10_binary_file_name = DOWNLOADS_PATH + "/downloads/" + URL_CIFAR10.match(%r`.+/(.+)`)[1]
      begin
        Zlib::GzipReader.open(cifar10_binary_file_name) do |gz|
          Archive::Tar::Minitar.unpack(gz, DOWNLOADS_PATH + "/downloads")
        end
      ensure
        File.unlink(cifar10_binary_file_name)
      end
    end

    def self.load_train
      downloads
      bin = ""
      (1..5).each do |i|
        fname = DOWNLOADS_PATH + "/downloads/#{DIR_CIFAR10}/data_batch_#{i}.bin"
        raise DNN_CIFAR10_LoadError, %`file "#{fname}" is not found.` unless File.exist?(fname)
        bin << File.binread(fname)
      end
      datas = Numo::UInt8.from_binary(bin).reshape(50000, 3073)
      x_train = datas[true, 1...3073]
      x_train = x_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).clone
      y_train = datas[true, 0]
      [x_train, y_train]
    end

    def self.load_test
      downloads
      fname = DOWNLOADS_PATH + "/downloads/#{DIR_CIFAR10}/test_batch.bin"
      raise DNN_CIFAR10_LoadError, %`file "#{fname}" is not found.` unless File.exist?(fname)
      bin = File.binread(fname)
      datas = Numo::UInt8.from_binary(bin).reshape(10000, 3073)
      x_test = datas[true, 1...3073]
      x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).clone
      y_test = datas[true, 0]
      [x_test, y_test]
    end
  end
end
