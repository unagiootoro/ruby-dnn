require "dnn"
require "dnn/ext/dataset_loader/dataset_loader"
require "open-uri"
require "zlib"
require "archive/tar/minitar"

URL_CIFAR10 = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_DIR = "cifar-10-batches-bin"

module DNN
  module CIFAR10
    class DNN_CIFAR10_LoadError < DNN_Error; end

    class DNN_CIFAR10_DownloadError < DNN_Error; end

    private_class_method :_cifar10_load

    def self.downloads
      return if Dir.exist?(__dir__ + "/" + CIFAR10_DIR)
      cifar10_binary_file_name = __dir__ + "/" + URL_CIFAR10.match(%r`.+/(.+)`)[1]
      puts "Now downloading..."
      open(URL_CIFAR10, "rb") do |f|
        File.binwrite(cifar10_binary_file_name, f.read)
        begin
          Zlib::GzipReader.open(cifar10_binary_file_name) do |gz|
            Archive::Tar::Minitar::unpack(gz, __dir__)
          end
        ensure
          File.unlink(cifar10_binary_file_name)
        end
      end
      puts "The download has ended."
    rescue => ex
      raise DNN_CIFAR10_DownloadError.new(ex.message)
    end

    def self.load_train
      downloads
      bin = ""
      (1..5).each do |i|
        fname = __dir__ + "/#{CIFAR10_DIR}/data_batch_#{i}.bin"
        raise DNN_CIFAR10_LoadError.new(%`file "#{fname}" is not found.`) unless File.exist?(fname)
        bin << File.binread(fname)
      end
      _cifar10_load(bin, 50000)
    end

    def self.load_test
      downloads
      fname = __dir__ + "/#{CIFAR10_DIR}/test_batch.bin"
      raise DNN_CIFAR10_LoadError.new(%`file "#{fname}" is not found.`) unless File.exist?(fname)
      bin = File.binread(fname)
      _cifar10_load(bin, 10000)
    end
  end
end
