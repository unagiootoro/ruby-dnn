require "open-uri"
require "zlib"
require "dnn/core/error"
require "dnn/ext/dataset_loader/dataset_loader"

module DNN
  module MNIST
    class DNN_MNIST_LoadError < DNN_Error; end

    class DNN_MNIST_DownloadError < DNN_Error; end

    URL_TRAIN_IMAGES = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    URL_TRAIN_LABELS = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    URL_TEST_IMAGES = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    URL_TEST_LABELS = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

    private_class_method :_mnist_load_images
    private_class_method :_mnist_load_labels

    def self.downloads
      return if Dir.exist?(mnist_dir)
      Dir.mkdir(mnist_dir)
      puts "Now downloading..."
      download(URL_TRAIN_IMAGES)
      download(URL_TRAIN_LABELS)
      download(URL_TEST_IMAGES)
      download(URL_TEST_LABELS)
      puts "The download has ended."
    end

    def self.load_train
      downloads
      train_images_file_name = url_to_file_name(URL_TRAIN_IMAGES)
      train_labels_file_name = url_to_file_name(URL_TRAIN_LABELS)
      unless File.exist?(train_images_file_name)
        raise DNN_MNIST_LoadError.new(%`file "#{train_images_file_name}" is not found.`)
      end
      unless File.exist?(train_labels_file_name)
        raise DNN_MNIST_LoadError.new(%`file "#{train_labels_file_name}" is not found.`)
      end
      images = load_images(train_images_file_name)
      labels = load_labels(train_labels_file_name)
      [images, labels]
    end

    def self.load_test
      downloads
      test_images_file_name = url_to_file_name(URL_TEST_IMAGES)
      test_labels_file_name = url_to_file_name(URL_TEST_LABELS)
      unless File.exist?(test_images_file_name)
        raise DNN_MNIST_LoadError.new(%`file "#{train_images_file_name}" is not found.`)
      end
      unless File.exist?(test_labels_file_name)
        raise DNN_MNIST_LoadError.new(%`file "#{train_labels_file_name}" is not found.`)
      end
      images = load_images(test_images_file_name)
      labels = load_labels(test_labels_file_name)
      [images, labels]
    end

    private_class_method

    def self.download(url)
      open(url, "rb") do |f|
        File.binwrite(url_to_file_name(url), f.read)
      end
    rescue => ex
      raise DNN_MNIST_DownloadError.new(ex.message)
    end

    def self.load_images(file_name)
      images = nil
      Zlib::GzipReader.open(file_name) do |f|
        magic, num_images = f.read(8).unpack("N2")
        rows, cols = f.read(8).unpack("N2")
        images = Numo::UInt8.from_binary(f.read)
        images = images.reshape(num_images, cols, rows)
      end
      images
    end

    def self.load_labels(file_name)
      labels = nil
      Zlib::GzipReader.open(file_name) do |f|
        magic, num_labels = f.read(8).unpack("N2")
        labels = Numo::UInt8.from_binary(f.read)
      end
      labels
    end

    def self.mnist_dir
      __dir__ + "/mnist"
    end

    def self.url_to_file_name(url)
      mnist_dir + "/" + url.match(%r`.+/(.+)$`)[1]
    end
  end
end