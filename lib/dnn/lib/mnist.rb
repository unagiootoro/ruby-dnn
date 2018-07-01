require "zlib"
require "dnn/ext/mnist/mnist_ext"

module DNN
  module MNIST
    class MNISTLoadError < StandardError
    end

    private_class_method :_mnist_load_images
    private_class_method :_mnist_load_labels

    def self.load_train
      train_images_file_name = "mnist/train-images-idx3-ubyte.gz"
      train_labels_file_name = "mnist/train-labels-idx1-ubyte.gz"
      unless File.exist?(train_images_file_name)
        raise MNISTLoadError.new(%`file "#{train_images_file_name}" is not found.`)
      end
      unless File.exist?(train_labels_file_name)
        raise MNISTLoadError.new(%`file "#{train_labels_file_name}" is not found.`)
      end
      images = load_images(train_images_file_name)
      labels = load_labels(train_labels_file_name)
      [images, labels]
    end

    def self.load_test
      test_images_file_name = "mnist/t10k-images-idx3-ubyte.gz"
      test_labels_file_name = "mnist/t10k-labels-idx1-ubyte.gz"
      unless File.exist?(test_images_file_name)
        raise MNISTLoadError.new(%`file "#{train_images_file_name}" is not found.`)
      end
      unless File.exist?(test_labels_file_name)
        raise MNISTLoadError.new(%`file "#{train_labels_file_name}" is not found.`)
      end
      images = load_images(test_images_file_name)
      labels = load_labels(test_labels_file_name)
      [images, labels]
    end

    private_class_method

    def self.load_images(file_name)
      images = nil
      Zlib::GzipReader.open(file_name) do |f|
        magic, num_images = f.read(8).unpack("N2")
        rows, cols = f.read(8).unpack("N2")
        images = _mnist_load_images(f.read, num_images, cols, rows)
      end
      images
    end

    def self.load_labels(file_name)
      labels = nil
      Zlib::GzipReader.open(file_name) do |f|
        magic, num_labels = f.read(8).unpack("N2")
        labels = _mnist_load_labels(f.read, num_labels)
      end
      labels
    end
  end
end