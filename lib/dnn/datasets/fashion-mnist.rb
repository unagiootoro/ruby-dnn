require "zlib"
require_relative "../core/error"
require_relative "downloader"
require_relative "mnist"

module DNN
  module FashionMNIST
    class DNN_MNIST_LoadError < DNNError; end

    URL_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    TRAIN_IMAGES_FILE_NAME = "train-images-idx3-ubyte.gz"
    TRAIN_LABELS_FILE_NAME = "train-labels-idx1-ubyte.gz"
    TEST_IMAGES_FILE_NAME = "t10k-images-idx3-ubyte.gz"
    TEST_LABELS_FILE_NAME = "t10k-labels-idx1-ubyte.gz"

    URL_TRAIN_IMAGES = URL_BASE + TRAIN_IMAGES_FILE_NAME
    URL_TRAIN_LABELS = URL_BASE + TRAIN_LABELS_FILE_NAME
    URL_TEST_IMAGES = URL_BASE + TEST_IMAGES_FILE_NAME
    URL_TEST_LABELS = URL_BASE + TEST_LABELS_FILE_NAME

    def self.downloads
      Dir.mkdir("#{DOWNLOADS_PATH}/downloads") unless Dir.exist?("#{DOWNLOADS_PATH}/downloads")
      Dir.mkdir(mnist_dir) unless Dir.exist?(mnist_dir)
      Downloader.download(URL_TRAIN_IMAGES, mnist_dir) unless File.exist?(get_file_path(TRAIN_IMAGES_FILE_NAME))
      Downloader.download(URL_TRAIN_LABELS, mnist_dir) unless File.exist?(get_file_path(TRAIN_LABELS_FILE_NAME))
      Downloader.download(URL_TEST_IMAGES, mnist_dir) unless File.exist?(get_file_path(TEST_IMAGES_FILE_NAME))
      Downloader.download(URL_TEST_LABELS, mnist_dir) unless File.exist?(get_file_path(TEST_LABELS_FILE_NAME))
    end

    def self.load_train
      downloads
      train_images_file_path = get_file_path(TRAIN_IMAGES_FILE_NAME)
      train_labels_file_path = get_file_path(TRAIN_LABELS_FILE_NAME)
      raise DNN_MNIST_LoadError, %`file "#{train_images_file_path}" is not found.` unless File.exist?(train_images_file_path)
      raise DNN_MNIST_LoadError, %`file "#{train_labels_file_path}" is not found.` unless File.exist?(train_labels_file_path)
      images = load_images(train_images_file_path)
      labels = load_labels(train_labels_file_path)
      [images, labels]
    end

    def self.load_test
      downloads
      test_images_file_path = get_file_path(TEST_IMAGES_FILE_NAME)
      test_labels_file_path = get_file_path(TEST_LABELS_FILE_NAME)
      raise DNN_MNIST_LoadError, %`file "#{test_images_file_path}" is not found.` unless File.exist?(test_images_file_path)
      raise DNN_MNIST_LoadError, %`file "#{test_labels_file_path}" is not found.` unless File.exist?(test_labels_file_path)
      images = load_images(test_images_file_path)
      labels = load_labels(test_labels_file_path)
      [images, labels]
    end

    private_class_method def self.load_images(file_name)
      images = nil
      Zlib::GzipReader.open(file_name) do |f|
        magic, num_images = f.read(8).unpack("N2")
        rows, cols = f.read(8).unpack("N2")
        images = Numo::UInt8.from_binary(f.read)
        images = images.reshape(num_images, cols, rows, 1)
      end
      images
    end

    private_class_method def self.load_labels(file_name)
      labels = nil
      Zlib::GzipReader.open(file_name) do |f|
        magic, num_labels = f.read(8).unpack("N2")
        labels = Numo::UInt8.from_binary(f.read)
      end
      labels
    end

    private_class_method def self.mnist_dir
      "#{DOWNLOADS_PATH}/downloads/fashion-mnist"
    end

    private_class_method def self.get_file_path(file_name)
      mnist_dir + "/" + file_name
    end
  end
end
