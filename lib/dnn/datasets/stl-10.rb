require "zlib"
require "archive/tar/minitar"
require_relative "downloader"

URL_STL10 = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
DIR_STL10 = "stl10_binary"

module DNN
  module STL10
    class DNN_STL10_LoadError < DNNError; end

    def self.downloads
      return if Dir.exist?(DOWNLOADS_PATH + "/downloads/" + DIR_STL10)
      Downloader.download(URL_STL10)
      stl10_binary_file_name = DOWNLOADS_PATH + "/downloads/" + URL_STL10.match(%r`.+/(.+)`)[1]
      begin
        Zlib::GzipReader.open(stl10_binary_file_name) do |gz|
          Archive::Tar::Minitar.unpack(gz, DOWNLOADS_PATH + "/downloads")
        end
      ensure
        File.unlink(stl10_binary_file_name)
      end
    end

    def self.load_train
      downloads
      x_fname = DOWNLOADS_PATH + "/downloads/#{DIR_STL10}/train_X.bin"
      raise DNN_STL10_LoadError, %`file "#{x_fname}" is not found.` unless File.exist?(x_fname)
      y_fname = DOWNLOADS_PATH + "/downloads/#{DIR_STL10}/train_y.bin"
      raise DNN_STL10_LoadError, %`file "#{y_fname}" is not found.` unless File.exist?(y_fname)
      x_bin = File.binread(x_fname)
      y_bin = File.binread(y_fname)
      x_train = Numo::UInt8.from_binary(x_bin).reshape(5000, 3, 96, 96).transpose(0, 3, 2, 1).clone
      y_train = Numo::UInt8.from_binary(y_bin)
      [x_train, y_train]
    end

    def self.load_test
      downloads
      x_fname = DOWNLOADS_PATH + "/downloads/#{DIR_STL10}/test_X.bin"
      raise DNN_STL10_LoadError, %`file "#{x_fname}" is not found.` unless File.exist?(x_fname)
      y_fname = DOWNLOADS_PATH + "/downloads/#{DIR_STL10}/test_y.bin"
      raise DNN_STL10_LoadError, %`file "#{y_fname}" is not found.` unless File.exist?(y_fname)
      x_bin = File.binread(x_fname)
      y_bin = File.binread(y_fname)
      x_test = Numo::UInt8.from_binary(x_bin).reshape(8000, 3, 96, 96).transpose(0, 3, 2, 1).clone
      y_test = Numo::UInt8.from_binary(y_bin)
      [x_test, y_test]
    end

    def self.load_unlabeled(range = 0...100000)
      raise DNNError, "Range must between 0 and 100000. (But the end is excluded)" unless range.begin >= 0 && range.end <= 100000
      downloads
      x_fname = DOWNLOADS_PATH + "/downloads/#{DIR_STL10}/unlabeled_X.bin"
      raise DNN_STL10_LoadError, %`file "#{x_fname}" is not found.` unless File.exist?(x_fname)
      num_datas = range.end - range.begin
      length = num_datas * 3 * 96 * 96
      ofs = range.begin * 3 * 96 * 96
      x_bin = File.binread(x_fname, length, ofs)
      Numo::UInt8.from_binary(x_bin).reshape(num_datas, 3, 96, 96).transpose(0, 3, 2, 1).clone
    end
  end
end
