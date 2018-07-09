require "numo/narray"
require "dnn/ext/image_io/image_io_ext"

module DNN
  module ImageIO
    private_class_method :_read
    private_class_method :_write_bmp
    private_class_method :_write_png
    private_class_method :_write_jpg

    def self.read(file_name)
      raise ImageIO::ReadError.new("#{file_name} is not found.") unless File.exist?(file_name)
      _read(file_name)
    end

    def self.write(file_name, nary, quality: 100)
      case file_name
      when /\.png$/
        _write_png(file_name, nary)
      when /\.bmp$/
        _write_bmp(file_name, nary)
      when /\.jpg$/
        _write_jpg(file_name, nary, quality)
      end
    rescue => ex
      raise ImageIO::WriteError.new(ex.message)
    end
  end

  class ImageIO::Error < StandardError; end

  class ImageIO::ReadError < ImageIO::Error; end

  class ImageIO::WriteError < ImageIO::Error; end
end
