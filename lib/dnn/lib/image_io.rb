require "numo/narray"
require "dnn/ext/rb_stb_image/rb_stb_image"

module DNN
  module ImageIO
    def self.read(file_name)
      raise ImageIO::ReadError.new("#{file_name} is not found.") unless File.exist?(file_name)
      img, = Stb.stbi_load(file_name, 3)
      img
    end

    def self.write(file_name, img, quality: 100)
      img = img.clone
      if img.shape.length == 2
        img = Numo::UInt8[img, img, img].transpose(1, 2, 0).clone
      elsif img.shape[2] == 1
        img = img.shape(img.shape[0], img.shape[1])
        img = Numo::UInt8[img, img, img].transpose(1, 2, 0).clone
      end
      case file_name
      when /\.png$/
        stride_in_bytes = img.shape[0] * img.shape[2]
        Stb.stbi_write_png(file_name, *img.shape, img, stride_in_bytes)
      when /\.bmp$/
        Stb.stbi_write_bmp(file_name, *img.shape, img)
      when /\.jpg$/
        Stb.stbi_write_jpg(file_name, *img.shape, img, quality)
      end
    rescue => ex
      raise ImageIO::WriteError.new(ex.message)
    end
  end

  class ImageIO::Error < StandardError; end

  class ImageIO::ReadError < ImageIO::Error; end

  class ImageIO::WriteError < ImageIO::Error; end
end
