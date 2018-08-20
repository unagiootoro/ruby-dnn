require "numo/narray"
require "dnn/ext/rb_stb_image/rb_stb_image"

module DNN
  module ImageIO
    def self.read(file_name)
      raise ImageIO::ReadError.new("#{file_name} is not found.") unless File.exist?(file_name)
      bin, w, h, n = Stb.stbi_load(file_name, 3)
      img = Numo::UInt8.from_binary(bin)
      img.reshape(h, w, 3)
    end

    def self.write(file_name, img, quality: 100)
      if img.shape.length == 2
        img = Numo::UInt8[img, img, img].transpose(1, 2, 0).clone
      elsif img.shape[2] == 1
        img = img.shape(img.shape[0], img.shape[1])
        img = Numo::UInt8[img, img, img].transpose(1, 2, 0).clone
      end
      h, w, ch = img.shape
      bin = img.to_binary
      case file_name
      when /\.png$/i
        stride_in_bytes = w * ch
        Stb.stbi_write_png(file_name, w, h, ch, bin, stride_in_bytes)
      when /\.bmp$/i
        Stb.stbi_write_bmp(file_name, w, h, ch, bin)
      when /\.jpg$/i, /\.jpeg/i
        Stb.stbi_write_jpg(file_name, w, h, ch, bin, quality)
      end
    rescue => ex
      raise ImageIO::WriteError.new(ex.message)
    end
  end

  class ImageIO::Error < StandardError; end

  class ImageIO::ReadError < ImageIO::Error; end

  class ImageIO::WriteError < ImageIO::Error; end
end
