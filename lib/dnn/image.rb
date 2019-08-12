require "numo/narray"
require_relative "../../ext/rb_stb_image/rb_stb_image"

module DNN
  module Image
    class ImageError < StandardError; end

    class ImageReadError < ImageError; end

    class ImageWriteError < ImageError; end

    def self.read(file_name)
      raise ImageReadError.new("#{file_name} is not found.") unless File.exist?(file_name)
      bin, w, h, n = Stb.stbi_load(file_name, 3)
      img = Numo::UInt8.from_binary(bin)
      img.reshape(h, w, 3)
    end

    def self.write(file_name, img, quality: 100)
      match_data = file_name.match(%r`(.*)/.+$`)
      if match_data
        dir_name = match_data[1]
        Dir.mkdir(dir_name) unless Dir.exist?(dir_name)
      end
      if img.shape.length == 2
        img = Numo::UInt8[img, img, img].transpose(1, 2, 0).clone
      elsif img.shape[2] == 1
        img = img.reshape(img.shape[0], img.shape[1])
        img = Numo::UInt8[img, img, img].transpose(1, 2, 0).clone
      end
      h, w, ch = img.shape
      bin = img.to_binary
      case file_name
      when /\.png$/i
        stride_in_bytes = w * ch
        res = Stb.stbi_write_png(file_name, w, h, ch, bin, stride_in_bytes)
      when /\.bmp$/i
        res = Stb.stbi_write_bmp(file_name, w, h, ch, bin)
      when /\.jpg$/i, /\.jpeg/i
        res = Stb.stbi_write_jpg(file_name, w, h, ch, bin, quality)
      end
      raise ImageWriteError.new("Image write failed.") if res == 0
    rescue => e
      raise ImageWriteError.new(e.message)
    end
  end
end
