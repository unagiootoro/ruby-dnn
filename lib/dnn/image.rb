require "numo/narray"
require_relative "../../ext/rb_stb_image/rb_stb_image"

module DNN
  module Image
    def self.read(file_name)
      raise Image::ReadError.new("#{file_name} is not found.") unless File.exist?(file_name)
      bin, w, h, n = Stb.stbi_load(file_name, 3)
      img = Numo::UInt8.from_binary(bin)
      img.reshape(h, w, 3)
    end

    def self.write(file_name, img, quality: 100)
      _write(file_name, img, quality: quality)
    rescue Errno::ENOENT => ex
      dir_name = file_name.match(%r`(.*)/.+$`)[1]
      Dir.mkdir(dir_name)
      _write(file_name, img, quality: quality)
    rescue => ex
      raise Image::WriteError.new(ex.message)
    end

    def self._write(file_name, img, quality: 100)
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
        Stb.stbi_write_png(file_name, w, h, ch, bin, stride_in_bytes)
      when /\.bmp$/i
        Stb.stbi_write_bmp(file_name, w, h, ch, bin)
      when /\.jpg$/i, /\.jpeg/i
        Stb.stbi_write_jpg(file_name, w, h, ch, bin, quality)
      end
    end
  end

  class Image::Error < StandardError; end

  class Image::ReadError < Image::Error; end

  class Image::WriteError < Image::Error; end
end
