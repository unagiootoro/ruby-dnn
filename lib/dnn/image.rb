require "numo/narray"
require_relative "../../ext/rb_stb_image/rb_stb_image"

module DNN
  module Image
    class ImageError < StandardError; end

    class ImageReadError < ImageError; end

    class ImageWriteError < ImageError; end

    class ImageShapeError < ImageError; end

    def self.read(file_name)
      raise ImageReadError.new("#{file_name} is not found.") unless File.exist?(file_name)
      bin, w, h, n = Stb.stbi_load(file_name, 3)
      raise ImageReadError.new("#{file_name} load failed.") if bin == ""
      img = Numo::UInt8.from_binary(bin)
      img.reshape(h, w, 3)
    end

    def self.write(file_name, img, quality: 100)
      img_check(img)
      match_data = file_name.match(%r`(.*)/.+$`)
      if match_data
        dir_name = match_data[1]
        Dir.mkdir(dir_name) unless Dir.exist?(dir_name)
      end
      if img.shape[2] == 1
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

    def self.resize(img, out_height, out_width)
      in_height, in_width, ch = *img.shape
      out_bin, res = Stb.stbir_resize_uint8(img.to_binary, in_width, in_height, 0, out_width, out_height, 0, ch)
      img2 = Numo::UInt8.from_binary(out_bin).reshape(out_height, out_width, ch)
      img2
    end

    def self.trim(img, y, x, height, width)
      img_check(img)
      img[y...(y + height), x...(x + width), true].clone
    end

    def self.gray_scale(img)
      img_check(img)
      x = Numo::SFloat.cast(img)
      x = x.mean(axis: 2, keepdims: true)
      Numo::UInt8.cast(x)
    end

    private_class_method def self.img_check(img)
      raise TypeError.new("img is not an instance of the Numo::UInt8 class.") unless img.is_a?(Numo::UInt8)
      if img.shape.length != 3
        raise ImageShapeError.new("img shape is #{img.shape}. But img shape must be 3 dimensional.")
      elsif img.shape[2] != 1 && img.shape[2] != 3
        raise ImageShapeError.new("img channel is #{img.shape[2]}. But img channel must be 1 or 3.")
      end
    end
  end
end
