require "numo/narray"
require_relative "../../ext/rb_stb_image/rb_stb_image"

module DNN
  module Image
    class ImageError < StandardError; end

    class ImageReadError < ImageError; end

    class ImageWriteError < ImageError; end

    class ImageShapeError < ImageError; end

    RGB = 3
    RGBA = 4

    # Read image from file.
    # @param [String] file_name File name to read.
    # @param [Integer] channel_type Specify channel type of image.
    def self.read(file_name, channel_type = RGB)
      raise ImageReadError.new("#{file_name} is not found.") unless File.exist?(file_name)
      bin, w, h, n = Stb.stbi_load(file_name, channel_type)
      raise ImageReadError.new("#{file_name} load failed.") if bin == ""
      img = Numo::UInt8.from_binary(bin)
      img.reshape(h, w, channel_type)
    end

    # Write image to file.
    # @param [String] file_name File name to write.
    # @param [Numo::UInt8] img Image to write.
    # @param [Integer] quality Image quality when jpeg write.
    def self.write(file_name, img, quality: 100)
      img_check(img)
      match_data = file_name.match(%r`(.*)/.+$`)
      if match_data
        dir_name = match_data[1]
        Dir.mkdir(dir_name) unless Dir.exist?(dir_name)
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
    end

    # Resize the image.
    # @param [Numo::UInt8] img Image to resize.
    # @param [Integer] out_height Image height to resize.
    # @param [Integer] out_width Image width to resize.
    def self.resize(img, out_height, out_width)
      img_check(img)
      in_height, in_width, ch = *img.shape
      out_bin, res = Stb.stbir_resize_uint8(img.to_binary, in_width, in_height, 0, out_width, out_height, 0, ch)
      img2 = Numo::UInt8.from_binary(out_bin).reshape(out_height, out_width, ch)
      img2
    end

    # Trimming the image.
    # @param [Numo::UInt8] img Image to resize.
    # @param [Integer] y The begin y coordinate of the image to trimming.
    # @param [Integer] x The begin x coordinate of the image to trimming.
    # @param [Integer] height Image height to trimming.
    # @param [Integer] width Image height to trimming.
    def self.trim(img, y, x, height, width)
      img_check(img)
      img[y...(y + height), x...(x + width), true].clone
    end

    # Image convert to gray scale.
    # @param [Numo::UInt8] img Image to gray scale.
    def self.to_gray_scale(img)
      img_check(img)
      if img.shape[2] == RGB
        x = Numo::SFloat.cast(img)
        x = x.mean(axis: 2, keepdims: true)
      elsif img.shape[2] == RGBA
        x = Numo::SFloat.cast(img[true, true, 0..2])
        x = x.mean(axis: 2, keepdims: true).concatenate(img[true, true, 3..3], axis: 2)
      end
      Numo::UInt8.cast(x)
    end

    private_class_method def self.img_check(img)
      raise TypeError.new("img: #{img.class} is not an instance of the Numo::UInt8 class.") unless img.is_a?(Numo::UInt8)
      if img.shape.length != 3
        raise ImageShapeError.new("img shape is #{img.shape}. But img shape must be 3 dimensional.")
      elsif !img.shape[2].between?(1, 4)
        raise ImageShapeError.new("img channel is #{img.shape[2]}. But img channel must be 1 or 2 or 3 or 4.")
      end
    end
  end
end
