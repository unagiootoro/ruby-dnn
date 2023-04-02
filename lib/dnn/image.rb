require "numo/narray"
require_relative "../rb_stb_image"

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
      raise ImageReadError, "#{file_name} is not found." unless File.exist?(file_name)
      bin, w, h, n = Stb.stbi_load(file_name, channel_type)
      raise ImageReadError, "#{file_name} load failed." if bin == ""
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
      match_data = file_name.match(/\.(\w+)$/i)
      if match_data
        ext = match_data[1]
      else
        raise ImageWriteError, "File name has not extension."
      end
      case ext
      when "png"
        stride_in_bytes = w * ch
        res = Stb.stbi_write_png(file_name, w, h, ch, img.to_binary, stride_in_bytes)
      when "bmp"
        res = Stb.stbi_write_bmp(file_name, w, h, ch, img.to_binary)
      when "jpg", "jpeg"
        raise TypeError, "quality:#{quality.class} is not an instance of Integer class." unless quality.is_a?(Integer)
        raise ArgumentError, "quality should be between 1 and 100." unless quality.between?(1, 100)
        res = Stb.stbi_write_jpg(file_name, w, h, ch, img.to_binary, quality)
      when "hdr"
        float_img = Numo::SFloat.cast(img) / 255
        res = Stb.stbi_write_hdr(file_name, w, h, ch, float_img.to_binary)
      when "tga"
        res = Stb.stbi_write_tga(file_name, w, h, ch, img.to_binary)
      else
        raise ImageWriteError, "Extension '#{ext}' is not support."
      end
      raise ImageWriteError, "Image write failed." if res == 0
    end

    # Create an image from binary.
    # @param [String] bin binary data.
    # @param [Integer] height Image height.
    # @param [Integer] width Image width.
    # @param [Integer] channel Image channel.
    def self.from_binary(bin, height, width, channel = DNN::Image::RGB)
      expected_size = height * width * channel
      unless bin.size == expected_size
        raise ImageError, "binary size is #{bin.size}, but expected binary size is #{expected_size}"
      end
      Numo::UInt8.from_binary(bin).reshape(height, width, channel)
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

    # Image convert image channel to RGB.
    # @param [Numo::UInt8] img Image to RGB.
    def self.to_rgb(img)
      img_check(img)
      case img.shape[2]
      when 1
        return img.concatenate(img, axis: 2).concatenate(img, axis: 2)
      when 2
        img = img[true, true, 0...1]
        return img.concatenate(img, axis: 2).concatenate(img, axis: 2)
      when 4
        return img[true, true, 0...3].clone
      end
      img
    end

    # Image convert image channel to RGBA.
    # @param [Numo::UInt8] img Image to RGBA.
    def self.to_rgba(img)
      img_check(img)
      case img.shape[2]
      when 1
        alpha = Numo::UInt8.new(*img.shape[0..1], 1).fill(255)
        return img.concatenate(img, axis: 2).concatenate(img, axis: 2).concatenate(alpha, axis: 2)
      when 2
        alpha = img[true, true, 1...2]
        img = img[true, true, 0...1]
        return img.concatenate(img, axis: 2).concatenate(img, axis: 2).concatenate(alpha, axis: 2)
      when 3
        alpha = Numo::UInt8.new(*img.shape[0..1], 1).fill(255)
        return img.concatenate(alpha, axis: 2)
      end
      img
    end

    private_class_method def self.img_check(img)
      raise TypeError, "img: #{img.class} is not an instance of the Numo::UInt8 class." unless img.is_a?(Numo::UInt8)
      if img.shape.length != 3
        raise ImageShapeError, "img shape is #{img.shape}. But img shape must be 3 dimensional."
      elsif !img.shape[2].between?(1, 4)
        raise ImageShapeError, "img channel is #{img.shape[2]}. But img channel must be 1 or 2 or 3 or 4."
      end
    end
  end
end
