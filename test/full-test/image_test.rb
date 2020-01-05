require "test_helper"
require "dnn/image"

class TestImage < MiniTest::Unit::TestCase
  def test_read
    img = DNN::Image.read("test/full-test/test_cifar10.png")
    assert_equal [32, 32, 3], img.shape
    assert_equal [200, 210, 217], img[15, 15, true].to_a
  end

  # Incorrect files cannot be read.
  def test_read2
    assert_raises DNN::Image::ImageReadError do
      img = DNN::Image.read("lib/dnn/version.rb")
    end
  end

  # Unknown files cannot be read.
  def test_read3
    assert_raises DNN::Image::ImageReadError do
      img = DNN::Image.read("nothing")
    end
  end

  # Successful png writing.
  def test_write
    write_path = "test/full-test/test_cifar10_tmp.png"
    write_img = DNN::Image.read("test/full-test/test_cifar10.png")
    DNN::Image.write(write_path, write_img)
    read_img = DNN::Image.read(write_path)
    File.unlink(write_path)
    assert_equal [32, 32, 3], read_img.shape
    assert_equal [200, 210, 217], read_img[15, 15, true].to_a
  end

  # Successful jpg writing.
  def test_write2
    write_path = "test/full-test/test_cifar10_tmp.jpg"
    write_img = DNN::Image.read("test/full-test/test_cifar10.png")
    DNN::Image.write(write_path, write_img)
    read_img = DNN::Image.read(write_path)
    File.unlink(write_path)
    assert_equal [32, 32, 3], read_img.shape
    # Because of jpg, pixel cannot be read completely.
    assert_equal [201, 209, 219], read_img[15, 15, true].to_a
  end

  # Successful bmp writing.
  def test_write3
    write_path = "test/full-test/test_cifar10_tmp.bmp"
    write_img = DNN::Image.read("test/full-test/test_cifar10.png")
    DNN::Image.write(write_path, write_img)
    read_img = DNN::Image.read(write_path)
    File.unlink(write_path)
    assert_equal [32, 32, 3], read_img.shape
    assert_equal [200, 210, 217], read_img[15, 15, true].to_a
  end

  # Successful hdr writing.
  def test_write4
    write_path = "test/full-test/test_cifar10_tmp.hdr"
    write_img = DNN::Image.read("test/full-test/test_cifar10.png")
    DNN::Image.write(write_path, write_img)
    read_img = DNN::Image.read(write_path)
    File.unlink(write_path)
    assert_equal [32, 32, 3], read_img.shape
    assert_equal [228, 233, 237], read_img[15, 15, true].to_a
  end

  # Successful tga writing.
  def test_write5
    write_path = "test/full-test/test_cifar10_tmp.tga"
    write_img = DNN::Image.read("test/full-test/test_cifar10.png")
    DNN::Image.write(write_path, write_img)
    read_img = DNN::Image.read(write_path)
    File.unlink(write_path)
    assert_equal [32, 32, 3], read_img.shape
    assert_equal [200, 210, 217], read_img[15, 15, true].to_a
  end

  def test_resize
    img = DNN::Image.read("test/full-test/test_cifar10.png")
    img = DNN::Image.resize(img, 64, 64)
    assert_equal [64, 64, 3], img.shape
    assert_equal [188, 198, 202], img[31, 31, true].to_a
  end

  def test_trim
    img = DNN::Image.read("test/full-test/test_cifar10.png")
    img = DNN::Image.trim(img, 8, 8, 16, 16)
    assert_equal [16, 16, 3], img.shape
    assert_equal [200, 210, 217], img[7, 7, true].to_a
  end
end
