require "test_helper"

include DNN
include Layers
include Activations
include Optimizers
include Losses

class TestConv2DModule < MiniTest::Unit::TestCase
  include Conv2DModule

  # im2col test.
  def test_im2col
    img = Numo::SFloat.cast([[
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ],
      [
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32],
      ]
    ]]).transpose(0, 2, 3, 1)
    expected_col = Numo::SFloat[
      [1, 17, 2, 18, 3, 19, 5, 21, 6, 22, 7, 23, 9, 25, 10, 26, 11, 27],
      [2, 18, 3, 19, 4, 20, 6, 22, 7, 23, 8, 24, 10, 26, 11, 27, 12, 28],
      [5, 21, 6, 22, 7, 23, 9, 25, 10, 26, 11, 27, 13, 29, 14, 30, 15, 31],
      [6, 22, 7, 23, 8, 24, 10, 26, 11, 27, 12, 28, 14, 30, 15, 31, 16, 32],
    ]
    col = im2col(img, 2, 2, 3, 3, [1, 1])
    assert_equal expected_col.round(4), col.round(4)
  end

  # im2col strides test.
  def test_im2col2
    img = Numo::SFloat.new(1, 2, 4, 4).seq(1).transpose(0, 2, 3, 1)
    expected_col = Numo::SFloat[
      [1, 17, 2, 18, 5, 21, 6, 22],
      [3, 19, 4, 20, 7, 23, 8, 24],
      [9, 25, 10, 26, 13, 29, 14, 30],
      [11, 27, 12, 28, 15, 31, 16, 32],
    ]
    col = im2col(img, 2, 2, 2, 2, [2, 2])
    assert_equal expected_col.round(4), col.round(4)
  end

  # col2im test.
  def test_col2im
    img_shape = [1, 4, 4, 2]
    col = Numo::SFloat[
      [1, 17, 2, 18, 3, 19, 5, 21, 6, 22, 7, 23, 9, 25, 10, 26, 11, 27],
      [2, 18, 3, 19, 4, 20, 6, 22, 7, 23, 8, 24, 10, 26, 11, 27, 12, 28],
      [5, 21, 6, 22, 7, 23, 9, 25, 10, 26, 11, 27, 13, 29, 14, 30, 15, 31],
      [6, 22, 7, 23, 8, 24, 10, 26, 11, 27, 12, 28, 14, 30, 15, 31, 16, 32],
    ]
    expected_img = Numo::SFloat.cast([[
      [
        [1*1, 2*2, 3*2, 4*1],
        [5*2, 6*4, 7*4, 8*2],
        [9*2, 10*4, 11*4, 12*2],
        [13*1, 14*2, 15*2, 16*1],
      ],
      [
        [17*1, 18*2, 19*2, 20*1],
        [21*2, 22*4, 23*4, 24*2],
        [25*2, 26*4, 27*4, 28*2],
        [29*1, 30*2, 31*2, 32*1],
      ]
    ]]).transpose(0, 2, 3, 1)
    img = col2im(col, img_shape, 2, 2, 3, 3, [1, 1])
    assert_equal expected_img.round(4), img.round(4)
  end

  # col2im stride test.
  def test_col2im2
    img_shape = [1, 4, 4, 2]
    col = Numo::SFloat[
      [1, 17, 2, 18, 5, 21, 6, 22],
      [3, 19, 4, 20, 7, 23, 8, 24],
      [9, 25, 10, 26, 13, 29, 14, 30],
      [11, 27, 12, 28, 15, 31, 16, 32],
    ]
    expected_img = Numo::SFloat.new(1, 2, 4, 4).seq(1).transpose(0, 2, 3, 1)
    img = col2im(col, img_shape, 2, 2, 2, 2, [2, 2])
    assert_equal expected_img.round(4), img.round(4)
  end

  def test_padding
    img = Numo::SFloat.new(1, 2, 4, 4).seq(1).transpose(0, 2, 3, 1)
    expected_img = Numo::SFloat.cast([[
      [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 0],
        [0, 5, 6, 7, 8, 0],
        [0, 9, 10, 11, 12, 0],
        [0, 13, 14, 15, 16, 0],
        [0, 0, 0, 0, 0, 0],
      ],
      [
        [0, 0, 0, 0, 0, 0],
        [0, 17, 18, 19, 20, 0],
        [0, 21, 22, 23, 24, 0],
        [0, 25, 26, 27, 28, 0],
        [0, 29, 30, 31, 32, 0],
        [0, 0, 0, 0, 0, 0],
      ]
    ]]).transpose(0, 2, 3, 1)
    assert_equal expected_img, padding(img, [2, 2])
  end

  def test_back_padding
    img = Numo::SFloat.cast([[
      [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 0],
        [0, 5, 6, 7, 8, 0],
        [0, 9, 10, 11, 12, 0],
        [0, 13, 14, 15, 16, 0],
        [0, 0, 0, 0, 0, 0],
      ],
      [
        [0, 0, 0, 0, 0, 0],
        [0, 17, 18, 19, 20, 0],
        [0, 21, 22, 23, 24, 0],
        [0, 25, 26, 27, 28, 0],
        [0, 29, 30, 31, 32, 0],
        [0, 0, 0, 0, 0, 0],
      ]
    ]]).transpose(0, 2, 3, 1)
    expected_img = Numo::SFloat.new(1, 2, 4, 4).seq(1).transpose(0, 2, 3, 1)
    assert_equal expected_img, back_padding(img, [2, 2])
  end

  def test_out_size
    assert_equal [29, 14], out_size(32, 32, 4, 5, [1, 2])
  end

  def test_out_size2
    assert_equal [32, 32], out_size(32, 32, 1, 1, [1, 1])
  end

  def test_padding_size
    assert_equal [3, 2], padding_size(32, 32, 29, 14, [1, 2])
  end
  
  def test_padding_size2
    assert_equal [0, 1], padding_size(32, 32, 32, 10, [1, 3])
  end
end


class TestConv2D < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      class: "DNN::Layers::Conv2D",
      num_filters: 16,
      filter_size: [3, 3],
      weight_initializer: RandomNormal.new.to_hash,
      bias_initializer: Zeros.new.to_hash,
      strides: [2, 2],
      padding: true,
      l1_lambda: 0,
      l2_lambda: 0,
      use_bias: true,
    }
    conv2d = Conv2D.load_hash(hash)
    assert_equal 16, conv2d.num_filters
    assert_equal [3, 3], conv2d.filter_size
    assert_equal [2, 2], conv2d.strides
  end

  def test_initialize
    conv2d = Conv2D.new(16, 3)
    assert_equal [3, 3], conv2d.filter_size
  end

  def test_initialize2
    conv2d = Conv2D.new(16, 3, strides: 2)
    assert_equal [2, 2], conv2d.strides
  end

  def test_build
    conv2d = Conv2D.new(16, [4, 5], strides: [1, 2])
    conv2d.build([32, 32, 3])
    assert_equal [29, 14], conv2d.instance_variable_get(:@out_size)
  end

  def test_build2
    conv2d = Conv2D.new(16, [4, 5], strides: [1, 2], padding: true)
    conv2d.build([32, 32, 3])
    assert_equal [32, 16], conv2d.instance_variable_get(:@out_size)
  end

  def test_forward
    x = Numo::SFloat.new(1, 32, 32, 3).seq
    conv2d = Conv2D.new(16, 5)
    conv2d.build([32, 32, 3])
    assert_equal [1, 28, 28, 16], conv2d.forward(x).shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 32, 32, 3).seq
    dout = Numo::SFloat.new(1, 28, 28, 16).seq
    conv2d = Conv2D.new(16, 5)
    conv2d.build([32, 32, 3])
    conv2d.forward(x)
    assert_equal [1, 32, 32, 3], conv2d.backward(dout).shape
  end

  def test_backward2
    x = Numo::SFloat.new(1, 32, 32, 3).seq
    dout = Numo::SFloat.new(1, 28, 28, 16).seq
    conv2d = Conv2D.new(16, 5, use_bias: false)
    conv2d.build([32, 32, 3])
    conv2d.forward(x)
    conv2d.backward(dout)
    assert_nil conv2d.params[:bias]
  end

  def test_output_shape
    conv2d = Conv2D.new(16, [4, 5], strides: [1, 2])
    conv2d.build([32, 32, 3])
    assert_equal [29, 14, 16], conv2d.output_shape
  end

  def test_filters
    conv2d = Conv2D.new(16, [4, 5])
    conv2d.build([32, 32, 3])
    assert_equal [4, 5, 3, 16], conv2d.filters.shape
  end

  def test_filters_set
    conv2d = Conv2D.new(16, [4, 5])
    conv2d.build([32, 32, 3])
    conv2d.filters = Xumo::SFloat.zeros(4, 5, 3, 16)
    assert_equal [4 * 5 * 3, 16], conv2d.params[:weight].data.shape
  end

  def test_filters_set2
    conv2d = Conv2D.new(16, [4, 5])
    conv2d.build([32, 32, 3])
    expected = conv2d.params[:weight].data
    conv2d.filters = expected
    assert_equal expected, conv2d.params[:weight].data
  end

  def test_to_hash
    conv2d = Conv2D.new(16, 5, strides: 2, padding: true, l1_lambda: 0.1, l2_lambda: 0.2)
    expected_hash = {
      class: "DNN::Layers::Conv2D",
      num_filters: 16,
      filter_size: [5, 5],
      weight_initializer: conv2d.weight_initializer.to_hash,
      bias_initializer: conv2d.bias_initializer.to_hash,
      strides: [2, 2],
      padding: true,
      l1_lambda: 0.1,
      l2_lambda: 0.2,
    }
    assert_equal expected_hash, conv2d.to_hash
  end
end


class TestMaxPool2D < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      class: "DNN::Layers::MaxPool2D",
      pool_size: [3, 3],
      strides: [2, 2],
      padding: true,
    }
    pool2d = MaxPool2D.load_hash(hash)
    assert_equal [3, 3], pool2d.pool_size
    assert_equal [2, 2], pool2d.strides
    assert_equal true, pool2d.padding?
  end

  def test_initialize
    pool2d = MaxPool2D.new(3)
    assert_equal [3, 3], pool2d.pool_size
    assert_equal [3, 3], pool2d.strides
  end

  def test_initialize2
    pool2d = MaxPool2D.new(3, strides: 2)
    assert_equal [2, 2], pool2d.strides
  end

  def test_build
    pool2d = MaxPool2D.new([4, 5], strides: [1, 2])
    pool2d.build([32, 32, 3])
    assert_equal 3, pool2d.instance_variable_get(:@num_channel)
    assert_equal [29, 14], pool2d.instance_variable_get(:@out_size)
  end

  def test_build2
    pool2d = MaxPool2D.new([4, 5], strides: [1, 2], padding: true)
    pool2d.build([32, 32, 3])
    assert_equal 3, pool2d.instance_variable_get(:@num_channel)
    assert_equal [32, 16], pool2d.instance_variable_get(:@out_size)
  end

  def test_forward
    x = Numo::SFloat.new(1, 32, 32, 3).seq
    pool2d = MaxPool2D.new(2)
    pool2d.build([32, 32, 3])
    assert_equal [1, 16, 16, 3], pool2d.forward(x).shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 32, 32, 3).seq
    dout = Numo::SFloat.new(1, 16, 16, 3).seq
    pool2d = MaxPool2D.new(2)
    pool2d.build([32, 32, 3])
    pool2d.forward(x)
    assert_equal [1, 32, 32, 3], pool2d.backward(dout).shape
  end

  def test_output_shape
    pool2d = MaxPool2D.new([4, 5], strides: [1, 2])
    pool2d.build([32, 32, 3])
    assert_equal [29, 14, 3], pool2d.output_shape
  end
end


class TestAvgPoo2D < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      class: "DNN::Layers::AvgPool2D",
      pool_size: [3, 3],
      strides: [2, 2],
      padding: true,
    }
    pool2d = AvgPool2D.load_hash(hash)
    assert_equal [3, 3], pool2d.pool_size
    assert_equal [2, 2], pool2d.strides
    assert_equal true, pool2d.padding?
  end

  def test_forward
    x = Numo::SFloat.new(1, 32, 32, 3).seq
    pool2d = AvgPool2D.new(2)
    pool2d.build([32, 32, 3])
    assert_equal [1, 16, 16, 3], pool2d.forward(x).shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 32, 32, 3).seq
    dout = Numo::SFloat.new(1, 16, 16, 3).seq
    pool2d = AvgPool2D.new(2)
    pool2d.build([32, 32, 3])
    pool2d.forward(x)
    assert_equal [1, 32, 32, 3], pool2d.backward(dout).shape
  end
end


class TestUnPool2D < MiniTest::Unit::TestCase
  def test_load_hash
    hash = {
      class: "DNN::Layers::UnPool2D",
      unpool_size: [2, 2],
    }
    unpool2d = UnPool2D.load_hash(hash)
    assert_equal [2, 2], unpool2d.unpool_size
  end

  def test_forward
    x = Numo::SFloat.new(1, 8, 8, 3).seq
    unpool2d = UnPool2D.new(2)
    unpool2d.build([8, 8, 3])
    out = unpool2d.forward(x)
    assert_equal [1, 16, 16, 3], out.shape
  end

  def test_backward
    x = Numo::SFloat.new(1, 8, 8, 3).seq
    dout = Numo::SFloat.new(1, 16, 16, 3).seq
    unpool2d = UnPool2D.new(2)
    unpool2d.build([8, 8, 3])
    unpool2d.forward(x)
    dout2 = unpool2d.backward(dout).round(4)
    assert_equal [1, 8, 8, 3], dout2.shape
  end

  def test_output_shape
    x = Numo::SFloat.new(1, 8, 8, 3).seq
    unpool2d = UnPool2D.new(2)
    unpool2d.build([8, 8, 3])
    assert_equal [16, 16, 3], unpool2d.output_shape
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Layers::UnPool2D",
      unpool_size: [2, 2],
    }
    unpool2d = UnPool2D.new(2)
    assert_equal expected_hash, unpool2d.to_hash
  end
  
end
