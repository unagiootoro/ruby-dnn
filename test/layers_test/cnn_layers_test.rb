require "test_helper"

class TestConv2D_Utils < MiniTest::Unit::TestCase
  include DNN::Layers::Conv2DUtils

  # im2col test.
  def test_im2col
    img = Xumo::SFloat.cast([[
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
    expected_col = Xumo::SFloat[
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
    img = Xumo::SFloat.new(1, 2, 4, 4).seq(1).transpose(0, 2, 3, 1)
    expected_col = Xumo::SFloat[
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
    col = Xumo::SFloat[
      [1, 17, 2, 18, 3, 19, 5, 21, 6, 22, 7, 23, 9, 25, 10, 26, 11, 27],
      [2, 18, 3, 19, 4, 20, 6, 22, 7, 23, 8, 24, 10, 26, 11, 27, 12, 28],
      [5, 21, 6, 22, 7, 23, 9, 25, 10, 26, 11, 27, 13, 29, 14, 30, 15, 31],
      [6, 22, 7, 23, 8, 24, 10, 26, 11, 27, 12, 28, 14, 30, 15, 31, 16, 32],
    ]
    expected_img = Xumo::SFloat.cast([[
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
    col = Xumo::SFloat[
      [1, 17, 2, 18, 5, 21, 6, 22],
      [3, 19, 4, 20, 7, 23, 8, 24],
      [9, 25, 10, 26, 13, 29, 14, 30],
      [11, 27, 12, 28, 15, 31, 16, 32],
    ]
    expected_img = Xumo::SFloat.new(1, 2, 4, 4).seq(1).transpose(0, 2, 3, 1)
    img = col2im(col, img_shape, 2, 2, 2, 2, [2, 2])
    assert_equal expected_img.round(4), img.round(4)
  end

  def test_zero_padding
    img = Xumo::SFloat.new(1, 2, 4, 4).seq(1).transpose(0, 2, 3, 1)
    expected_img = Xumo::SFloat.cast([[
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
    assert_equal expected_img, zero_padding(img, [2, 2])
  end

  def test_zero_padding_bwd
    img = Xumo::SFloat.cast([[
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
    expected_img = Xumo::SFloat.new(1, 2, 4, 4).seq(1).transpose(0, 2, 3, 1)
    assert_equal expected_img, zero_padding_bwd(img, [2, 2])
  end

  def test_calc_conv2d_out_size
    assert_equal [29, 14], calc_conv2d_out_size(32, 32, 4, 5, 0, 0, [1, 2])
  end

  def test_calc_conv2d_out_size2
    assert_equal [32, 32], calc_conv2d_out_size(32, 32, 1, 1, 0, 0, [1, 1])
  end

  def test_calc_conv2d_transpose_out_size
    assert_equal [32, 31], calc_conv2d_transpose_out_size(29, 14, 4, 5, 0, 0, [1, 2])
  end

  def test_calc_conv2d_transpose_out_size2
    assert_equal [32, 32], calc_conv2d_transpose_out_size(32, 32, 1, 1, 0, 0, [1, 1])
  end

  def test_calc_conv2d_padding_size
    assert_equal [4, 3], calc_conv2d_padding_size(32, 32, 5, 5, [1, 2])
  end
  
  def test_calc_conv2d_padding_size2
    assert_equal [2, 0], calc_conv2d_padding_size(32, 32, 4, 5, [2, 3])
  end

  def test_calc_conv2d_transpose_padding_size
    assert_equal [4, 3], calc_conv2d_transpose_padding_size(32, 32, 5, 5, [1, 2])
  end
  
  def test_calc_conv2d_transpose_padding_size2
    assert_equal [2, 2], calc_conv2d_transpose_padding_size(32, 32, 4, 5, [2, 3])
  end
end

class TestConv2D < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::Conv2D",
      num_filters: 16,
      filter_size: [3, 3],
      weight_initializer: DNN::Initializers::RandomNormal.new.to_hash,
      bias_initializer: DNN::Initializers::Zeros.new.to_hash,
      strides: [2, 2],
      padding: true,
      weight_regularizer: DNN::Regularizers::L1.new.to_hash,
      bias_regularizer: DNN::Regularizers::L2.new.to_hash,
      l2_lambda: 0,
      use_bias: false,
    }
    conv2d = DNN::Layers::Conv2D.from_hash(hash)
    assert_equal false, conv2d.use_bias
    assert_equal 16, conv2d.num_filters
    assert_equal [3, 3], conv2d.filter_size
    assert_equal [2, 2], conv2d.strides
    assert_equal true, conv2d.padding
  end

  def test_initialize
    conv2d = DNN::Layers::Conv2D.new(16, 3)
    assert_equal [3, 3], conv2d.filter_size
  end

  def test_initialize2
    conv2d = DNN::Layers::Conv2D.new(16, 3, strides: 2)
    assert_equal [2, 2], conv2d.strides
  end

  def test_build
    conv2d = DNN::Layers::Conv2D.new(16, [4, 5], strides: [1, 2])
    conv2d.build([32, 32, 3])
    assert_equal [29, 14], conv2d.instance_variable_get(:@out_size)
  end

  def test_build2
    conv2d = DNN::Layers::Conv2D.new(16, [4, 5], strides: [1, 2], padding: true)
    conv2d.build([32, 32, 3])
    assert_equal [32, 16], conv2d.instance_variable_get(:@out_size)
  end

  def test_build3
    conv2d = DNN::Layers::Conv2D.new(16, 4, padding: 3)
    conv2d.build([32, 32, 3])
    assert_equal [32, 32], conv2d.instance_variable_get(:@out_size)
  end

  def test_build4
    conv2d = DNN::Layers::Conv2D.new(16, [4, 3], strides: [2, 2], padding: true)
    conv2d.build([28, 28, 3])
    assert_equal [14, 14], conv2d.instance_variable_get(:@out_size)
  end


  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 32, 32, 3).seq)
    conv2d = DNN::Layers::Conv2D.new(16, 5)
    conv2d.build([32, 32, 3])
    assert_equal [1, 28, 28, 16], conv2d.forward(x).shape
  end

  def test_filters
    conv2d = DNN::Layers::Conv2D.new(16, [4, 5])
    conv2d.build([32, 32, 3])
    assert_equal [4, 5, 3, 16], conv2d.filters.shape
  end

  def test_filters_set
    conv2d = DNN::Layers::Conv2D.new(16, [4, 5])
    conv2d.build([32, 32, 3])
    conv2d.filters = Xumo::SFloat.zeros(4, 5, 3, 16)
    assert_equal [4 * 5 * 3, 16], conv2d.weight.data.shape
  end

  def test_filters_set2
    conv2d = DNN::Layers::Conv2D.new(16, [4, 5])
    conv2d.build([32, 32, 3])
    expected = conv2d.weight.data
    conv2d.filters = expected
    assert_equal expected, conv2d.weight.data
  end

  def test_to_hash
    conv2d = DNN::Layers::Conv2D.new(16, 5, strides: 2, padding: true, use_bias: false,
                                     weight_regularizer: DNN::Regularizers::L1.new,
                                     bias_regularizer: DNN::Regularizers::L2.new)
    expected_hash = {
      class: "DNN::Layers::Conv2D",
      num_filters: 16,
      filter_size: [5, 5],
      weight_initializer: conv2d.weight_initializer.to_hash,
      bias_initializer: conv2d.bias_initializer.to_hash,
      strides: [2, 2],
      padding: true,
      weight_regularizer: conv2d.weight_regularizer.to_hash,
      bias_regularizer: conv2d.bias_regularizer.to_hash,
      use_bias: false,
    }
    assert_equal expected_hash, conv2d.to_hash
  end
end

class TestConv2DTranspose < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::Conv2DTranspose",
      num_filters: 16,
      filter_size: [3, 3],
      weight_initializer: DNN::Initializers::RandomNormal.new.to_hash,
      bias_initializer: DNN::Initializers::Zeros.new.to_hash,
      strides: [2, 2],
      padding: true,
      weight_regularizer: DNN::Regularizers::L1.new.to_hash,
      bias_regularizer: DNN::Regularizers::L2.new.to_hash,
      use_bias: true,
    }
    conv2d_t = DNN::Layers::Conv2DTranspose.from_hash(hash)
    assert_equal 16, conv2d_t.num_filters
    assert_equal [3, 3], conv2d_t.filter_size
    assert_equal [2, 2], conv2d_t.strides
    assert_equal true, conv2d_t.padding
  end

  def test_initialize
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, 3)
    assert_equal [3, 3], conv2d_t.filter_size
  end

  def test_initialize2
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, 3, strides: 2)
    assert_equal [2, 2], conv2d_t.strides
  end

  def test_build
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, [4, 6], strides: [1, 2])
    conv2d_t.build([29, 14, 3])
    assert_equal [32, 32], conv2d_t.instance_variable_get(:@out_size)
  end

  def test_build2
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, [4, 6], strides: [1, 2], padding: true)
    conv2d_t.build([32, 16, 3])
    assert_equal [32, 32], conv2d_t.instance_variable_get(:@out_size)
  end

  def test_build3
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, 4, padding: 3)
    conv2d_t.build([32, 32, 3])
    assert_equal [32, 32], conv2d_t.instance_variable_get(:@out_size)
  end

  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 16, 16, 3).seq)
    conv2d_t = DNN::Layers::Conv2DTranspose.new(8, 2, strides: 2)
    conv2d_t.build([16, 16, 3])
    assert_equal [1, 32, 32, 8], conv2d_t.forward(x).shape
  end

  def test_filters
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, [4, 5])
    conv2d_t.build([32, 32, 3])
    assert_equal [4, 5, 16, 3], conv2d_t.filters.shape
  end

  def test_filters_set
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, [4, 5])
    conv2d_t.build([32, 32, 3])
    conv2d_t.filters = Xumo::SFloat.zeros(4, 5, 3, 16)
    assert_equal [4 * 5 * 16, 3], conv2d_t.weight.data.shape
  end

  def test_filters_set2
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, [4, 5])
    conv2d_t.build([32, 32, 3])
    expected = conv2d_t.weight.data
    conv2d_t.filters = expected
    assert_equal expected, conv2d_t.weight.data
  end

  def test_to_hash
    conv2d_t = DNN::Layers::Conv2DTranspose.new(16, 5, strides: 2, padding: true, use_bias: false,
                                                weight_regularizer: DNN::Regularizers::L1.new,
                                                bias_regularizer: DNN::Regularizers::L2.new)
    expected_hash = {
      class: "DNN::Layers::Conv2DTranspose",
      num_filters: 16,
      filter_size: [5, 5],
      weight_initializer: conv2d_t.weight_initializer.to_hash,
      bias_initializer: conv2d_t.bias_initializer.to_hash,
      strides: [2, 2],
      padding: true,
      weight_regularizer: conv2d_t.weight_regularizer.to_hash,
      bias_regularizer: conv2d_t.bias_regularizer.to_hash,
      use_bias: false
    }
    assert_equal expected_hash, conv2d_t.to_hash
  end
end

class TestMaxPool2D < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::MaxPool2D",
      pool_size: [3, 3],
      strides: [2, 2],
      padding: true,
    }
    pool2d = DNN::Layers::MaxPool2D.from_hash(hash)
    assert_kind_of DNN::Layers::MaxPool2D, pool2d
    assert_equal [3, 3], pool2d.pool_size
    assert_equal [2, 2], pool2d.strides
    assert_equal true, pool2d.padding
  end

  def test_initialize
    pool2d = DNN::Layers::MaxPool2D.new(3)
    assert_equal [3, 3], pool2d.pool_size
    assert_equal [3, 3], pool2d.strides
  end

  def test_initialize2
    pool2d = DNN::Layers::MaxPool2D.new(3, strides: 2)
    assert_equal [2, 2], pool2d.strides
  end

  def test_build
    pool2d = DNN::Layers::MaxPool2D.new([4, 5], strides: [1, 2])
    pool2d.build([32, 32, 3])
    assert_equal 3, pool2d.instance_variable_get(:@num_channel)
    assert_equal [29, 14], pool2d.instance_variable_get(:@out_size)
  end

  def test_build2
    pool2d = DNN::Layers::MaxPool2D.new([4, 5], strides: [1, 2], padding: true)
    pool2d.build([32, 32, 3])
    assert_equal 3, pool2d.instance_variable_get(:@num_channel)
    assert_equal [32, 16], pool2d.instance_variable_get(:@out_size)
  end

  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 32, 32, 3).seq)
    pool2d = DNN::Layers::MaxPool2D.new(2)
    pool2d.build([32, 32, 3])
    assert_equal [1, 16, 16, 3], pool2d.forward(x).shape
  end
end

class TestAvgPoo2D < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::AvgPool2D",
      pool_size: [3, 3],
      strides: [2, 2],
      padding: true,
    }
    pool2d = DNN::Layers::AvgPool2D.from_hash(hash)
    assert_kind_of DNN::Layers::AvgPool2D, pool2d
    assert_equal [3, 3], pool2d.pool_size
    assert_equal [2, 2], pool2d.strides
    assert_equal true, pool2d.padding
  end

  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 32, 32, 3).seq)
    pool2d = DNN::Layers::AvgPool2D.new(2)
    pool2d.build([32, 32, 3])
    y = pool2d.(x)
    assert_equal [1, 16, 16, 3], y.shape
  end
end

class TestGlobalAvgPoo2D < MiniTest::Unit::TestCase
  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat.new(1, 8, 8, 64).seq)
    pool2d = DNN::Layers::GlobalAvgPool2D.new
    pool2d.build([8, 8, 64])
    assert_equal [1, 64], pool2d.forward(x).shape
  end
end

class TestUnPool2D < MiniTest::Unit::TestCase
  def test_from_hash
    hash = {
      class: "DNN::Layers::UnPool2D",
      unpool_size: [2, 2],
    }
    unpool2d = DNN::Layers::UnPool2D.from_hash(hash)
    assert_equal [2, 2], unpool2d.unpool_size
  end

  def test_forward
    x = DNN::Tensor.new(Xumo::SFloat[
      [
        [[1, 5], [2, 6]],
        [[3, 7], [4, 8]],
      ]
    ])
    unpool2d = DNN::Layers::UnPool2D.new(2)
    unpool2d.build([2, 2, 1])
    y = unpool2d.(x)
    expected = Xumo::SFloat[
      [
        [[1, 5], [1, 5], [2, 6], [2, 6]],
        [[1, 5], [1, 5], [2, 6], [2, 6]],
        [[3, 7], [3, 7], [4, 8], [4, 8]],
        [[3, 7], [3, 7], [4, 8], [4, 8]],
      ]
    ]
    assert_equal expected, y.data
  end

  def test_backward
    x = DNN::Variable.new(Xumo::SFloat[
      [
        [[1, 5], [2, 6]],
        [[3, 7], [4, 8]],
      ]
    ])
    unpool2d = DNN::Layers::UnPool2D.new(2)
    unpool2d.build([2, 2, 1])
    y = unpool2d.(x)
    dy = Xumo::SFloat[
      [
        [[1, 5], [1, 5], [2, 6], [2, 6]],
        [[1, 5], [1, 5], [2, 6], [2, 6]],
        [[3, 7], [3, 7], [4, 8], [4, 8]],
        [[3, 7], [3, 7], [4, 8], [4, 8]],
      ]
    ]
    y.backward(dy)
    expected = Xumo::SFloat[
      [
        [[4, 20], [8, 24]],
        [[12, 28], [16, 32]],
      ]
    ]
    assert_equal expected, x.grad
  end

  def test_to_hash
    expected_hash = {
      class: "DNN::Layers::UnPool2D",
      unpool_size: [2, 2],
    }
    unpool2d = DNN::Layers::UnPool2D.new(2)
    assert_equal expected_hash, unpool2d.to_hash
  end
  
end
