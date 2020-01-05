require "test_helper"
require "fileutils"
require "dnn/datasets/mnist"
require "dnn/datasets/fashion-mnist"
require "dnn/datasets/cifar10"
require "dnn/datasets/cifar100"
require "dnn/datasets/stl-10"
require "dnn/image"

class TestMNIST < MiniTest::Unit::TestCase
  def test_load_train
    x, y = DNN::MNIST.load_train
    assert_equal [60000, 28, 28, 1], x.shape
    assert_equal 186, x[0, 15, 15, 0]
    assert_equal [60000], y.shape
    assert_equal 5, y[0]
  end

  def test_load_test
    x, y = DNN::MNIST.load_test
    assert_equal [10000, 28, 28, 1], x.shape
    assert_equal 0, x[0, 15, 15, 0]
    assert_equal [10000], y.shape
    assert_equal 7, y[0]
  end
end

class TestFashionMNIST < MiniTest::Unit::TestCase
  def test_load_train
    x, y = DNN::FashionMNIST.load_train
    assert_equal [60000, 28, 28, 1], x.shape
    assert_equal 221, x[0, 15, 15, 0]
    assert_equal [60000], y.shape
    assert_equal 9, y[0]
  end

  def test_load_test
    x, y = DNN::FashionMNIST.load_test
    assert_equal [10000, 28, 28, 1], x.shape
    assert_equal 117, x[0, 15, 15, 0]
    assert_equal [10000], y.shape
    assert_equal 9, y[0]
  end
end

class TestCIFAR10 < MiniTest::Unit::TestCase
  def test_load_train
    x, y = DNN::CIFAR10.load_train
    assert_equal [50000, 32, 32, 3], x.shape
    assert_equal [247, 234, 212], x[0, 15, 15, true].to_a
    assert_equal [50000], y.shape
    assert_equal 6, y[0]
  end

  def test_load_test
    x, y = DNN::CIFAR10.load_test
    assert_equal [10000, 32, 32, 3], x.shape
    assert_equal [131, 121, 112], x[0, 15, 15, true].to_a
    assert_equal [10000], y.shape
    assert_equal 3, y[0]
  end
end

class TestCIFAR100 < MiniTest::Unit::TestCase
  def test_load_train
    x, y = DNN::CIFAR100.load_train
    assert_equal [50000, 32, 32, 3], x.shape
    assert_equal [203, 107, 117], x[0, 15, 15, true].to_a
    assert_equal [50000, 2], y.shape
    assert_equal 11, y[0, 0]
  end

  def test_load_test
    x, y = DNN::CIFAR100.load_test
    assert_equal [10000, 32, 32, 3], x.shape
    assert_equal [196, 190, 207], x[0, 15, 15, true].to_a
    assert_equal [10000, 2], y.shape
    assert_equal 10, y[0, 0]
  end
end

class TestSTL10 < MiniTest::Unit::TestCase
  def test_load_train
    x, y = DNN::STL10.load_train
    assert_equal [5000, 96, 96, 3], x.shape
    assert_equal [51, 34, 33], x[0, 47, 47, true].to_a
    assert_equal [5000], y.shape
    assert_equal 2, y[0]
  end

  def test_load_test
    x, y = DNN::STL10.load_test
    assert_equal [8000, 96, 96, 3], x.shape
    assert_equal [109, 125, 137], x[0, 47, 47, true].to_a
    assert_equal [8000], y.shape
    assert_equal 7, y[0]
  end

  def test_load_unlabeled
    x, y = DNN::STL10.load_unlabeled(100...200)
    assert_equal [100, 96, 96, 3], x.shape
    assert_equal [72, 36, 21], x[0, 47, 47, true].to_a
  end
end
