require "test_helper"

Iterator = DNN::Iterator

class TestIterator < MiniTest::Unit::TestCase
  def test_next_batch
    x_datas = Numo::Int32.zeros(10, 10)
    y_datas = Numo::Int32.zeros(10, 10)
    iter = DNN::Iterator.new(x_datas, y_datas)

    iter.next_batch(7)
    x, y = iter.next_batch(7)
    assert_equal [7, 10], x.shape
    assert_equal [7, 10], y.shape
  end

  def test_next_batch2
    x_datas = Numo::Int32.new(10, 1).seq
    iter = DNN::Iterator.new(x_datas, x_datas, random: false)

    iter.next_batch(7)
    x, * = iter.next_batch(7)
    assert_equal Numo::SFloat[7, 8, 9], x.flatten
  end
end
