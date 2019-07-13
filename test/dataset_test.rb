require "test_helper"

include DNN

class TestDataset < MiniTest::Unit::TestCase
  def test_next_batch
    x_datas = Numo::Int32.zeros(10, 10)
    y_datas = Numo::Int32.zeros(10, 10)
    dataset = Dataset.new(x_datas, y_datas)

    dataset.next_batch(7)
    x, y = dataset.next_batch(7)
    assert_equal [7, 10], x.shape
    assert_equal [7, 10], y.shape
  end

  def test_next_batch2
    x_datas = Numo::Int32.new(10, 1).seq
    dataset = Dataset.new(x_datas, x_datas, false)

    dataset.next_batch(7)
    x, * = dataset.next_batch(7)
    assert_equal Numo::SFloat[7, 8, 9], x.flatten
  end
end
