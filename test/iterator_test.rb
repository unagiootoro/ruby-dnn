require "test_helper"

class TestIterator < MiniTest::Unit::TestCase
  def test_next_batch
    x_datas = Xumo::Int32.zeros(10, 10)
    y_datas = Xumo::Int32.zeros(10, 10)
    iter = DNN::Iterator.new(x_datas, y_datas)

    iter.next_batch(7)
    x, y = iter.next_batch(7)
    assert_equal [3, 10], x.shape
    assert_equal [3, 10], y.shape
  end

  def test_next_batch2
    x_datas = Xumo::Int32.new(10, 1).seq
    iter = DNN::Iterator.new(x_datas, x_datas, random: false)

    iter.next_batch(7)
    x, * = iter.next_batch(7)
    assert_equal Xumo::SFloat[7, 8, 9], x.flatten
  end

  def test_next_batch3
    x_datas = Xumo::Int32.new(10, 1).seq
    iter = DNN::Iterator.new([x_datas, x_datas], [x_datas, x_datas], random: false)

    x, y = iter.next_batch(3)
    assert_equal Xumo::SFloat[0, 1, 2], x[0].flatten
    assert_equal Xumo::SFloat[0, 1, 2], y[1].flatten
  end

  def test_foreach
    x_datas = Xumo::Int32.new(10, 1).seq
    iter = DNN::Iterator.new(x_datas, x_datas, random: false)

    cnt = 0
    iter.foreach(3) do |x_batch, y_batch|
      cnt += 1
    end
    assert_equal 4, cnt
  end

  def test_foreach2
    x_datas = Xumo::Int32.new(10, 1).seq
    iter = DNN::Iterator.new(x_datas, x_datas, random: false, last_round_down: true)

    cnt = 0
    iter.foreach(3) do |x_batch, y_batch|
      cnt += 1
    end
    assert_equal 3, cnt
  end

  def test_max_steps
    x_datas = Xumo::Int32.new(10, 1).seq
    iter = DNN::Iterator.new(x_datas, x_datas, random: false)
    assert_equal 4, iter.max_steps(3)
  end

  def test_max_steps2
    x_datas = Xumo::Int32.new(10, 1).seq
    iter = DNN::Iterator.new(x_datas, x_datas, random: false, last_round_down: true)
    assert_equal 3, iter.max_steps(3)
  end
end
