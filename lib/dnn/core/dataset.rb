class DNN::Dataset
  def initialize(x_datas, y_datas)
    @x_datas = x_datas
    @y_datas = y_datas
    @num_datas = x_datas.shape[0]
    @indexes = @num_datas.times.to_a.shuffle
  end

  def get_batch(batch_size)
    if @indexes.length < batch_size
      @indexes = @num_datas.times.to_a.shuffle
    end
    batch_indexes = @indexes.shift(batch_size)
    x_batch = @x_datas[batch_indexes, false]
    y_batch = @y_datas[batch_indexes, false]
    [x_batch, y_batch]
  end
end
