# This class manages input datas and output datas together.
module DNN
  class Dataset
    # @param [Numo::SFloat] x_datas input datas.
    # @param [Numo::SFloat] y_datas output datas.
    # @param [Boolean] random Set true to return batches randomly. Setting false returns batches in order of index.
    def initialize(x_datas, y_datas, random = true)
      @x_datas = x_datas
      @y_datas = y_datas
      @random = random
      @num_datas = x_datas.shape[0]
      reset_indexs
    end

    # Return the next batch.
    # If the number of remaining data < batch size, and random = true then shuffle the data again and return a batch.
    # If random = false, all remaining data will be returned regardless of the batch size.
    # @param [Integer] batch_size Required batch size.
    def next_batch(batch_size)
      if @indexes.length < batch_size
        batch_indexes = @indexes unless @random
        reset_indexs
        batch_indexes = @indexes.shift(batch_size) if @random
      else
        batch_indexes = @indexes.shift(batch_size)
      end
      x_batch = @x_datas[batch_indexes, false]
      y_batch = @y_datas[batch_indexes, false]
      [x_batch, y_batch]
    end

    private def reset_indexs
      @indexes = @num_datas.times.to_a
      @indexes.shuffle! if @random
    end
  end
end
