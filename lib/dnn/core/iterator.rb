module DNN
  # This class manages input datas and output datas together.
  class Iterator
    # @param [Numo::SFloat] x_datas input datas.
    # @param [Numo::SFloat] y_datas output datas.
    # @param [Boolean] random Set true to return batches randomly. Setting false returns batches in order of index.
    # @param [Boolean] last_round_down Set true to round down for last batch data when call foreach.
    def initialize(x_datas, y_datas, random: true, last_round_down: false)
      @x_datas = x_datas
      @y_datas = y_datas
      @random = random
      @last_round_down = last_round_down
      @num_datas = x_datas.is_a?(Array) ? x_datas[0].shape[0] : x_datas.shape[0]
      reset
    end

    # Return the next batch.
    # @param [Integer] batch_size Required batch size.
    def next_batch(batch_size)
      raise DNN_Error, "This iterator has not next batch. Please call reset." unless has_next?

      if @indexes.length <= batch_size
        batch_indexes = @indexes
        @has_next = false
      else
        batch_indexes = @indexes.shift(batch_size)
      end
      x_batch = if @x_datas.is_a?(Array)
                  @x_datas.map { |datas| datas[batch_indexes, false] }
                else
                  @x_datas[batch_indexes, false]
                end
      y_batch = if @y_datas.is_a?(Array)
                  @y_datas.map { |datas| datas[batch_indexes, false] }
                else
                  @y_datas[batch_indexes, false]
                end
      [x_batch, y_batch]
    end

    # Reset input datas and output datas.
    def reset
      @has_next = true
      @indexes = @num_datas.times.to_a
      @indexes.shuffle! if @random
    end

    # Return the true if has next batch.
    def has_next?
      @has_next
    end

    def foreach(batch_size, &block)
      steps = @last_round_down ? @num_datas / batch_size : (@num_datas.to_f / batch_size).ceil
      steps.times do |step|
        x_batch, y_batch = next_batch(batch_size)
        block.call(x_batch, y_batch, step)
      end
      reset
    end
  end
end
