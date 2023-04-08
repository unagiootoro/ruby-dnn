module DNN
  # Super class of all iterators.
  class BaseIterator
    attr_reader :num_datas
    attr_reader :last_round_down

    # @param [Boolean] last_round_down Set true to round down for last batch data when call foreach.
    def initialize(last_round_down: false)
      @last_round_down = last_round_down
      @has_next = false
      @num_datas = 0
    end

    # Return the next batch.
    # @param [Integer] batch_size Required batch size.
    # @return [Array] Returns the mini batch in the form (*batches).
    def next_batch(batch_size)
      raise NotImplementedError, "Class '#{self.class.name}' has implement method 'next_batch'"
    end

    # Reset input datas and output datas.
    def reset
      raise NotImplementedError, "Class '#{self.class.name}' has implement method 'reset'"
    end

    # Return the true if has next batch.
    def has_next?
      @has_next
    end

    # Raise an error if there is no next batch.
    private def check_next_batch
      raise DNNError, "This iterator has not next batch. Please call reset." unless has_next?
    end

    # Run a loop with all data separated by batch
    # @param [Integer] batch_size Batch size.
    # @yield Executes block by receiving the specified arguments (*batches).
    def foreach(batch_size, &block)
      max_steps(batch_size).times do |step|
        batches = next_batch(batch_size)
        block.call(*batches, step)
      end
      reset
    end

    # Return the number of available data considering last_round_down.
    def num_usable_datas(batch_size)
      if @last_round_down
        max_steps(batch_size) * batch_size
      else
        @num_datas
      end
    end

    # Get max steps for iteration.
    # @param [Integer] batch_size Batch size.
    def max_steps(batch_size)
      @last_round_down ? @num_datas / batch_size : (@num_datas.to_f / batch_size).ceil
    end
  end


  # This class manages input datas and output datas together.
  class Iterator < BaseIterator
    # @param [Array] datas input datas.
    # @param [Boolean] random Set true to return batches randomly. Setting false returns batches in order of index.
    # @param [Boolean] last_round_down Set true to round down for last batch data when call foreach.
    def initialize(*datas, random: true, last_round_down: false)
      super(last_round_down: last_round_down)
      datas.each.with_index do |data, i|
        Utils.check_input_data_type("datas[#{i}]", data, Xumo::NArray)
      end
      @datas = datas
      @random = random
      @num_datas = datas[0].is_a?(Array) ? datas[0][0].shape[0] : datas[0].shape[0]
      reset
    end

    # Return the next batch.
    # @param [Integer] batch_size Required batch size.
    # @return [Array] Returns the mini batch in the form (*batches).
    def next_batch(batch_size)
      check_next_batch
      if @indexes.length <= batch_size
        batch_indexes = @indexes
        @has_next = false
      else
        batch_indexes = @indexes.shift(batch_size)
      end
      get_batch(batch_indexes)
    end

    # Implement a process to get mini batch.
    # @param [Array] batch_indexes Index of batch to get.
    # @return [Array] Returns the mini batch in the form (*batches).
    private def get_batch(batch_indexes)
      batches = @datas.map do |data|
        if data.is_a?(Array)
          data.map { |datas| datas[batch_indexes, false] }
        else
          data[batch_indexes, false]
        end
      end
      batches
    end

    # Reset input datas and output datas.
    def reset
      @has_next = true
      @indexes = @num_datas.times.to_a
      @indexes.shuffle! if @random
    end
  end
end
