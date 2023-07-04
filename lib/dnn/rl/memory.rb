module DNN
  module RL
    class Memory
      attr_reader :max_size
      attr_reader :train_size

      def initialize(max_size, train_size)
        @max_size = max_size
        @train_size = train_size
        @memory = []
      end

      def add(v)
        @memory << v
        @memory.shift if @memory.size > @max_size
        v
      end

      def size
        @memory.size
      end

      def shuffle
        @memory.shuffle
      end

      def to_a
        @memory
      end

      def can_train?
        @memory.size >= @train_size 
      end
    end
  end
end
