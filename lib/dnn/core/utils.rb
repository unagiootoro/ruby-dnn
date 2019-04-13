module DNN
  # This module provides utility functions.
  module Utils
    # Create a mini batch for "batch_size".
    def self.get_minibatch(x, y, batch_size)
      indexes = (0...x.shape[0]).to_a.sample(batch_size)
      [x[indexes, false], y[indexes, false]]
    end

    # Categorize labels into "num_classes" classes.
    def self.to_categorical(y, num_classes, narray_type = nil)
      narray_type ||= y.class
      y2 = narray_type.zeros(y.shape[0], num_classes)
      y.shape[0].times do |i|
        y2[i, y[i]] = 1
      end
      y2
    end

    # Convert hash to an object.
    def self.load_hash(hash)
      dnn_class = DNN.const_get(hash[:class])
      if dnn_class.respond_to?(:load_hash)
        return dnn_class.load_hash(hash)
      end
      dnn_class.new
    end
  end
end
