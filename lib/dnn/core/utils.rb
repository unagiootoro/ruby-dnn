module DNN
  # This module provides utility functions.
  module Utils
    NMath = Xumo::NMath

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

    def self.sigmoid(x)
      1 / (1 + NMath.exp(-x))
    end

    def self.softmax(x)
      NMath.exp(x) / NMath.exp(x).sum(1).reshape(x.shape[0], 1)
    end
  end
end
