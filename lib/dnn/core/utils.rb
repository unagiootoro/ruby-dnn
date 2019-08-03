module DNN
  # This module provides utility functions.
  module Utils
    # Categorize labels into "num_classes" classes.
    # @param [Numo::SFloat] y Label data.
    # @param [Numo::SFloat] num_classes Number of classes to classify.
    # @param [Class] narray_type Type of Numo::Narray data after classification.
    def self.to_categorical(y, num_classes, narray_type = nil)
      narray_type ||= y.class
      y2 = narray_type.zeros(y.shape[0], num_classes)
      y.shape[0].times do |i|
        y2[i, y[i]] = 1
      end
      y2
    end

    # Convert hash to an object.
    def self.from_hash(hash)
      return nil if hash == nil
      dnn_class = DNN.const_get(hash[:class])
      if dnn_class.respond_to?(:from_hash)
        return dnn_class.from_hash(hash)
      end
      dnn_class.new
    end

    # Return the result of the sigmoid function.
    def self.sigmoid(x)
      Sigmoid.new.forward(x)
    end

    # Return the result of the softmax function.
    def self.softmax(x)
      SoftmaxCrossEntropy.softmax(x)
    end
  end
end
