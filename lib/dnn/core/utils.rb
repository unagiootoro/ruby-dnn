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
    def self.hash_to_obj(hash)
      return nil if hash == nil
      dnn_class = DNN.const_get(hash[:class])
      dnn_class.from_hash(hash)
    end

    # Return the result of the sigmoid function.
    def self.sigmoid(x)
      Functions::Sigmoid.new.forward(x)
    end

    # Return the result of the softmax function.
    def self.softmax(x)
      Functions::Softmax.new.forward(x)
    end

    # Check training or evaluate input data type.
    def self.check_input_data_type(data_name, data, expected_type)
      if !data.is_a?(expected_type) && !data.is_a?(Array)
        raise TypeError, "#{data_name}:#{data.class.name} is not an instance of #{expected_type.name} class or Array class."
      end
      if data.is_a?(Array)
        data.each.with_index do |v, i|
          unless v.is_a?(expected_type)
            raise TypeError, "#{data_name}[#{i}]:#{v.class.name} is not an instance of #{expected_type.name} class."
          end
        end
      end
    end

    # Perform numerical differentiation.
    def self.numerical_grad(x, func)
      (func.(x + 1e-7) - func.(x)) / 1e-7
    end

    # Convert numo to cumo.
    def self.numo2cumo(na)
      b = na.to_binary
      ca = Cumo::SFloat.from_binary(b)
      ca.reshape(*na.shape)
    end

    # Convert cumo to numo.
    def self.cumo2numo(ca)
      b = ca.to_binary
      na = Numo::SFloat.from_binary(b)
      na.reshape(*ca.shape)
    end

    # Force convert to Float.
    def self.to_f(x)
      if x.is_a?(Xumo::NArray)
        x[0].to_f
      else
        x.to_f
      end
    end
  end
end
