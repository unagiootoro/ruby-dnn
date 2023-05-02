module DNN
  module Functions
    class Function
      def self.call(*inputs)
        new.(*inputs)
      end

      def call(*inputs)
        xs = inputs.map(&:data)
        ys = forward(*xs)
        num_outputs = (ys.is_a?(Array) ? ys.length : 1)
        if inputs.find { |prev| prev && prev.requires_grad }
          link = Link.new(prevs: inputs, node: self, num_outputs: num_outputs)
          inputs.each do |input|
            input.add_next_link(link) if input.is_a?(Tensor)
          end
        else
          link = nil
        end
        if ys.is_a?(Array)
          ys.map.with_index { |y, i| Tensor.new(y, prev_link: link, backward_index: i) }
        else
          Tensor.new(ys, prev_link: link, backward_index: 0)
        end
      end

      def forward(*xs)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'forward'"
      end

      def backward(*dys)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'backward'"
      end
    end
  end
end

if RUBY_PLATFORM != "wasm32-wasi"
  require_relative "functions/basic"
  require_relative "functions/math"
  require_relative "functions/activations"
  require_relative "functions/losses"
  require_relative "functions/dropout"
  require_relative "functions/normalizations"
  require_relative "functions/embedding"
  require_relative "functions/cnn"
  require_relative "functions/rnn"
end
