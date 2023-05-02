module DNN
  class Tensor < BaseTensor
    attr_reader :prev_link

    def self.convert(inputs)
      inputs.is_a?(Array) ? inputs.map { |input| self.new(input) } : self.new(inputs)
    end

    def initialize(data, prev_link: nil, backward_index: nil)
      super(data)
      @prev_link = prev_link
      @backward_index = backward_index
    end

    private def backward_internal(grad)
      @prev_link.backward(grad, @backward_index) if @prev_link
    end

    def requires_grad
      @prev_link ? @prev_link.requires_grad : false
    end
  end
end
