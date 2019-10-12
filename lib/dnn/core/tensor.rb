module DNN
  class Tensor
    attr_reader :value
    attr_accessor :link

    def initialize(value, link = nil)
      @value = value
      @link = link
    end
  end
end
