module DNN
  class Link
    attr_accessor :prev
    attr_accessor :layer

    def initialize(prev = nil, layer = nil)
      @prev = prev
      @layer = layer
    end
  end


  class TwoInputLink
    attr_accessor :prev1
    attr_accessor :prev2
    attr_accessor :layer

    def initialize(prev1 = nil, prev2 = nil, layer = nil)
      @prev1 = prev1
      @prev2 = prev2
      @layer = layer
    end
  end
end
