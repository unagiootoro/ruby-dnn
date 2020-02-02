module DNN
  class Link
    attr_accessor :prev
    attr_accessor :next
    attr_accessor :layer_node

    def initialize(prev = nil, layer_node = nil)
      @prev = prev
      @layer_node = layer_node
      @next = nil
    end

    def forward(x)
      x = @layer_node.(x)
      @next ? @next.forward(x) : x
    end

    def backward(dy = Numo::SFloat[1])
      dy = @layer_node.backward_node(dy)
      @prev&.backward(dy)
    end
  end

  class TwoInputLink
    attr_accessor :prev1
    attr_accessor :prev2
    attr_accessor :next
    attr_accessor :layer_node

    def initialize(prev1 = nil, prev2 = nil, layer_node = nil)
      @prev1 = prev1
      @prev2 = prev2
      @layer_node = layer_node
      @next = nil
      @hold = []
    end

    def forward(x)
      @hold << x
      return if @hold.length < 2
      x = @layer_node.(*@hold)
      @hold = []
      @next ? @next.forward(x) : x
    end

    def backward(dy = Numo::SFloat[1])
      dys = @layer_node.backward_node(dy)
      if dys.is_a?(Array)
        dy1, dy2 = *dys
      else
        dy1 = dys
      end
      @prev1&.backward(dy1)
      @prev2&.backward(dy2) if dy2
    end
  end
end
