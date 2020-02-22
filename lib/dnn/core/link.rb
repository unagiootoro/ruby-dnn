module DNN
  class Link
    attr_accessor :prevs
    attr_accessor :next
    attr_accessor :layer_node

    def initialize(prevs = nil, layer_node = nil)
      @prevs = prevs
      @layer_node = layer_node
      @next = nil
      @hold = []
    end

    def forward(x)
      @hold << x
      return if @hold.length < @prevs.length
      x = @layer_node.(*@hold)
      @hold = []
      @next ? @next.forward(x) : x
    end

    def backward(dy = Xumo::SFloat[1])
      dys = @layer_node.backward_node(dy)
      if dys.is_a?(Array)
        dys.each.with_index do |dy, i|
          @prevs[i]&.backward(dy)
        end
      else
        @prevs.first&.backward(dys)
      end
    end
  end
end
