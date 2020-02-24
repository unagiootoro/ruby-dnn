module DNN
  class Link
    attr_accessor :prevs
    attr_accessor :next
    attr_accessor :layer_node
    attr_reader :num_outputs

    def initialize(prevs: nil, layer_node: nil, num_outputs: 1)
      @prevs = prevs
      @layer_node = layer_node
      @num_outputs = num_outputs
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
      @hold << dy
      return if @hold.length < @num_outputs
      dys = @layer_node.backward_node(*@hold)
      @hold = []
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
