module DNN
  class Link
    attr_reader :prevs
    attr_reader :node
    attr_reader :num_outputs

    def initialize(prevs: nil, node: nil, num_outputs: 1)
      @prevs = prevs
      @node = node
      @num_outputs = num_outputs
      @hold_datas = []
      @held_flags = []
      @requires_grad = nil
    end

    def requires_grad
      @requires_grad = check_requires_grad if @requires_grad == nil
      @requires_grad
    end

    private def check_requires_grad
      @prevs.each do |prev|
        return true if prev && prev.requires_grad
      end
      false
    end

    def backward(dy, index)
      @hold_datas[index] = dy
      @held_flags[index] = true
      return if @held_flags.compact.length < @num_outputs
      return unless requires_grad
      dys = @node.backward(*@hold_datas)
      @hold_datas = []
      @held_flags = []
      if dys.is_a?(Array)
        dys.each.with_index do |dy, i|
          if @prevs[i].is_a?(Tensor)
            link_index = @prevs[i].next_link_index(self)
            @prevs[i].backward(dy, link_index) if @prevs[i] && @prevs[i].requires_grad
          else
            @prevs[i].backward(dy) if @prevs[i] && @prevs[i].requires_grad
          end
        end
      else
        if @prevs.first.is_a?(Tensor)
          link_index = @prevs.first.next_link_index(self)
          @prevs.first.backward(dys, link_index) if @prevs.first && @prevs.first.requires_grad
        else
          @prevs.first.backward(dys) if @prevs.first && @prevs.first.requires_grad
        end
      end
    end
  end
end
