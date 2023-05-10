module DNN
  class Variable < BaseTensor
    attr_accessor :requires_grad
    attr_accessor :grad
    attr_writer :data

    def initialize(*args, requires_grad: true)
      if args.length == 0
        data = nil
        grad = nil
      elsif args.length == 1
        data = args[0]
        grad = nil
      elsif args.length == 2
        data = args[0]
        grad = args[1]
      else
        raise ArgumentError, "wrong number of arguments (given #{args.length}, expected 0 or 1 or 2)"
      end
      super(data)
      @grad = grad
      @requires_grad = requires_grad
    end

    private def backward_internal(grad)
      if @requires_grad
        @grad ||= Xumo::SFloat[0]
        if @data.shape == grad.shape || (grad.shape[-1] == 1 && @data.shape == grad.shape[0...-1])
          @grad += grad
        elsif @data.shape == grad.shape[1..-1]
          @grad += grad.sum(0)
        else
          p [@data.shape, grad.shape]
          raise DNNError, "Shape is missmatch."
        end
      else
        @grad = Xumo::SFloat[0]
      end
    end
  end
end
