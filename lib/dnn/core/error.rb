module DNN
  class DNN_Error < StandardError; end

  class DNN_TypeError < DNN_Error; end

  class DNN_SharpError < DNN_Error; end

  class DNN_GradUnfairError < DNN_Error
    def initialize(grad, n_grad)
      super("gradient is #{grad}, but numerical gradient is #{n_grad}")
    end
  end
end
