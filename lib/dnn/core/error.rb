module DNN
  class DNN_Error < StandardError; end

  class DNN_ShapeError < DNN_Error; end

  class DNN_UnknownEventError < DNN_Error; end
end
