require "numo/narray"

module DNN
  if ENV["RUBY_DNN_USE_CUMO"] == "ENABLE"
    require "cumo/narray"
    Xumo = ::Cumo
  else
    if defined? ::Cumo
      Xumo = ::Cumo
    else
      Xumo = ::Numo
    end
  end

  def self.use_cumo?
    defined? ::Cumo
  end

  def self.cudnn_available?
    return false unless defined? ::Cumo
    Cumo::CUDA::CUDNN.available?
  end

  def self.use_cudnn?
    return false unless ENV["RUBY_DNN_USE_CUDNN"] == "ENABLE"
    cudnn_available?
  end
end

require_relative "dnn/version"
require_relative "dnn/core/monkey_patch"
require_relative "dnn/core/error"
require_relative "dnn/core/global"
require_relative "dnn/core/tensor"
require_relative "dnn/core/param"
require_relative "dnn/core/link"
require_relative "dnn/core/iterator"
require_relative "dnn/core/models"
require_relative "dnn/core/layers/basic_layers"
require_relative "dnn/core/layers/normalizations"
require_relative "dnn/core/layers/activations"
require_relative "dnn/core/layers/merge_layers"
require_relative "dnn/core/layers/split_layers"
require_relative "dnn/core/layers/cnn_layers"
require_relative "dnn/core/layers/embedding"
require_relative "dnn/core/layers/rnn_layers"
require_relative "dnn/core/layers/math_layers"
require_relative "dnn/core/optimizers"
require_relative "dnn/core/losses"
require_relative "dnn/core/initializers"
require_relative "dnn/core/regularizers"
require_relative "dnn/core/callbacks"
require_relative "dnn/core/savers"
require_relative "dnn/core/utils"
