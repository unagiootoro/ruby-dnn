if RUBY_PLATFORM == "wasm32-wasi"
  require "narray.so"
else
  require "numo/narray"
end

if ENV["RUNY_DNN_USE_NUMO_LINALG"] == "ENABLE"
  require "numo/linalg/autoloader"
end

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

if RUBY_PLATFORM != "wasm32-wasi"
  require_relative "dnn/version"
  require_relative "dnn/core/monkey_patch"
  require_relative "dnn/core/error"
  require_relative "dnn/core/progress_bar"
  require_relative "dnn/core/base_tensor"
  require_relative "dnn/core/tensor"
  require_relative "dnn/core/variable"
  require_relative "dnn/core/link"
  require_relative "dnn/core/iterator"
  require_relative "dnn/core/models"
  require_relative "dnn/core/functions"
  require_relative "dnn/core/layers/basic_layers"
  require_relative "dnn/core/layers/normalizations"
  require_relative "dnn/core/layers/activations"
  require_relative "dnn/core/layers/cnn_layers"
  require_relative "dnn/core/layers/embedding"
  require_relative "dnn/core/layers/rnn_layers"
  require_relative "dnn/core/optimizers"
  require_relative "dnn/core/losses"
  require_relative "dnn/core/initializers"
  require_relative "dnn/core/regularizers"
  require_relative "dnn/core/callbacks"
  require_relative "dnn/core/savers"
  require_relative "dnn/core/utils"
  require_relative "dnn/core/process_runner"
  require_relative "dnn/core/evaluator"
  require_relative "dnn/core/predictor"
  require_relative "dnn/core/trainer"
end
