module DNN
  if defined? ::Cumo
    Xumo = ::Cumo
  else
    require "numo/narray"
    Xumo = ::Numo
  end
end

require_relative "dnn/version"
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
