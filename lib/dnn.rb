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
require_relative "dnn/core/models"
require_relative "dnn/core/param"
require_relative "dnn/core/link"
require_relative "dnn/core/iterator"
require_relative "dnn/core/initializers"
require_relative "dnn/core/layers"
require_relative "dnn/core/normalizations"
require_relative "dnn/core/activations"
require_relative "dnn/core/merge_layers"
require_relative "dnn/core/losses"
require_relative "dnn/core/regularizers"
require_relative "dnn/core/cnn_layers"
require_relative "dnn/core/embedding"
require_relative "dnn/core/rnn_layers"
require_relative "dnn/core/optimizers"
require_relative "dnn/core/utils"
