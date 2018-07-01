require "numo/narray"

Numo::SFloat.srand(rand(2**64))

module DNN; end

require "dnn/core/version"
require "dnn/core/error"
require "dnn/core/model"
require "dnn/core/initializers"
require "dnn/core/layers"
require "dnn/core/activations"
require "dnn/core/optimizers"
require "dnn/core/util"
