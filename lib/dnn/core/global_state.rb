module DNN
  module GlobalState
    def self.learning_phase
      @learning_phase
    end

    def self.learning_phase=(bool)
      @learning_phase = bool
    end
  end
end
