module DNN
  module RL
    module Policies
  
      class Policy
        def predict?(episode)
          raise NotImplementedError, "Class '#{self.class.name}' has implement method 'predict?'"
        end
      end
  
      class EpsGreedy < Policy
        def initialize(base_eps: 0.0001, initial_eps: 0.5, decay: 0.9)
          @eps = initial_eps
          @base_eps = base_eps
          @decay = decay
        end
  
        def predict?(episode)
          eps = @base_eps + @eps
          return false if eps > 1
          @eps *= @decay
          eps <= rand
        end
      end
  
    end
  end  
end
