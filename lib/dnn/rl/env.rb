module DNN
  module RL
    class Env
      attr_reader :state_size
      attr_reader :action_size
      attr_reader :max_step

      def initialize(state_size, action_size, max_step)
        @state_size = state_size
        @action_size = action_size
        @max_step = max_step
      end

      def step
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'step'"
      end

      def reset
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'reset'"
      end

      def render
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'render'"
      end
    end
  end
end
