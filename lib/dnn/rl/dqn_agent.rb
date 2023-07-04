module DNN
  module RL
    class DQNAgent < Agent
      def initialize(model, env,
                    max_memory_size: 1024,
                    train_memory_size: nil,
                    batch_size: 64,
                    gamma: 0.99,
                    policy: Policies::EpsGreedy.new,
                    ddqn: true)
        super(model, env, max_memory_size: max_memory_size, train_memory_size: train_memory_size, batch_size: batch_size,
              gamma: gamma, policy: policy)
        @ddqn = ddqn
      end

      def pre_epispde
        if @ddqn
          @target_model = @model.copy
        end
      end

      def replay
        sum_loss = 0
        steps = @memory.size / @batch_size
        return nil if steps == 0
        mem = @memory.shuffle
        @target_model = @model unless @ddqn
        steps.times do
          x, y = make_batch(mem.shift(@batch_size))
          sum_loss += @model.train_on_batch(x, y)
          @target_model = @model.copy unless @ddqn
        end
        sum_loss / steps
      end

      def compute_q_value(model, target_model, reward, next_state)
        if next_state
          q_values = model.predict1(next_state)
          next_action = q_values.max_index
          next_q_value = target_model.predict1(next_state)[next_action]
          reward + @gamma * next_q_value
        else
          reward
        end
      end
    end
  end
end
