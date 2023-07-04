module DNN
  module RL
    class Agent < ProcessRunner
      attr_reader :model
      attr_reader :env
      attr_accessor :gamma

      # @param [DNN::Models::Model] model Model used for training.
      # @param [RL::Env] env Training environment.
      # @param [Integer] max_memory_size Max size of memory for storing actions.
      # @param [Integer | NilClass] train_memory_size Memory size where training can start.
      #                                               Setting nil sets batch_size to train_memory_size.
      # @param [Integer] batch_size Batch size used for one training.
      # @param [Float] gamma Discount rate of reward.
      # @param [RL::Policies::Policy] policy The policy to use for training.
      def initialize(model, env,
                    max_memory_size: 1024,
                    train_memory_size: nil,
                    batch_size: 64,
                    gamma: 0.99,
                    policy: Policies::EpsGreedy.new)
        super()
        @model = model
        @env = env
        @memory = Memory.new(max_memory_size, train_memory_size || batch_size)
        @batch_size = batch_size
        @gamma = gamma
        @target_model = nil
        @policy = policy
      end

      def replay
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'replay'"
      end

      def compute_q_value(reward, next_state)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'compute_q_value'"
      end

      # def pre_epispde; end
      # def post_episode; end

      # Start training.
      # @param [Integer] epochs Number of training.
      def train(epochs)
        (1..epochs).each do |episode|
          puts "episode: #{episode}"
          pre_epispde if respond_to?(:pre_epispde)
          observation = nil
          @env.max_step.times do |step|
            if step == 0
              observation, reward, done = *@env.step(nil)
            else
              action = get_action(observation, episode)
              next_observation, reward, done = *@env.step(action)
              next_observation = nil if done
              @memory.add([observation, action, reward, next_observation])
              observation = next_observation unless done
            end
            if done
              add_log(:step, step)
              add_log(:observation, observation)
              add_log(:action, action)
              add_log(:reward, reward)
              call_callbacks(:before_replay)
              loss = replay if @memory.can_train?
              add_log(:loss, loss)
              call_callbacks(:after_replay)
              @env.reset
              break
            end
          end
          post_episode if respond_to?(:post_episode)
        end
      end

      # Render the environment based on the learning results.
      # @param [Integer] max_steps Max rendering times.
      def run(max_steps: 200)
        call_callbacks(:before_run)
        observation = nil
        logging = true
        max_steps.times do |step|
          call_callbacks(:before_running)
          action = if step == 0
            rand(@env.action_size)
          else
            result = @model.predict1(Numo::SFloat.cast(observation))
            result.max_index
          end
          observation, reward, done, info = *@env.step(action)
          if logging
            add_log(:step, step)
            add_log(:observation, observation)
            add_log(:action, action)
            add_log(:reward, reward)
            call_callbacks(:after_running)
          end
          logging = false if done
          @env.render
        end
        call_callbacks(:after_run)
      end

      def make_batch(memory_batch)
        x = Numo::SFloat.zeros(@batch_size, @env.state_size)
        y = Numo::SFloat.zeros(@batch_size, @env.action_size)
        memory_batch.each.with_index do |(state, action, reward, next_state), i|
          x[i, false] = state
          q_value = compute_q_value(@model, @target_model, reward, next_state)
          q_values = @model.predict1(state)
          q_values[action] = q_value
          y[i, false] = q_values
        end
        [x, y]
      end

      def get_action(observation, episode)
        if @policy.predict?(episode) && observation
          res = @model.predict1(observation)
          action = res.max_index
        else
          action = rand(@env.action_size)
        end
        action
      end
    end
  end
end
