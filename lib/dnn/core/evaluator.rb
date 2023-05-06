module DNN
  module EvaluatorImpl
    # Evaluate model and get accuracy and loss of test data.
    # @param [Numo::SFloat] x Input test data.
    # @param [Numo::SFloat] y Output test data.
    # @param [Integer] batch_size Batch size used for one test.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
    #                 If accuracy is not needed returns in the form [nil, mean_loss].
    def evaluate(x, y, batch_size: 100, need_accuracy: true)
      Utils.check_input_data_type("x", x, Xumo::SFloat)
      Utils.check_input_data_type("y", y, Xumo::SFloat)
      start_evaluate(x, y, batch_size: batch_size, need_accuracy: need_accuracy)
      update while evaluating?
      [@last_logs[:test_accuracy], @last_logs[:test_loss]]
    end

    # Evaluate model by iterator.
    # @param [DNN::Iterator] test_iterator Iterator used for testing.
    # @param [Integer] batch_size Batch size used for one test.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
    #                 If accuracy is not needed returns in the form [nil, mean_loss].
    def evaluate_by_iterator(test_iterator, batch_size: 100, need_accuracy: true)
      start_evaluate_by_iterator(test_iterator, batch_size: batch_size, need_accuracy: need_accuracy)
      update while evaluating?
      [@last_logs[:test_accuracy], @last_logs[:test_loss]]
    end

    # Start evaluate model and get accuracy and loss of test data.
    # @param [Numo::SFloat] x Input test data.
    # @param [Numo::SFloat] y Output test data.
    # @param [Integer] batch_size Batch size used for one test.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
    #                 If accuracy is not needed returns in the form [nil, mean_loss].
    def start_evaluate(x, y, batch_size: 100, need_accuracy: true)
      Utils.check_input_data_type("x", x, Xumo::SFloat)
      Utils.check_input_data_type("y", y, Xumo::SFloat)
      start_evaluate_by_iterator(Iterator.new(x, y, random: false), batch_size: batch_size, need_accuracy: need_accuracy)
    end

    # Start Evaluate model by iterator.
    # @param [DNN::Iterator] test_iterator Iterator used for testing.
    # @param [Integer] batch_size Batch size used for one test.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @return [Array] Returns the test data accuracy and mean loss in the form [accuracy, mean_loss].
    #                 If accuracy is not needed returns in the form [nil, mean_loss].
    def start_evaluate_by_iterator(test_iterator, batch_size: 100, need_accuracy: true)
      @test_iterator = test_iterator
      @num_test_datas = test_iterator.num_datas
      @batch_size = batch_size >= @num_test_datas ? @num_test_datas : batch_size
      @need_accuracy = need_accuracy
      if @loss_func.is_a?(Array)
        @total_correct = Array.new(@loss_func.length, 0)
        @sum_test_loss = Array.new(@loss_func.length, 0)
      else
        @total_correct = 0
        @sum_test_loss = 0
      end
      @evaluate_step = 1
      @evaluate_max_steps = (@num_test_datas.to_f / @batch_size).ceil
      @evaluate_state = :start_evaluate_step
    end

    # Check if it is currently evaluating.
    # @return [Boolean] Returns true if currently evaluating.
    def evaluating?
      @evaluate_state != :none
    end

    private

    def init_evaluator_impl
      @evaluate_state = :none
    end

    def on_test_step_default(model, x_batch, y_batch)
      model.set_learning_phase(false)
      output_tensors = model.(Tensor.new(x_batch))
      if output_tensors.is_a?(Array)
        output_data = []
        loss_data = []
        output_tensors.each.with_index do |out, i|
          output_data << out.data
          loss = model.loss_func[i].(out, Tensor.new(y_batch[i]))
          loss_data << loss.data
        end
      else
        out = output_tensors
        output_data = out.data
        loss = model.loss_func.(out, Tensor.new(y_batch))
        loss_data = loss.data
      end

      if loss_data.is_a?(Array)
        loss_value = []
        accuracy = []
        loss_data.each_index do |i|
          loss_value << Utils.to_f(loss_data)
          accuracy << accuracy(output_data[i], y_batch[i]).to_f / y_batch[i].shape[0]
        end
      else
        loss_value = Utils.to_f(loss_data)
      end
      { test_loss: loss_value, test_accuracy: model.accuracy(output_data, y_batch) }
    end

    def update_evaluator_impl
      case @evaluate_state
      when :start_evaluate_step
        start_evaluate_step
      when :test_step
        test_step
      when :end_evaluate_step
        end_evaluate_step
      when :end_evaluate
        end_evaluate
      end
    end

    def start_evaluate_step
      @last_logs[:step] = @evaluate_step
      @evaluate_state = :test_step
    end

    # Testing process to be performed in one step.
    # @param [Numo::SFloat] x Input training data.
    # @param [Numo::SFloat] y Output training data.
    # @return [Hash] Hash of contents to be output to log.
    def test_step
      batches = @test_iterator.next_batch(@batch_size)
      call_callbacks(:before_test_on_batch)
      test_met = on_test_step(*batches)
      call_callbacks(:after_test_on_batch)
      if @loss_func.is_a?(Array)
        @loss_func.each_index do |i|
          @total_correct[i] += test_met[:test_accuracy][i] if @need_accuracy
          @sum_test_loss[i] += test_met[:test_loss][i]
        end
      else
        @total_correct += test_met[:test_accuracy] if @need_accuracy
        @sum_test_loss += test_met[:test_loss]
      end
      @evaluate_state = :end_evaluate_step
    end

    def end_evaluate_step
      @evaluate_step += 1
      if @evaluate_step <= @evaluate_max_steps
        @evaluate_state = :start_evaluate_step
      else
        @evaluate_state = :end_evaluate
      end
    end

    def end_evaluate
      acc = nil
      if @loss_func.is_a?(Array)
        mean_loss = Array.new(@loss_func.length, 0)
        acc = Array.new(@loss_func.length, 0) if @need_accuracy
        @loss_func.each_index do |i|
          mean_loss[i] += @sum_test_loss[i] / @evaluate_max_steps
          acc[i] += @total_correct[i].to_f / @num_test_datas if @need_accuracy
        end
      else
        mean_loss = @sum_test_loss / @evaluate_max_steps
        acc = @total_correct.to_f / @num_test_datas if @need_accuracy
      end
      @last_logs[:test_loss] = mean_loss
      @last_logs[:test_accuracy] = acc
      @evaluate_state = :none
    end
  end

  class BaseEvaluator < ProcessRunner
    include EvaluatorImpl

    def initialize
      super()
      init_evaluator_impl
    end

    # Update evaluator status.
    def update
      update_evaluator_impl
    end
  end

  class Evaluator < BaseEvaluator
    def initialize(model)
      super()
      @model = model
    end

    def check_model_setup_complete
      raise DNNError, "The model is not loss_func setup complete." unless @model.loss_func
    end

    def on_test_step(x_batch, y_batch)
      on_test_step_default(@model, x_batch, y_batch)
    end
  end
end
