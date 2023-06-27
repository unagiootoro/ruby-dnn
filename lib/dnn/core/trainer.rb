module DNN
  module TrainerImpl
    # Start training.
    # Setup the model before use this method.
    # @param [Numo::SFloat] x Input training data.
    # @param [Numo::SFloat] y Output training data.
    # @param [Integer] epochs Number of training.
    # @param [Integer] batch_size Batch size used for one training.
    # @param [Integer] initial_epoch Initial epoch.
    # @param [Array | NilClass] test If you to test the model for every 1 epoch,
    #                                specify [x_test, y_test]. Don't test to the model, specify nil.
    # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @param [IO] io Specifies the IO object to use for logging.
    def fit(x, y, epochs,
      batch_size: 1,
      initial_epoch: 1,
      test: nil,
      verbose: true,
      need_accuracy: true,
      io: $stdout)
      start_fit(x, y, epochs,
                batch_size: batch_size,
                initial_epoch: initial_epoch,
                test: test,
                verbose: verbose,
                need_accuracy: need_accuracy,
                io: io)
      update while training?
    end

    # Start training by iterator.
    # Setup the model before use this method.
    # @param [DNN::Iterator] train_iterator Iterator used for training.
    # @param [Integer] epochs Number of training.
    # @param [Integer] batch_size Batch size used for one training.
    # @param [Integer] initial_epoch Initial epoch.
    # @param [Array | NilClass] test If you to test the model for every 1 epoch,
    #                                specify [x_test, y_test]. Don't test to the model, specify nil.
    # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @param [IO] io Specifies the IO object to use for logging.
    def fit_by_iterator(train_iterator, epochs,
                        batch_size: 1,
                        initial_epoch: 1,
                        test: nil,
                        verbose: true,
                        need_accuracy: true,
                        io: $stdout)
      start_fit_by_iterator(train_iterator, epochs,
                            batch_size: batch_size,
                            initial_epoch: initial_epoch,
                            test: test,
                            verbose: verbose,
                            need_accuracy: need_accuracy,
                            io: io)
      update while training?
    end

    # Start training.
    # Setup the model before use this method.
    # @param [Numo::SFloat] x Input training data.
    # @param [Numo::SFloat] y Output training data.
    # @param [Integer] epochs Number of training.
    # @param [Integer] batch_size Batch size used for one training.
    # @param [Integer] initial_epoch Initial epoch.
    # @param [Array | NilClass] test If you to test the model for every 1 epoch,
    #                                specify [x_test, y_test]. Don't test to the model, specify nil.
    # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @param [IO] io Specifies the IO object to use for logging.
    def start_fit(x, y, epochs,
                batch_size: 1,
                initial_epoch: 1,
                test: nil,
                verbose: true,
                need_accuracy: true,
                io: $stdout)
      Utils.check_input_data_type("x", x, Xumo::SFloat)
      Utils.check_input_data_type("y", y, Xumo::SFloat)
      train_iterator = Iterator.new(x, y)
      start_fit_by_iterator(train_iterator, epochs,
                            batch_size: batch_size,
                            initial_epoch: initial_epoch,
                            test: test,
                            verbose: verbose,
                            need_accuracy: need_accuracy,
                            io: io)
    end

    # Start training by iterator.
    # Setup the model before use this method.
    # @param [DNN::Iterator] train_iterator Iterator used for training.
    # @param [Integer] epochs Number of training.
    # @param [Integer] batch_size Batch size used for one training.
    # @param [Integer] initial_epoch Initial epoch.
    # @param [Array | NilClass] test If you to test the model for every 1 epoch,
    #                                specify [x_test, y_test]. Don't test to the model, specify nil.
    # @param [Boolean] verbose Set true to display the log. If false is set, the log is not displayed.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @param [IO] io Specifies the IO object to use for logging.
    def start_fit_by_iterator(train_iterator, epochs,
                              batch_size: 1,
                              initial_epoch: 1,
                              test: nil,
                              verbose: true,
                              need_accuracy: true,
                              io: $stdout)
      check_model_setup_complete
      check_stop_requested # Clear stop request.
      @train_iterator = train_iterator
      @max_epochs = epochs
      @train_batch_size = batch_size
      @epoch = initial_epoch
      @test = test
      @verbose = verbose
      @need_accuracy = need_accuracy
      @io = io
      @train_state = :start_train_epoch
      @train_max_steps = train_iterator.max_steps(batch_size)
      @num_train_datas = train_iterator.num_usable_datas(batch_size)
      @line_first_pos = 0
      call_callbacks(:before_train)
    end

    # Check if it is currently evaluating.
    # @return [Boolean] Returns true if currently training.
    def training?
      @train_state != :none
    end

    # Request training early stop.
    def request_stop(message)
      @stop_requested_message = message
    end

    private

    def init_trainer_impl
      @train_state = :none
      @initial_epoch = 1
      @train_step = 1
      @train_max_steps = 1
      @train_iterator = nil
      @max_epochs = 1
      @train_batch_size = 1
      @epoch = 1
      @test = nil
      @verbose = false
      @need_accuracy = false
      @io = nil
      @num_train_datas = 0
      @stop_requested_message = nil
    end

    def on_train_step_default(model, x_batch, y_batch)
      model.set_learning_phase(true)
      x = Tensor.convert(x_batch)
      y = Tensor.convert(y_batch)
      outputs = model.(*x)
      losses = model.optimize(outputs, y)
      if losses.is_a?(Array)
        loss_value = []
        acc = [] if @need_accuracy
        losses.each_index do |i|
          loss_value << Utils.to_f(losses[i].data)
          acc << accuracy(outputs[i].data, y_batch[i]).to_f / y_batch[i].shape[0] if @need_accuracy
        end
      else
        loss_value = Utils.to_f(losses.data)
        acc = model.accuracy(outputs.data, y_batch).to_f / y_batch.shape[0] if @need_accuracy
      end
      if @need_accuracy
        { loss: loss_value, accuracy: acc }
      else
        { loss: loss_value }
      end
    end

    def update_trainer_impl
      case @train_state
      when :start_train_epoch
        start_train_epoch
      when :start_train_step
        start_train_step
      when :train_step
        train_step
      when :end_train_step
        end_train_step
      when :end_train_epoch
        end_train_epoch
      when :trainer_start_evaluate
        trainer_start_evaluate
      when :trainer_evaluating
        trainer_evaluating
      when :trainer_end_evaluate
        trainer_end_evaluate
      when :trainer_end_training
        trainer_end_training
      end
    end

    def check_model_setup_complete
    end

    def check_stop_requested
      if @stop_requested_message
        stop_requested_message = @stop_requested_message
        @stop_requested_message = nil
        return stop_requested_message
      end
      nil
    end

    def start_train_epoch
      @last_logs[:epoch] = @epoch
      call_callbacks(:before_epoch)
      if @verbose
        @io.puts "Epoch #{@epoch}/#{@max_epochs}"
        @progress_bar = ProgressBar.new(@num_train_datas, io: @io)
      end
      @train_step = 1
      @train_state = :start_train_step
    end

    def start_train_step
      @last_logs[:step] = @train_step
      @train_state = :train_step
    end

    # Implement the training process to be performed in one step.
    # @param [Numo::SFloat] x Input training data.
    # @param [Numo::SFloat] y Output training data.
    # @param [Boolean] need_accuracy Set true to compute the accuracy.
    # @return [Hash] Hash of contents to be output to log.
    def train_step
      batches = @train_iterator.next_batch(@train_batch_size)
      call_callbacks(:before_train_on_batch)
      train_step_met = on_train_step(*batches)
      @last_logs.merge!(train_step_met)
      call_callbacks(:after_train_on_batch)
      if @verbose
        @progress_bar.progress(@train_batch_size)
        metrics = metrics_to_str(train_step_met)
        if @io == $stdout
          @progress_bar.print(prepare: "\r", append: metrics)
        else
          @line_first_pos = @io.pos
          @progress_bar.print(append: metrics)
        end
      end
      stop_requested_message = check_stop_requested
      if stop_requested_message
        @io.puts("\n#{stop_requested_message}") if @verbose
        @train_state = :trainer_end_training
      else
        @train_state = :end_train_step
      end
    end

    def end_train_step
      @train_step += 1
      if @train_step <= @train_max_steps
        unless @io == $stdout
          @io.pos = @line_first_pos
        end
        @train_state = :start_train_step
      else
        @train_state = :end_train_epoch
      end
    end

    def end_train_epoch
      @epoch += 1
      if @test
        @train_state = :trainer_start_evaluate
      else
        @io.puts "" if @verbose
        call_callbacks(:after_epoch)
        if @epoch <= @max_epochs
          @train_iterator.reset
          @train_state = :start_train_epoch
        else
          @train_state = :none
        end
      end
    end

    def trainer_start_evaluate
      if @test.is_a?(Array)
        start_evaluate(@test[0], @test[1], batch_size: @train_batch_size, need_accuracy: @need_accuracy)
      else
        start_evaluate_by_iterator(iter, batch_size: @train_batch_size, need_accuracy: @need_accuracy)
      end
      @train_state = :trainer_evaluating
    end

    def trainer_evaluating
      unless evaluating?
        @train_state = :trainer_end_evaluate
      end
    end

    def trainer_end_evaluate
      if @verbose
        metrics = if @need_accuracy
                    { test_accuracy: @last_logs[:test_accuracy], test_loss: @last_logs[:test_loss] }
                  else
                    { test_loss: @last_logs[:test_loss] }
                  end
        @io.print "  " + metrics_to_str(metrics)
      end
      @io.puts "" if @verbose
      call_callbacks(:after_epoch)
      if @epoch <= @max_epochs
        @train_iterator.reset
        stop_requested_message = check_stop_requested
        if stop_requested_message
          @io.puts(stop_requested_message) if @verbose
          @train_state = :trainer_end_training
        else
          @train_state = :start_train_epoch
        end
      else
        @train_state = :trainer_end_training
      end
    end

    def trainer_end_training
      call_callbacks(:after_train)
      @train_state = :none
    end

    def metrics_to_str(mertics)
      mertics.map { |key, values|
        str_values = if values.is_a?(Array)
                       values_fmt = values.map { |v| sprintf('%.4f', Utils.to_f(v)) }
                       "[#{values_fmt.join(", ")}]"
                     else
                       sprintf('%.4f', Utils.to_f(values))
                     end
        "#{key}: #{str_values}"
      }.join(", ")
    end
  end

  class BaseTrainer < ProcessRunner
    include TrainerImpl
    include EvaluatorImpl
    include PredictorImpl

    def initialize
      super()
      init_trainer_impl
      init_evaluator_impl
      init_predictor_impl
    end

    # Update trainer status.
    def update
      update_evaluator_impl
      update_trainer_impl
      update_predictor_impl
    end
  end

  class Trainer < BaseTrainer
    def initialize(model)
      super()
      @model = model
    end

    def check_model_setup_complete
      raise DNNError, "The model is not optimizer setup complete." unless @model.optimizer
      raise DNNError, "The model is not loss_func setup complete." unless @model.loss_func
    end

    def on_train_step(x_batch, y_batch)
      on_train_step_default(@model, x_batch, y_batch)
    end

    def on_test_step(x_batch, y_batch)
      on_test_step_default(@model, x_batch, y_batch)
    end

    def on_predict_step(x_batch)
      on_predict_step_default(@model, x_batch)
    end
  end
end
