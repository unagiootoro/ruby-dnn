module DNN
  module PredictorImpl
    attr_reader :predicted_data
    attr_accessor :last_predicted_batch

    # Predict model and get accuracy and loss of predict data.
    # @param [Numo::SFloat] x Input predict data.
    # @param [Integer] batch_size Batch size used for one predict.
    # @return [Array] Returns the predict data accuracy and mean loss in the form [accuracy, mean_loss].
    #                 If accuracy is not needed returns in the form [nil, mean_loss].
    def predict(x, batch_size: 100)
      Utils.check_input_data_type("x", x, Xumo::SFloat)
      start_predict(x, batch_size: batch_size)
      update while predicting?
      predicted_data
    end

    # Predict model by iterator.
    # @param [DNN::Iterator] predict_iterator Iterator used for predicting.
    # @param [Integer] batch_size Batch size used for one predict.
    # @return [Array] Returns the predict data accuracy and mean loss in the form [accuracy, mean_loss].
    #                 If accuracy is not needed returns in the form [nil, mean_loss].
    def predict_by_iterator(predict_iterator, batch_size: 100)
      start_predict_by_iterator(predict_iterator, batch_size: batch_size)
      update while evaluating?
      predicted_data
    end

    # Start predict model and get accuracy and loss of predict data.
    # @param [Numo::SFloat] x Input data.
    # @param [Integer] batch_size Batch size used for one predict.
    # @return [Array] Returns the predict data accuracy and mean loss in the form [accuracy, mean_loss].
    #                 If accuracy is not needed returns in the form [nil, mean_loss].
    def start_predict(x, batch_size: 100)
      Utils.check_input_data_type("x", x, Xumo::SFloat)
      start_predict_by_iterator(Iterator.new(x, random: false), batch_size: batch_size)
    end

    # Start Predict model by iterator.
    # @param [DNN::Iterator] predict_iterator Iterator used for predicting.
    # @param [Integer] batch_size Batch size used for one predict.
    # @return [Array] Returns the predict data accuracy and mean loss in the form [accuracy, mean_loss].
    #                 If accuracy is not needed returns in the form [nil, mean_loss].
    def start_predict_by_iterator(predict_iterator, batch_size: 100)
      @predicted_data = nil
      @last_predicted_batch = nil
      @predict_iterator = predict_iterator
      @num_predict_datas = predict_iterator.num_datas
      @batch_size = batch_size >= @num_predict_datas ? @num_predict_datas : batch_size
      @predict_step = 1
      @predict_max_steps = (@num_predict_datas.to_f / @batch_size).ceil
      @predict_state = :start_predict_step
    end

    # Check if it is currently predicting.
    # @return [Boolean] Returns true if currently predicting.
    def predicting?
      @predict_state != :none
    end

    private

    def init_predictor_impl
      @predict_state = :none
      @predicted_data = nil
      @last_predicted_batch = nil
    end

    def on_predict_step_default(model, x_batch)
      model.set_learning_phase(false)
      output_tensors = model.(Tensor.new(x_batch))
      if output_tensors.is_a?(Array)
        output_data = []
        output_tensors.each.with_index do |out, i|
          output_data << out.data
        end
      else
        out = output_tensors
        output_data = out.data
      end
      output_data
    end

    def update_predictor_impl
      case @predict_state
      when :start_predict_step
        start_predict_step
      when :predict_step
        predict_step
      when :end_predict_step
        end_predict_step
      when :end_predict
        end_predict
      end
    end

    def start_predict_step
      @last_logs[:step] = @predict_step
      @predict_state = :predict_step
    end

    # Predicting process to be performed in one step.
    # @param [Numo::SFloat] x Input training data.
    # @param [Numo::SFloat] y Output training data.
    # @return [Hash] Hash of contents to be output to log.
    def predict_step
      batches = @predict_iterator.next_batch(@batch_size)
      call_callbacks(:before_predict_on_batch)
      @last_predicted_batch = on_predict_step(*batches)
      call_callbacks(:after_predict_on_batch)
      if @predicted_data
        @predicted_data = @predicted_data.concatenate(@last_predicted_batch, axis: 0)
      else
        @predicted_data = @last_predicted_batch
      end
      @predict_state = :end_predict_step
    end

    def end_predict_step
      @predict_step += 1
      if @predict_step <= @predict_max_steps
        @predict_state = :start_predict_step
      else
        @predict_state = :end_predict
      end
    end

    def end_predict
      @predict_state = :none
    end
  end

  class BasePredictor < ProcessRunner
    include PredictorImpl

    def initialize
      super()
      init_predictor_impl
    end

    # Update predictor status.
    def update
      update_predictor_impl
    end
  end

  class Predictor < BasePredictor
    def initialize(model)
      super()
      @model = model
    end

    def check_model_setup_complete
      raise DNNError, "The model is not loss_func setup complete." unless @model.loss_func
    end

    def on_predict_step(x_batch)
      on_predict_step_default(@model, x_batch)
    end
  end
end
