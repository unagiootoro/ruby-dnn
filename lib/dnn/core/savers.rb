require "zlib"
require "json"
require "base64"

module DNN
  module Loaders

    class Loader
      def initialize(model)
        @model = model
      end

      def load(file_name)
        load_bin(File.binread(file_name))
      end

      private

      def load_bin(bin)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'load_bin'"
      end
    end

    class MarshalLoader < Loader
      private def load_bin(bin)
        data = Marshal.load(Zlib::Inflate.inflate(bin))
        unless @model.class.name == data[:class]
          raise DNN_Error, "Class name is not mismatch. Target model is #{@model.class.name}. But loading model is #{data[:class]}."
        end
        if data[:model]
          data[:model].instance_variables.each do |ivar|
            obj = data[:model].instance_variable_get(ivar)
            @model.instance_variable_set(ivar, obj)
          end
        end
        @model.set_all_params_data(data[:params])
      end
    end

    class JSONLoader < Loader
      private

      def load_bin(bin)
        data = JSON.parse(bin, symbolize_names: true)
        opt = Optimizers::Optimizer.from_hash(data[:optimizer])
        loss_func = Losses::Loss.from_hash(data[:loss_func])
        @model.setup(opt, loss_func)
        @model.load_hash(data[:layers_hash]) if data[:layers_hash]
        @model.instance_variable_set(:@built, false)
        @model.predict1(Xumo::SFloat.zeros(*data[:input_shape]))
        set_all_params_base64_data(data[:params])
      end

      def set_all_params_base64_data(params_data)
        @model.trainable_layers.each.with_index do |layer, i|
          params_data[i].each do |(key, (shape, base64_data))|
            bin = Base64.decode64(base64_data)
            data = Xumo::SFloat.from_binary(bin).reshape(*shape)
            layer.get_params[key].data = data
          end
        end
      end
    end

  end

  module Savers

    class Saver
      def initialize(model)
        @model = model
      end

      def save(file_name)
        bin = dump_bin
        begin
          File.binwrite(file_name, bin)
        rescue Errno::ENOENT
          dir_name = file_name.match(%r`(.*)/.+$`)[1]
          Dir.mkdir(dir_name)
          File.binwrite(file_name, bin)
        end
      end

      private

      def dump_bin
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'dump_bin'"
      end
    end

    class MarshalSaver < Saver
      def initialize(model, include_model: true)
        super(model)
        @include_model = include_model
      end

      private def dump_bin
        if @include_model
          data = {
            version: VERSION, class: @model.class.name, input_shape: @model.layers.first.input_shape,
            params: @model.get_all_params_data, model: @model.dump
          }
        else
          data = {
            version: VERSION, class: @model.class.name, params: @model.get_all_params_data
          }
        end
        Zlib::Deflate.deflate(Marshal.dump(data))
      end
    end

    class JSONSaver < Saver
      private

      def dump_bin
        data = {
          version: VERSION, class: @model.class.name, input_shape: @model.layers.first.input_shape, params: get_all_params_base64_data,
          layers_hash: @model.to_hash, optimizer: @model.optimizer.to_hash, loss_func: @model.loss_func.to_hash
        }
        JSON.dump(data)
      end

      def get_all_params_base64_data
        @model.trainable_layers.map do |layer|
          layer.get_params.to_h do |key, param|
            base64_data = Base64.encode64(param.data.to_binary)
            [key, [param.data.shape, base64_data]]
          end
        end
      end
    end

  end
end
