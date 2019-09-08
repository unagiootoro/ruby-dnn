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
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'load_bin'")
      end

      def set_all_params(layer_params)
        @model.has_param_layers.uniq.each.with_index do |layer, index|
          layer_params[index].each do |key, data|
            layer.get_params[key].data = data
          end
        end
      end
    end


    class MarshalLoader < Loader
      private def load_bin(bin)
        data = Marshal.load(Zlib::Inflate.inflate(bin))
        opt = Optimizers::Optimizer.load(data[:optimizer])
        loss_func = Utils.hash_to_obj(data[:loss_func])
        @model.setup(opt, loss_func)
        @model.predict1(Xumo::SFloat.zeros(*data[:input_shape]))
        set_all_params(data[:params])
      end
    end

    class JSONLoader < Loader
      private

      def load_bin(bin)
        data = JSON.parse(bin, symbolize_names: true)
        opt = Utils.hash_to_obj(data[:optimizer])
        loss_func = Utils.hash_to_obj(data[:loss_func])
        @model.setup(opt, loss_func)
        @model.predict1(Xumo::SFloat.zeros(*data[:input_shape]))
        base64_to_params(data[:params])
      end

      def base64_to_params(base64_params)
        layer_params = base64_params.map do |params|
          params.map { |key, (shape, base64_data)|
            bin = Base64.decode64(base64_data)
            [key, Xumo::SFloat.from_binary(bin).reshape(*shape)]
          }.to_h
        end
        set_all_params(layer_params)
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
        raise NotImplementedError.new("Class '#{self.class.name}' has implement method 'dump_bin'")
      end

      def get_all_params
        @model.has_param_layers.uniq.map do |layer|
          layer.get_params.map { |key, param| [key, param.data] }.to_h
        end
      end
    end


    class MarshalSaver < Saver
      def initialize(model, include_optimizer: true)
        super(model)
        @include_optimizer = include_optimizer
      end

      private def dump_bin
        opt = @include_optimizer ? @model.optimizer.dump : @model.optimizer.class.new.dump
        data = {
          version: VERSION, class: @model.class.name, input_shape: @model.layers.first.input_shape, params: get_all_params,
          optimizer: opt, loss_func: @model.loss_func.to_hash
        }
        Zlib::Deflate.deflate(Marshal.dump(data))
      end
    end

    class JSONSaver < Saver
      private

      def dump_bin
        data = {
          version: VERSION, class: @model.class.name, input_shape: @model.layers.first.input_shape, params: params_to_base64,
          optimizer: @model.optimizer.to_hash, loss_func: @model.loss_func.to_hash
        }
        JSON.dump(data)
      end

      def params_to_base64
        get_all_params.map do |params|
          params.map { |key, data|
            base64_data = Base64.encode64(data.to_binary)
            [key, [data.shape, base64_data]]
          }.to_h
        end
      end
    end

  end
end
