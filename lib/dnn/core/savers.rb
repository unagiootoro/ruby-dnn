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

      def load_bin(bin)
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'load_bin'"
      end
    end

    class MarshalLoader < Loader
      def load_bin(bin)
        data = Marshal.load(Zlib::Inflate.inflate(bin))
        unless @model.class.name == data[:class]
          raise DNNError, "Class name is mismatch. Target model is #{@model.class.name}. But loading model is #{data[:class]}."
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
      def load_bin(bin)
        data = JSON.parse(bin, symbolize_names: true)
        unless @model.class.name == data[:class]
          raise DNNError, "Class name is mismatch. Target model is #{@model.class.name}. But loading model is #{data[:class]}."
        end
        set_all_params_base64_data(data[:params])
      end

      private def set_all_params_base64_data(params_data)
        @model.layers.each.with_index do |layer, i|
          params_data[i].each do |(key, (shape, base64_data))|
            bin = Base64.decode64(base64_data)
            data = Xumo::SFloat.from_binary(bin).reshape(*shape)
            layer.get_variables[key].data = data
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

      def dump_bin
        raise NotImplementedError, "Class '#{self.class.name}' has implement method 'dump_bin'"
      end
    end

    class MarshalSaver < Saver
      def initialize(model, include_model: true)
        super(model)
        @include_model = include_model
      end

      def dump_bin
        params_data = @model.get_all_params_data
        if @include_model
          @model.clean_layers
          data = {
            version: VERSION, class: @model.class.name,
            params: params_data, model: @model
          }
        else
          data = { version: VERSION, class: @model.class.name, params: params_data }
        end
        bin = Zlib::Deflate.deflate(Marshal.dump(data))
        @model.set_all_params_data(params_data) if @include_model
        bin
      end
    end

    class JSONSaver < Saver
      def dump_bin
        data = { version: VERSION, class: @model.class.name, params: get_all_params_base64_data }
        JSON.dump(data)
      end

      private def get_all_params_base64_data
        @model.layers.map do |layer|
          layer.get_variables.to_h do |key, param|
            base64_data = Base64.encode64(param.data.to_binary)
            [key, [param.data.shape, base64_data]]
          end
        end
      end
    end

  end
end
