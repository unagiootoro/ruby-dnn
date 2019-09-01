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

      # Load hash model parameters.
      # @param [Hash] hash Hash to load model parameters.
      def hash_to_params(hash)
        has_param_layers_params = hash[:params]
        @model.has_param_layers.uniq.each.with_index do |layer, index|
          hash_params = has_param_layers_params[index]
          hash_params.each do |key, (shape, bin)|
            data = Xumo::SFloat.from_binary(bin).reshape(*shape)
            layer.get_params[key].data = data
          end
        end
      end
    end


    class MarshalLoader < Loader
      private def load_bin(bin)
        hash, opt, loss_hash = *Marshal.load(Zlib::Inflate.inflate(bin))
        hash_to_params(hash)
        @model.setup(opt, Utils.hash_to_obj(loss_hash))
      end
    end

    class JSONLoader < Loader
      private

      def load_bin(bin)
        json_to_params(bin)
      end

      # Load json model parameters.
      # @param [String] json_str JSON string to load model parameters.
      def json_to_params(json_str)
        hash = JSON.parse(json_str, symbolize_names: true)
        hash[:params] = hash[:params].map do |layer_hash|
          layer_hash.map { |key, (shape, base64_data)|
            bin = Base64.decode64(base64_data)
            [key, [shape, bin]]
          }.to_h
        end
        hash_to_params(hash)
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

      # Convert model parameters to hash.
      # @return [Hash] Return the hash of model parameters.
      def params_to_hash
        has_param_layers_params = @model.has_param_layers.uniq.map do |layer|
          layer.get_params.map { |key, param|
            [key, [param.data.shape, param.data.to_binary]]
          }.to_h
        end
        { version: VERSION, params: has_param_layers_params }
      end
    end


    class MarshalSaver < Saver
      def initialize(model, include_optimizer: true)
        super(model)
        @include_optimizer = include_optimizer
      end

      private def dump_bin
        opt = @include_optimizer ? @model.optimizer : @model.optimizer.class.new
        Zlib::Deflate.deflate(Marshal.dump([params_to_hash, opt, @model.loss_func.to_hash]))
      end
    end

    class JSONSaver < Saver
      private

      def dump_bin
        params_to_json
      end

      # Convert model parameters to JSON string.
      # @return [String] Return the JSON string.
      def params_to_json
        hash = params_to_hash
        hash[:params] = hash[:params].map do |layer_hash|
          layer_hash.map { |key, (shape, bin)|
            base64_data = Base64.encode64(bin)
            [key, [shape, base64_data]]
          }.to_h
        end
        JSON.dump(hash)
      end
    end

  end
end
