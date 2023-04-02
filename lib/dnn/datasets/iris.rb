require "csv"
require_relative "downloader"

module DNN
  class DNN_Iris_LoadError < DNNError; end

  module Iris
    URL_CSV = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Iris-setosa
    SETOSA = 0
    # Iris-versicolor
    VERSICOLOR = 1
    # Iris-virginica
    VIRGINICA = 2

    def self.downloads
      return if File.exist?(url_to_file_name(URL_CSV))
      Downloader.download(URL_CSV)
    end

    def self.load(shuffle = false, shuffle_seed = rand(1 << 31))
      downloads
      csv_array = CSV.read(url_to_file_name(URL_CSV)).reject(&:empty?)
      x = Numo::SFloat.zeros(csv_array.length, 4)
      y = Numo::SFloat.zeros(csv_array.length)
      csv_array.each.with_index do |(sepal_length, sepal_width, petal_length, petal_width, classes), i|
        x[i, 0] = sepal_length.to_f
        x[i, 1] = sepal_width.to_f
        x[i, 2] = petal_length.to_f
        x[i, 3] = petal_width.to_f
        y[i] = case classes
               when "Iris-setosa"
                 SETOSA
               when "Iris-versicolor"
                 VERSICOLOR
               when "Iris-virginica"
                 VIRGINICA
               else
                 raise DNN_Iris_LoadError, "Unknown class name '#{classes}' for iris"
        end
      end
      if shuffle
        if RUBY_VERSION.split(".")[0].to_i >= 3
          orig_seed = Random.seed
        else
          orig_seed = Random::DEFAULT.seed
        end
        srand(shuffle_seed)
        indexs = (0...csv_array.length).to_a.shuffle
        x[indexs, true] = x
        y[indexs] = y
        srand(orig_seed)
      end
      [x, y]
    end

    private_class_method def self.url_to_file_name(url)
      DOWNLOADS_PATH + "/downloads/" + url.match(%r`.+/(.+)$`)[1]
    end
  end
end
