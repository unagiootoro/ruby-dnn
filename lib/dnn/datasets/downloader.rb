require "net/https"

module DNN
  DOWNLOADS_PATH = ENV["RUBY_DNN_DOWNLOADS_PATH"] || __dir__

  class DNN_DownloadError < DNNError; end

  class Downloader
    def self.download(url, dir_path = nil)
      unless dir_path
        Dir.mkdir("#{DOWNLOADS_PATH}/downloads") unless Dir.exist?("#{DOWNLOADS_PATH}/downloads")
        dir_path = "#{DOWNLOADS_PATH}/downloads"
      end
      Downloader.new(url).download(dir_path)
    rescue => e
      raise DNN_DownloadError.new(e.message)
    end

    def initialize(url)
      @url = url
      *, @protocol, @fqdn, @path = *url.match(%r`(https?)://(.+?)(/.+)`)
    end

    def download(dir_path)
      puts %`download "#{@url}"`
      buf = ""
      if @protocol == "http"
        port = 80
      elsif @protocol == "https"
        port = 443
      else
        raise "Protocol(#{@protocol}) is not supported."
      end
      http = Net::HTTP.new(@fqdn, port)
      http.use_ssl = true if @protocol == "https"
      http.start do |http|
        content_length = http.head(@path).content_length
        progress_bar = ProgressBar.new(content_length)
        http.get(@path) do |body_segment|
          buf << body_segment
          progress_bar.progress(body_segment.size)
          progress_bar.print
        end
        puts ""
      end
      file_name = @path.match(%r`.*/(.+)`)[1]
      File.binwrite("#{dir_path}/#{file_name}", buf)
    end
  end

end
