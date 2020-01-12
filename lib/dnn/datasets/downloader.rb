require "net/http"

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
      *, @fqdn, @path = *url.match(%r`https?://(.+?)(/.+)`)
    end

    def download(dir_path)
      puts %`download "#{@url}"`
      buf = ""
      Net::HTTP.start(@fqdn) do |http|
        content_length = http.head(@path).content_length
        http.get(@path) do |body_segment|
          buf << body_segment
          log = "\r"
          40.times do |i|
            if i < buf.size * 40 / content_length
              log << "="
            elsif i == buf.size * 40 / content_length
              log << ">"
            else
              log << "_"
            end
          end
          log << "  #{buf.size}/#{content_length}"
          print log
        end
        puts ""
      end
      file_name = @path.match(%r`.*/(.+)`)[1]
      File.binwrite("#{dir_path}/#{file_name}", buf)
    end
  end

end
