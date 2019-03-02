require "net/http"

module DNN

  class DNN_DownloadError < DNN_Error; end

  class Downloader
    def self.download(url, dir_path = __dir__)
      Downloader.new(url).download(dir_path)
    rescue => ex
      raise DNN_DownloadError.new(ex.message)
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
      file_name = @path.match(%r`.+/(.+)`)[1]
      File.binwrite("#{dir_path}/#{file_name}", buf)
    end
  end

end
