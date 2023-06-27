class ProgressBar
  def initialize(max_size, length: 40, io: $stdout)
    @max_size = max_size
    @length = length
    @io = io
    @current_size = 0
  end

  def progress(size)
    if @current_size + size > @max_size
      @current_size = @max_size
    else
      @current_size += size
    end
  end

  def print(prepare: nil, append: nil)
    log = prepare ? "#{prepare}[" : "["
    @length.times do |i|
      if i < @current_size * @length / @max_size
        log << "="
      elsif i == @current_size * @length / @max_size
        log << ">"
      else
        log << "_"
      end
    end
    log << "]  #{@current_size}/#{@max_size} "
    log << append if append
    @io.print log
  end

  def finished?
    @current_size >= @max_size
  end

  def reset
    @current_size = 0
  end
end
