module DNN
  module Layers

    #Super class of all optimizer classes.
    class Layer
      include Numo

      def initialize
        @builded = false
      end

      #Initialize layer when model is compiled.
      def build(model)
        @builded = true
        @model = model
      end

      def builded?
        @builded
      end

      #Forward propagation.
      def forward() end

      #Backward propagation.
      def backward() end
    
      #Get the shape of the layer.
      def shape
        prev_layer.shape
      end

      #Layer to a hash.
      def to_hash
        {name: self.class.name}
      end
    
      #Get the previous layer.
      def prev_layer
        @model.layers[@model.layers.index(self) - 1]
      end
    end
    
    
    class HasParamLayer < Layer
      attr_reader :params #The parameters of the layer.
      attr_reader :grads  #Differential value of parameter of layer.
    
      def initialize
        @params = {}
        @grads = {}
      end
    
      def build(model)
        super
        init_params
      end
    
      #Update the parameters.
      def update
        @model.optimizer.update(self)
      end
    
      private
      
      #Initialize of the parameters.
      def init_params() end
    end
    
    
    class InputLayer < Layer
      attr_reader :shape

      def self.load_hash(hash)
        self.new(hash[:shape])
      end

      def initialize(dim_or_shape)
        @shape = dim_or_shape.is_a?(Array) ? dim_or_shape : [dim_or_shape]
      end

      def forward(x)
        x
      end
    
      def backward(dout)
        dout
      end

      def to_hash
        {name: self.class.name, shape: @shape}
      end
    end
    
    
    class Dense < HasParamLayer
      include Initializers

      attr_reader :num_nodes
      attr_reader :weight_decay
    
      def initialize(num_nodes,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     weight_decay: 0)
        super()
        @num_nodes = num_nodes
        @weight_initializer = (weight_initializer || RandomNormal.new)
        @bias_initializer = (bias_initializer || Zeros.new)
        @weight_decay = weight_decay
      end

      def self.load_hash(hash)
        self.new(hash[:num_nodes],
                 weight_initializer: Util.load_hash(hash[:weight_initializer]),
                 bias_initializer: Util.load_hash(hash[:bias_initializer]),
                 weight_decay: hash[:weight_decay])
      end
    
      def forward(x)
        @x = x
        @x.dot(@params[:weight]) + @params[:bias]
      end
    
      def backward(dout)
        @grads[:weight] = @x.transpose.dot(dout)
        if @weight_decay > 0
          dridge = @weight_decay * @params[:weight]
          @grads[:weight] += dridge
        end
        @grads[:bias] = dout.sum(0)
        dout.dot(@params[:weight].transpose)
      end
    
      def shape
        [@num_nodes]
      end

      def to_hash
        {
          name: self.class.name,
          num_nodes: @num_nodes,
          weight_initializer: @weight_initializer.to_hash,
          bias_initializer: @bias_initializer.to_hash,
          weight_decay: @weight_decay,
        }
      end
    
      private
    
      def init_params
        num_prev_nodes = prev_layer.shape[0]
        @params[:weight] = SFloat.new(num_prev_nodes, @num_nodes)
        @params[:bias] = SFloat.new(@num_nodes)
        @weight_initializer.init_param(self, :weight)
        @bias_initializer.init_param(self, :bias)
      end
    end
    
    
    #private module
    module Convert
      private

      def im2col(img, out_w, out_h, fil_w, fil_h, strides)
        bsize = img.shape[0]
        ch = img.shape[3]
        col = SFloat.zeros(bsize, ch, fil_w, fil_h, out_w, out_h)
        img = img.transpose(0, 3, 1, 2)
        (0...fil_h).each do |i|
          i_range = (i...(i + strides[1] * out_h)).step(strides[1]).to_a
          (0...fil_w).each do |j|
            j_range = (j...(j + strides[0] * out_w)).step(strides[0]).to_a
            col[true, true, j, i, true, true] = img[true, true, j_range, i_range]
          end
        end
        col.transpose(0, 4, 5, 2, 3, 1).reshape(bsize * out_w * out_h, fil_w * fil_h * ch)
      end

      def col2im(col, img_shape, out_w, out_h, fil_w, fil_h, strides)
        bsize, img_w, img_h, ch = img_shape
        col = col.reshape(bsize, out_w, out_h, fil_w, fil_h, ch).transpose(0, 5, 3, 4, 1, 2)
        img = SFloat.zeros(bsize, ch, img_w, img_h)
        (0...fil_h).each do |i|
          i_range = (i...(i + strides[1] * out_h)).step(strides[1]).to_a
          (0...fil_w).each do |j|
            j_range = (j...(j + strides[0] * out_w)).step(strides[0]).to_a
            img[true, true, j_range, i_range] += col[true, true, j, i, true, true]
          end
        end
        img.transpose(0, 2, 3, 1)
      end

      def padding(img, pad)
        bsize, img_w, img_h, ch = img.shape
        img2 = SFloat.zeros(bsize, img_w + pad[0], img_h + pad[1], ch)
        i_begin = pad[1] / 2
        i_end = i_begin + img_h
        j_begin = pad[0] / 2
        j_end = j_begin + img_w
        img2[true, j_begin...j_end, i_begin...i_end, true] = img
        img2
      end

      def back_padding(img, pad)
        i_begin = pad[1] / 2
        i_end = img.shape[2] - (pad[1] / 2.0).round
        j_begin = pad[0] / 2
        j_end = img.shape[1] - (pad[0] / 2.0).round
        img[true, j_begin...j_end, i_begin...i_end, true]
      end

      def out_size(prev_w, prev_h, fil_w, fil_h, strides)
        out_w = (prev_w - fil_w) / strides[1] + 1
        out_h = (prev_h - fil_h) / strides[0] + 1
        [out_w, out_h]
      end
    end
    
    
    class Conv2D < HasParamLayer
      include Initializers
      include Convert
    
      def initialize(num_filters, filter_width, filter_height,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     strides: [1, 1],
                     padding: false,
                     weight_decay: 0)
        super()
        @num_filters = num_filters
        @filter_width = filter_width
        @filter_height = filter_height
        @weight_initializer = (weight_initializer || RandomNormal.new)
        @bias_initializer = (bias_initializer || Zeros.new)
        @strides = strides
        @padding = padding
        @weight_decay = weight_decay
      end

      def self.load_hash(hash)
        Conv2D.new(hash[:num_filters], hash[:filter_width], hash[:filter_height],
                   weight_initializer: Util.load_hash(hash[:weight_initializer]),
                   bias_initializer: Util.load_hash(hash[:bias_initializer]),
                   strides: hash[:strides],
                   padding: hash[:padding],
                   weight_decay: hash[:weight_decay])
      end

      def build(model)
        super
        prev_width, prev_height = prev_layer.shape[0..1]
        @out_width, @out_height = out_size(prev_width, prev_height, @filter_width, @filter_height, @strides)
        if @padding
          @pad = [prev_width - @out_width, prev_height - @out_height]
          @out_width = prev_width
          @out_height = prev_height
        end
      end

      def forward(x)
        x = padding(x, @pad) if @padding
        @x_shape = x.shape
        @col = im2col(x, @out_width, @out_height, @filter_width, @filter_height, @strides)
        out = @col.dot(@params[:weight])
        out.reshape(x.shape[0], @out_width, @out_height, out.shape[3])
      end

      def backward(dout)
        dout = dout.reshape(dout.shape[0..2].reduce(:*), dout.shape[3])
        @grads[:weight] = @col.transpose.dot(dout)
        if @weight_decay > 0
          dridge = @weight_decay * @params[:weight]
          @grads[:weight] += dridge
        end
        @grads[:bias] = dout.sum(0)
        dcol = dout.dot(@params[:weight].transpose)
        dx = col2im(dcol, @x_shape, @out_width, @out_height, @filter_width, @filter_height, @strides)
        @padding ? back_padding(dx, @pad) : dx
      end

      def shape
        [@out_width, @out_height, @num_filters]
      end

      def to_hash
        {
          name: self.class.name,
          num_filters: @num_filters,
          filter_width: @filter_width,
          filter_height: @filter_height,
          weight_initializer: @weight_initializer.to_hash,
          bias_initializer: @bias_initializer.to_hash,
          strides: @strides,
          padding: @padding,
          weight_decay: @weight_decay,
        }
      end
    
      private
    
      def init_params
        num_prev_filter = prev_layer.shape[2]
        @params[:weight] = SFloat.new(num_prev_filter * @filter_width * @filter_height, @num_filters)
        @params[:bias] = SFloat.new(@num_filters)
        @weight_initializer.init_param(self, :weight)
        @bias_initializer.init_param(self, :bias)
      end
    end
    
    
    class MaxPool2D < Layer
      include Convert

      def initialize(pool_width, pool_height, strides: nil, padding: false)
        super()
        @pool_width = pool_width
        @pool_height = pool_height
        @strides = strides ? strides : [@pool_width, @pool_height]
        @padding = padding
      end

      def self.load_hash(hash)
        MaxPool2D.new(hash[:pool_width], hash[:pool_height], strides: hash[:strides], padding: hash[:padding])
      end

      def build(model)
        super
        prev_width, prev_height = prev_layer.shape[0..1]
        @num_channel = prev_layer.shape[2]
        @out_width, @out_height = out_size(prev_width, prev_height, @pool_width, @pool_height, @strides)
        if @padding
          @pad = [prev_width - @out_width, prev_height - @out_height]
          @out_width = prev_width
          @out_height = prev_height
        end
      end

      def forward(x)
        x = padding(x, @pad) if @padding
        @x_shape = x.shape
        col = im2col(x, @out_width, @out_height, @pool_width, @pool_height, @strides)
        col = col.reshape(x.shape[0] * @out_width * @out_height * x.shape[3], @pool_width * @pool_height)
        @max_index = col.max_index(1)
        col.max(1).reshape(x.shape[0], @out_width, @out_height, x.shape[3])#.transpose(0, 3, 1, 2)
      end

      def backward(dout)
        pool_size = @pool_width * @pool_height
        dmax = SFloat.zeros(dout.size * pool_size)
        dmax[@max_index] = dout.flatten
        dcol = dmax.reshape(dout.shape[0..2].reduce(:*), dout.shape[3] * pool_size)
        dx = col2im(dcol, @x_shape, @out_width, @out_height, @pool_width, @pool_height, @strides)
        @padding ? back_padding(dx, @pad) : dx
      end

      def shape
        [@out_width, @out_height, @num_channel]
      end

      def to_hash
        {
          name: self.class.name,
          pool_width: @pool_width,
          pool_height: @pool_height,
          strides: @strides,
          padding: @padding,
        }
      end
    end
    
    
    class Flatten < Layer
      def forward(x)
        @shape = x.shape
        x.reshape(x.shape[0], x.shape[1..-1].reduce(:*))
      end
    
      def backward(dout)
        dout.reshape(*@shape)
      end
    
      def shape
        [prev_layer.shape.reduce(:*)]
      end
    end


    class Reshape < Layer
      attr_reader :shape
      
      def initialize(shape)
        super()
        @shape = shape
        @x_shape = nil
      end

      def self.load_hash(hash)
        self.new(hash[:shape])
      end

      def forward(x)
        @x_shape = x.shape
        x.reshape(*@shape)
      end

      def backward(dout)
        dout.reshape(@x_shape)
      end

      def to_hash
        {name: self.class.name, shape: @shape}
      end
    end


    class OutputLayer < Layer
      private
    
      def ridge
        @model.layers.select { |layer| layer.respond_to?(:weight_decay) }
                     .reduce(0) { |sum, layer| layer.weight_decay * (layer.params[:weight]**2).sum }
      end
    end
    
    
    class Dropout < Layer
      def initialize(dropout_ratio)
        super()
        @dropout_ratio = dropout_ratio
        @mask = nil
      end

      def self.load_hash(hash)
        self.new(hash[:dropout_ratio])
      end

      def self.load(hash)
        self.new(hash[:dropout_ratio])
      end
    
      def forward(x)
        if @model.training?
          @mask = SFloat.ones(*x.shape).rand < @dropout_ratio
          x[@mask] = 0
        else
          x *= (1 - @dropout_ratio)
        end
        x
      end
    
      def backward(dout)
        dout[@mask] = 0 if @model.training
        dout
      end

      def to_hash
        {name: self.class.name, dropout_ratio: @dropout_ratio}
      end
    end
    
    
    class BatchNormalization < HasParamLayer
      def initialize(momentum: 0.9, running_mean: nil, running_var: nil)
        super()
        @momentum = momentum
        @running_mean = running_mean
        @running_var = running_var
      end

      def self.load_hash(hash)
        running_mean = SFloat.cast(hash[:running_mean])
        running_var = SFloat.cast(hash[:running_var])
        self.new(momentum: hash[:momentum], running_mean: running_mean, running_var: running_var)
      end

      def build(model)
        super
        @running_mean ||= SFloat.zeros(*shape)
        @running_var ||= SFloat.zeros(*shape)
      end

      def forward(x)
        if @model.training?
          mean = x.mean(0)
          @xc = x - mean
          var = (@xc**2).mean(0)
          @std = NMath.sqrt(var + 1e-7)
          xn = @xc / @std
          @xn = xn
          @running_mean = @momentum * @running_mean + (1 - @momentum) * mean
          @running_var = @momentum * @running_var + (1 - @momentum) * var
        else
          xc = x - @running_mean
          xn = xc / NMath.sqrt(@running_var + 1e-7)
        end
        @params[:gamma] * xn + @params[:beta]
      end
    
      def backward(dout)
        batch_size = dout.shape[0]
        @grads[:beta] = dout.sum(0)
        @grads[:gamma] = (@xn * dout).sum(0)
        dxn = @params[:gamma] * dout
        dxc = dxn / @std
        dstd = -((dxn * @xc) / (@std**2)).sum(0)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / batch_size) * @xc * dvar
        dmean = dxc.sum(0)
        dxc - dmean / batch_size
      end

      def to_hash
        {
          name: self.class.name,
          momentum: @momentum,
          running_mean: @running_mean.to_a,
          running_var: @running_var.to_a,
        }
      end
    
      private
    
      def init_params
        @params[:gamma] = SFloat.ones(*shape)
        @params[:beta] = SFloat.zeros(*shape)
      end
    end
  end
  
end
