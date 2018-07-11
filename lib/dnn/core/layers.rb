module DNN
  module Layers

    #Super class of all optimizer classes.
    class Layer
      include Numo

      #Initialize layer when model is compiled.
      def init(model)
        @model = model
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
    
      def init(model)
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

      def im2col(img, out_h, out_w, fh, fw, strides)
        bs, fn = img.shape[0..1]
        col = SFloat.zeros(bs, fn, fh, fw, out_h, out_w)
        (0...fh).each do |i|
          i_range = (i...(i + strides[0] * out_h)).step(strides[0]).to_a
          (0...fw).each do |j|
            j_range = (j...(j + strides[1] * out_w)).step(strides[1]).to_a
            col[true, true, i, j, true, true] = img[true, true, i_range, j_range]
          end
        end
        col.transpose(0, 4, 5, 1, 2, 3).reshape(bs * out_h * out_w, fn * fh * fw)
      end
      
      def col2im(col, img_shape, out_h, out_w, fh, fw, strides)
        bs, fn, ih, iw = img_shape
        col = col.reshape(bs, out_h, out_w, fn, fh, fw).transpose(0, 3, 4, 5, 1, 2)
        img = SFloat.zeros(bs, fn, ih, iw)
        (0...fh).each do |i|
          i_range = (i...(i + strides[0] * out_h)).step(strides[0]).to_a
          (0...fw).each do |j|
            j_range = (j...(j + strides[1] * out_w)).step(strides[1]).to_a
            img[true, true, i_range, j_range] += col[true, true, i, j, true, true]
          end
        end
        img
      end

      def padding(img, pad)
        bs, c, ih, iw = img.shape
        ih2 = ih + pad * 2
        iw2 = iw + pad * 2
        img2 = SFloat.zeros(bs, c, ih2, iw2)
        img2[true, true, pad...(ih + pad), pad...(iw + pad)] = img
        img2
      end

      def back_padding(img, pad)
        i_end = img.shape[2] - pad
        j_end = img.shape[3] - pad
        img[true, true, pad...i_end, pad...j_end]
      end
    end
    
    
    class Conv2D < HasParamLayer
      include Initializers
      include Convert
    
      def initialize(num_filters, filter_height, filter_width,
                     weight_initializer: nil,
                     bias_initializer: nil,
                     strides: [1, 1],
                     padding: 0,
                     weight_decay: 0)
        super()
        @num_filters = num_filters
        @filter_height = filter_height
        @filter_width = filter_width
        @weight_initializer = (weight_initializer || RandomNormal.new)
        @bias_initializer = (bias_initializer || Zeros.new)
        @strides = strides
        @padding = padding
        @weight_decay = weight_decay
      end

      def self.load_hash(hash)
        Conv2D.new(hash[:num_filters], hash[:filter_height], hash[:filter_width],
                   weight_initializer: Util.load_hash(hash[:weight_initializer]),
                   bias_initializer: Util.load_hash(hash[:bias_initializer]),
                   strides: hash[:strides],
                   padding: hash[:padding],
                   weight_decay: hash[:weight_decay])
      end
    
      def init(model)
        super
        prev_height, prev_width = prev_layer.shape[1], prev_layer.shape[2]
        @out_height = (prev_height + @padding * 2 - @filter_height) / @strides[0] + 1
        @out_width = (prev_width + @padding * 2 - @filter_width) / @strides[1] + 1
      end
    
      def forward(x)
        x = padding(x, 2) if @padding > 0
        @x_shape = x.shape
        @col = im2col(x, @out_height, @out_width, @filter_height, @filter_width, @strides)
        out = @col.dot(@params[:weight])
        out.reshape(@model.batch_size, @out_height, @out_width, out.shape[3]).transpose(0, 3, 1, 2)
      end
    
      def backward(dout)
        dout = dout.transpose(0, 2, 3, 1)
        dout = dout.reshape(dout.shape[0..2].reduce(:*), dout.shape[3])
        @grads[:weight] = @col.transpose.dot(dout)
        if @weight_decay > 0
          dridge = @weight_decay * @params[:weight]
          @grads[:weight] += dridge
        end
        @grads[:bias] = dout.sum(0)
        dcol = dout.dot(@params[:weight].transpose)
        dx = col2im(dcol, @x_shape, @out_height, @out_width, @filter_height, @filter_width, @strides)
        @padding ? back_padding(dx, @padding) : dx
      end
    
      def shape
        [@num_filters, @out_height, @out_width]
      end

      def to_hash
        {
          name: self.class.name,
          num_filters: @num_filters,
          filter_height: @filter_height,
          filter_width: @filter_width,
          weight_initializer: @weight_initializer.to_hash,
          bias_initializer: @bias_initializer.to_hash,
          strides: @strides,
          padding: @padding,
          weight_decay: @weight_decay,
        }
      end
    
      private
    
      def init_params
        num_prev_filter = prev_layer.shape[0]
        @params[:weight] = SFloat.new(num_prev_filter * @filter_height * @filter_height, @num_filters)
        @params[:bias] = SFloat.new(@num_filters)
        @weight_initializer.init_param(self, :weight)
        @bias_initializer.init_param(self, :bias)
      end
    end
    
    
    class MaxPool2D < Layer
      include Convert

      def initialize(pool_height, pool_width, strides: nil, padding: 0)
        @pool_height = pool_height
        @pool_width = pool_width
        @strides = strides ? strides : [@pool_height, @pool_width]
        @padding = padding
      end
    
      def init(model)
        super
        prev_height, prev_width = prev_layer.shape[1], prev_layer.shape[2]
        @num_channel = prev_layer.shape[0]
        @out_height = (prev_height + @padding * 2 - @pool_height) / @strides[0] + 1
        @out_width = (prev_width + @padding * 2 - @pool_width) / @strides[1] + 1
      end
    
      def forward(x)
        x = padding(x, 2) if @padding > 0
        @x_shape = x.shape
        col = im2col(x, @out_height, @out_width, @pool_height, @pool_width, @strides)
        col = col.reshape(x.shape[0] * @out_height * @out_width * x.shape[1], @pool_height * @pool_width)
        @max_index = col.max_index(1)
        col.max(1).reshape(x.shape[0], @out_height, @out_width, x.shape[1]).transpose(0, 3, 1, 2)
      end
    
      def backward(dout)
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = @pool_height * @pool_width
        dmax = SFloat.zeros(dout.size * pool_size)
        dmax[@max_index] = dout.flatten
        dcol = dmax.reshape(dout.shape[0..2].reduce(:*), dout.shape[3] * pool_size)
        dx = col2im(dcol, @x_shape, @out_height, @out_width, @pool_height, @pool_width, @strides)
        @padding ? back_padding(dx, @padding) : dx
      end
    
      def shape
        [@num_channel, @out_height, @out_width]
      end

      def to_hash
        {
          name: self.class.name,
          pool_height: @pool_height,
          pool_width: @pool_width,
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
        @dropout_ratio = dropout_ratio
        @mask = nil
      end

      def self.load(hash)
        self.new(hash[:dropout_ratio])
      end
    
      def forward(x)
        if @model.training
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
    end
    
    
    class BatchNormalization < HasParamLayer
      def forward(x)
        @mean = x.mean(0)
        @xc = x - @mean
        @var = (@xc**2).mean(0)
        @std = NMath.sqrt(@var + 1e-7)
        @xn = @xc / @std
        @params[:gamma] * @xn + @params[:beta]
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
    
      private
    
      def init_params
        @params[:gamma] = SFloat.ones(*shape)
        @params[:beta] = SFloat.zeros(*shape)
      end
    end
  end
  
end
