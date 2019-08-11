module DNN
  module Layers

    # This module is used for convolution.
    module Conv2DUtils
      private

      # img[bsize, out_h, out_w, ch] to col[bsize * out_h * out_w, fil_h * fil_w * ch]
      def im2col(img, out_h, out_w, fil_h, fil_w, strides)
        bsize = img.shape[0]
        ch = img.shape[3]
        col = Xumo::SFloat.zeros(bsize, out_h, out_w, fil_h, fil_w, ch)
        (0...fil_h).each do |i|
          i_range = (i...(i + strides[0] * out_h)).step(strides[0]).to_a
          (0...fil_w).each do |j|
            j_range = (j...(j + strides[1] * out_w)).step(strides[1]).to_a
            col[true, true, true, i, j, true] = img[true, i_range, j_range, true]
          end
        end
        col.reshape(bsize * out_h * out_w, fil_h * fil_w * ch)
      end

      # col[bsize * out_h * out_w, fil_h * fil_w * ch] to img[bsize, out_h, out_w, ch]
      def col2im(col, img_shape, out_h, out_w, fil_h, fil_w, strides)
        bsize, img_h, img_w, ch = img_shape
        col = col.reshape(bsize, out_h, out_w, fil_h, fil_w, ch)
        img = Xumo::SFloat.zeros(bsize, img_h, img_w, ch)
        (0...fil_h).each do |i|
          i_range = (i...(i + strides[0] * out_h)).step(strides[0]).to_a
          (0...fil_w).each do |j|
            j_range = (j...(j + strides[1] * out_w)).step(strides[1]).to_a
            img[true, i_range, j_range, true] += col[true, true, true, i, j, true]
          end
        end
        img
      end

      def zero_padding(img, pad)
        bsize, img_h, img_w, ch = img.shape
        img2 = Xumo::SFloat.zeros(bsize, img_h + pad[0], img_w + pad[1], ch)
        i_begin = pad[0] / 2
        i_end = i_begin + img_h
        j_begin = pad[1] / 2
        j_end = j_begin + img_w
        img2[true, i_begin...i_end, j_begin...j_end, true] = img
        img2
      end

      def zero_padding_bwd(img, pad)
        i_begin = pad[0] / 2
        i_end = img.shape[1] - (pad[0] / 2.0).round
        j_begin = pad[1] / 2
        j_end = img.shape[2] - (pad[1] / 2.0).round
        img[true, i_begin...i_end, j_begin...j_end, true]
      end

      def calc_conv2d_out_size(prev_h, prev_w, fil_h, fil_w, pad_h, pad_w, strides)
        out_h = (prev_h + pad_h - fil_h) / strides[0] + 1
        out_w = (prev_w + pad_w - fil_w) / strides[1] + 1
        [out_h, out_w]
      end

      def calc_conv2d_transpose_out_size(prev_h, prev_w, fil_h, fil_w, pad_h, pad_w, strides)
        out_h = (prev_h - 1) * strides[0] + fil_h - pad_h
        out_w = (prev_w - 1) * strides[1] + fil_w - pad_w
        [out_h, out_w]
      end

      def calc_conv2d_padding_size(prev_h, prev_w, fil_h, fil_w, strides)
        out_h = prev_h / strides[0]
        out_w = prev_w / strides[1]
        pad_h = out_h * strides[0] - prev_h + fil_h - strides[0]
        pad_w = out_w * strides[1] - prev_w + fil_w - strides[1]
        [pad_h, pad_w]
      end

      def calc_conv2d_transpose_padding_size(prev_h, prev_w, fil_h, fil_w, strides)
        out_h = prev_h * strides[0]
        out_w = prev_w * strides[1]
        pad_h = (prev_h - 1) * strides[0] + fil_h - out_h
        pad_w = (prev_w - 1) * strides[1] + fil_w - out_w
        [pad_h, pad_w]
      end
    end


    class Conv2D < Connection
      include Conv2DUtils

      attr_reader :num_filters
      attr_reader :filter_size
      attr_reader :strides
      attr_reader :padding

      def self.from_hash(hash)
        self.new(hash[:num_filters], hash[:filter_size],
                 weight_initializer: Utils.hash_to_obj(hash[:weight_initializer]),
                 bias_initializer: Utils.hash_to_obj(hash[:bias_initializer]),
                 weight_regularizer: Utils.hash_to_obj(hash[:weight_regularizer]),
                 bias_regularizer: Utils.hash_to_obj(hash[:bias_regularizer]),
                 use_bias: hash[:use_bias],
                 strides: hash[:strides],
                 padding: hash[:padding])
      end

      # @param [Integer] num_filters Number of filters.
      # @param [Array | Integer] filter_size Filter size. Filter size is of the form [height, width].
      # @param [Array | Integer] strides Stride length. Stride length is of the form [height, width].
      # @param [Array | Boolean] padding Padding size or whether to padding. Padding size is of the form [height, width].
      def initialize(num_filters, filter_size,
                     weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true,
                     strides: 1,
                     padding: false)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              weight_regularizer: weight_regularizer, bias_regularizer: bias_regularizer, use_bias: use_bias)
        @num_filters = num_filters
        @filter_size = filter_size.is_a?(Integer) ? [filter_size, filter_size] : filter_size
        @strides = strides.is_a?(Integer) ? [strides, strides] : strides
        @padding = padding.is_a?(Integer) ? [padding, padding] : padding
      end

      def build(input_shape)
        unless input_shape.length == 3
          raise DNN_ShapeError.new("Input shape is #{input_shape}. But input shape must be 3 dimensional.")
        end
        super
        prev_h, prev_w, num_prev_filter = *input_shape
        @weight.data = Xumo::SFloat.new(@filter_size.reduce(:*) * num_prev_filter, @num_filters)
        @bias.data = Xumo::SFloat.new(@num_filters) if @bias
        init_weight_and_bias
        @pad_size = if @padding == true
          calc_conv2d_padding_size(prev_h, prev_w, *@filter_size, @strides)
        elsif @padding.is_a?(Array)
          @padding
        else
          [0, 0]
        end
        @out_size = calc_conv2d_out_size(prev_h, prev_w, *@filter_size, *@pad_size, @strides)
      end

      def forward(x)
        x = zero_padding(x, @pad_size) if @padding
        @x_shape = x.shape
        @col = im2col(x, *@out_size, *@filter_size, @strides)
        y = @col.dot(@weight.data)
        y += @bias.data if @bias
        y.reshape(x.shape[0], *@out_size, y.shape[3])
      end

      def backward(dy)
        dy = dy.reshape(dy.shape[0..2].reduce(:*), dy.shape[3])
        if @trainable
          @weight.grad += @col.transpose.dot(dy)
          @bias.grad += dy.sum(0) if @bias
        end
        dcol = dy.dot(@weight.data.transpose)
        dx = col2im(dcol, @x_shape, *@out_size, *@filter_size, @strides)
        @padding ? zero_padding_bwd(dx, @pad_size) : dx
      end

      def output_shape
        [*@out_size, @num_filters]
      end

      # @return [Numo::SFloat] Convert weight to filter and return.
      def filters
        num_prev_filter = @input_shape[2]
        @weight.data.reshape(*@filter_size, num_prev_filter, @num_filters)
      end

      # @param [Numo::SFloat] filters Convert weight to filters and set.
      def filters=(filters)
        num_prev_filter = @input_shape[2]
        @weight.data = filters.reshape(@filter_size.reduce(:*) * num_prev_filter, @num_filters)
      end

      def to_hash
        super(num_filters: @num_filters,
              filter_size: @filter_size,
              strides: @strides,
              padding: @padding)
      end
    end


    class Conv2DTranspose < Connection
      include Conv2DUtils

      attr_reader :num_filters
      attr_reader :filter_size
      attr_reader :strides
      attr_reader :padding

      def self.from_hash(hash)
        self.new(hash[:num_filters], hash[:filter_size],
                 weight_initializer: Utils.hash_to_obj(hash[:weight_initializer]),
                 bias_initializer: Utils.hash_to_obj(hash[:bias_initializer]),
                 weight_regularizer: Utils.hash_to_obj(hash[:weight_regularizer]),
                 bias_regularizer: Utils.hash_to_obj(hash[:bias_regularizer]),
                 use_bias: hash[:use_bias],
                 strides: hash[:strides],
                 padding: hash[:padding])
      end

      # @param [Integer] num_filters Number of filters.
      # @param [Array | Integer] filter_size Filter size. Filter size is of the form [height, width].
      # @param [Array | Integer] strides Stride length. Stride length is of the form [height, width].
      # @param [Array] padding Padding size. Padding size is of the form [height, width].
      def initialize(num_filters, filter_size,
                     weight_initializer: Initializers::RandomNormal.new,
                     bias_initializer: Initializers::Zeros.new,
                     weight_regularizer: nil,
                     bias_regularizer: nil,
                     use_bias: true,
                     strides: 1,
                     padding: false)
        super(weight_initializer: weight_initializer, bias_initializer: bias_initializer,
              weight_regularizer: weight_regularizer, bias_regularizer: bias_regularizer, use_bias: use_bias)
        @num_filters = num_filters
        @filter_size = filter_size.is_a?(Integer) ? [filter_size, filter_size] : filter_size
        @strides = strides.is_a?(Integer) ? [strides, strides] : strides
        @padding = padding.is_a?(Integer) ? [padding, padding] : padding
      end

      def build(input_shape)
        unless input_shape.length == 3
          raise DNN_ShapeError.new("Input shape is #{input_shape}. But input shape must be 3 dimensional.")
        end
        super
        prev_h, prev_w, num_prev_filter = *input_shape
        @weight.data = Xumo::SFloat.new(@filter_size.reduce(:*) * @num_filters, num_prev_filter)
        @bias.data = Xumo::SFloat.new(@num_filters) if @bias
        init_weight_and_bias
        @pad_size = if @padding == true
          calc_conv2d_transpose_padding_size(prev_h, prev_w, *@filter_size, @strides)
        elsif @padding.is_a?(Array)
          @padding
        else
          [0, 0]
        end
        @out_size = calc_conv2d_transpose_out_size(prev_h, prev_w, *@filter_size, *@pad_size, @strides)
      end

      def forward(x)
        bsize = x.shape[0]
        x = x.reshape(x.shape[0..2].reduce(:*), x.shape[3])
        @x = x
        col = x.dot(@weight.data.transpose)
        img_shape = [bsize, @out_size[0] + @pad_size[0], @out_size[1] + @pad_size[1], @num_filters]
        y = col2im(col, img_shape, *input_shape[0..1], *@filter_size, @strides)
        y += @bias.data if @bias
        @padding ? zero_padding_bwd(y, @pad_size) : y
      end

      def backward(dy)
        dy = zero_padding(dy, @pad_size) if @padding
        col = im2col(dy, *input_shape[0..1], *@filter_size, @strides)
        if @trainable
          @weight.grad += col.transpose.dot(@x)
          @bias.grad += col.reshape(col.shape[0] * @filter_size.reduce(:*), @num_filters).sum(0) if @bias
        end
        dx = col.dot(@weight.data)
        dx.reshape(dy.shape[0], *input_shape)
      end

      def output_shape
        [*@out_size, @num_filters]
      end

      # @return [Numo::SFloat] Convert weight to filter and return.
      def filters
        num_prev_filter = @input_shape[2]
        @weight.data.reshape(*@filter_size, @num_filters, num_prev_filter)
      end

      # @param [Numo::SFloat] filters Convert weight to filters and set.
      def filters=(filters)
        num_prev_filter = @input_shape[2]
        @weight.data = filters.reshape(@filter_size.reduce(:*) * @num_filters, num_prev_filter)
      end

      def to_hash
        super(num_filters: @num_filters,
              filter_size: @filter_size,
              strides: @strides,
              padding: @padding)
      end
    end


    # Super class of all pooling2D class.
    class Pool2D < Layer
      include Conv2DUtils

      attr_reader :pool_size
      attr_reader :strides
      attr_reader :padding

      def self.from_hash(hash)
        self.new(hash[:pool_size], strides: hash[:strides], padding: hash[:padding])
      end

      # @param [Array | Integer] pool_size Pooling size. Pooling size is of the form [height, width].
      # @param [Array | Integer | NilClass] strides Stride length. Stride length is of the form [height, width].
      #                                             If you set nil, treat pool_size as strides.
      # @param [Array | Boolean] padding Padding size or whether to padding. Padding size is of the form [height, width].
      def initialize(pool_size, strides: nil, padding: false)
        super()
        @pool_size = pool_size.is_a?(Integer) ? [pool_size, pool_size] : pool_size
        @strides = if strides
          strides.is_a?(Integer) ? [strides, strides] : strides
        else
          @pool_size.clone
        end
        @padding = padding.is_a?(Integer) ? [padding, padding] : padding
      end

      def build(input_shape)
        unless input_shape.length == 3
          raise DNN_ShapeError.new("Input shape is #{input_shape}. But input shape must be 3 dimensional.")
        end
        super
        prev_h, prev_w = input_shape[0..1]
        @num_channel = input_shape[2]
        @pad_size = if @padding == true
          calc_conv2d_padding_size(prev_h, prev_w, *@pool_size, @strides)
        elsif @padding.is_a?(Array)
          @padding
        else
          [0, 0]
        end
        @out_size = calc_conv2d_out_size(prev_h, prev_w, *@pool_size, *@pad_size, @strides)
      end

      def output_shape
        [*@out_size, @num_channel]
      end

      def to_hash
        super(pool_size: @pool_size,
              strides: @strides,
              padding: @padding)
      end
    end


    class MaxPool2D < Pool2D
      def forward(x)
        x = zero_padding(x, @pad_size) if @padding
        @x_shape = x.shape
        col = im2col(x, *@out_size, *@pool_size, @strides)
        col = col.reshape(x.shape[0] * @out_size.reduce(:*), @pool_size.reduce(:*), x.shape[3]).transpose(0, 2, 1)
                 .reshape(x.shape[0] * @out_size.reduce(:*) * x.shape[3], @pool_size.reduce(:*))
        @max_index = col.max_index(1)
        col.max(1).reshape(x.shape[0], *@out_size, x.shape[3])
      end

      def backward(dy)
        dmax = Xumo::SFloat.zeros(dy.size * @pool_size.reduce(:*))
        dmax[@max_index] = dy.flatten
        dcol = dmax.reshape(dy.shape[0..2].reduce(:*), @pool_size.reduce(:*) * dy.shape[3])
        dx = col2im(dcol, @x_shape, *@out_size, *@pool_size, @strides)
        @padding ? zero_padding_bwd(dx, @pad_size) : dx
      end
    end


    class AvgPool2D < Pool2D
      def forward(x)
        x = zero_padding(x, @pad_size) if @padding
        @x_shape = x.shape
        col = im2col(x, *@out_size, *@pool_size, @strides)
        col = col.reshape(x.shape[0] * @out_size.reduce(:*), @pool_size.reduce(:*), x.shape[3]).transpose(0, 2, 1)
                 .reshape(x.shape[0] * @out_size.reduce(:*) * x.shape[3], @pool_size.reduce(:*))
        col.mean(1).reshape(x.shape[0], *@out_size, x.shape[3])
      end

      def backward(dy)
        row_length = @pool_size.reduce(:*)
        dy /= row_length
        davg = Xumo::SFloat.zeros(dy.size, row_length)
        row_length.times do |i|
          davg[true, i] = dy.flatten
        end
        dcol = davg.reshape(dy.shape[0..2].reduce(:*), dy.shape[3] * @pool_size.reduce(:*))
        dx = col2im(dcol, @x_shape, *@out_size, *@pool_size, @strides)
        @padding ? zero_padding_bwd(dx, @pad_size) : dx
      end
    end


    class UnPool2D < Layer
      include Conv2DUtils

      attr_reader :unpool_size

      # @param [Array | Integer] unpool_size Unpooling size. unpooling size is of the form [height, width].
      def initialize(unpool_size)
        super()
        @unpool_size = unpool_size.is_a?(Integer) ? [unpool_size, unpool_size] : unpool_size
      end

      def self.from_hash(hash)
        self.new(hash[:unpool_size])
      end

      def build(input_shape)
        unless input_shape.length == 3
          raise DNN_ShapeError.new("Input shape is #{input_shape}. But input shape must be 3 dimensional.")
        end
        super
        prev_h, prev_w = input_shape[0..1]
        unpool_h, unpool_w = @unpool_size
        out_h = prev_h * unpool_h
        out_w = prev_w * unpool_w
        @out_size = [out_h, out_w]
        @num_channel = input_shape[2]
      end

      def forward(x)
        @x_shape = x.shape
        unpool_h, unpool_w = @unpool_size
        x2 = Xumo::SFloat.zeros(x.shape[0], x.shape[1], unpool_h, x.shape[2], unpool_w, @num_channel)
        unpool_h.times do |i|
          unpool_w.times do |j|
            x2[true, true, i, true, j, true] = x
          end
        end
        x2.reshape(x.shape[0], *@out_size, x.shape[3])
      end

      def backward(dy)
        in_size = input_shape[0..1]
        col = im2col(dy, *input_shape[0..1], *@unpool_size, @unpool_size)
        col = col.reshape(dy.shape[0] * in_size.reduce(:*), @unpool_size.reduce(:*), dy.shape[3]).transpose(0, 2, 1)
                 .reshape(dy.shape[0] * in_size.reduce(:*) * dy.shape[3], @unpool_size.reduce(:*))
        col.sum(1).reshape(dy.shape[0], *in_size, dy.shape[3])
      end

      def output_shape
        [*@out_size, @num_channel]
      end

      def to_hash
        super(unpool_size: @unpool_size)
      end
    end

  end
end
