module DNN
  module Layers

    # This module is used for convolution.
    module Conv2DUtils
      module_function

      # img[bsize, out_h, out_w, ch] to col[bsize * out_h * out_w, fil_h * fil_w * ch]
      def im2col(*args)
        if DNN.use_cumo?
          im2col_gpu(*args)
        else
          im2col_cpu(*args)
        end
      end

      # col[bsize * out_h * out_w, fil_h * fil_w * ch] to img[bsize, out_h, out_w, ch]
      def col2im(*args)
        if DNN.use_cumo?
          col2im_gpu(*args)
        else
          col2im_cpu(*args)
        end
      end

      def im2col_cpu(img, out_h, out_w, fil_h, fil_w, strides)
        bsize = img.shape[0]
        ch = img.shape[3]
        col = img.class.zeros(bsize, out_h, out_w, fil_h, fil_w, ch)
        (0...fil_h).each do |i|
          i_range = (i...(i + strides[0] * out_h)).step(strides[0]).to_a
          (0...fil_w).each do |j|
            j_range = (j...(j + strides[1] * out_w)).step(strides[1]).to_a
            col[true, true, true, i, j, true] = img[true, i_range, j_range, true]
          end
        end
        col.reshape(bsize * out_h * out_w, fil_h * fil_w * ch)
      end

      def im2col_gpu(img, out_h, out_w, fil_h, fil_w, strides)
        img = Utils.cumo2numo(img)
        col = im2col_cpu(img, out_h, out_w, fil_h, fil_w, strides)
        Utils.numo2cumo(col)
      end

      def col2im_cpu(col, img_shape, out_h, out_w, fil_h, fil_w, strides)
        bsize, img_h, img_w, ch = img_shape
        col = col.reshape(bsize, out_h, out_w, fil_h, fil_w, ch)
        img = col.class.zeros(bsize, img_h, img_w, ch)
        (0...fil_h).each do |i|
          i_range = (i...(i + strides[0] * out_h)).step(strides[0]).to_a
          (0...fil_w).each do |j|
            j_range = (j...(j + strides[1] * out_w)).step(strides[1]).to_a
            img[true, i_range, j_range, true] += col[true, true, true, i, j, true]
          end
        end
        img
      end

      def col2im_gpu(col, img_shape, out_h, out_w, fil_h, fil_w, strides)
        col = Utils.cumo2numo(col)
        img = col2im_cpu(col, img_shape, out_h, out_w, fil_h, fil_w, strides)
        Utils.numo2cumo(img)
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
          raise DNNShapeError, "Input shape is #{input_shape}. But input shape must be 3 dimensional."
        end
        prev_h, prev_w, num_prev_filters = *input_shape
        @pad_size = if @padding == true
          calc_conv2d_padding_size(prev_h, prev_w, *@filter_size, @strides)
        elsif @padding.is_a?(Array)
          @padding
        else
          [0, 0]
        end
        @out_size = calc_conv2d_out_size(prev_h, prev_w, *@filter_size, *@pad_size, @strides)
        super
        @weight.data = Xumo::SFloat.new(@filter_size.reduce(:*) * num_prev_filters, @num_filters)
        @bias.data = Xumo::SFloat.new(@num_filters) if @bias
        init_weight_and_bias
      end

      def forward(x)
        batch_size = x.shape[0]
        x = Functions::ZeroPadding2D.new(@pad_size).(x) if @padding
        x = Functions::Im2col.new(@out_size, @filter_size, @strides).(x)
        x = x.dot(@weight)
        x += @bias if @bias
        Functions::FunctionSpace.reshape(x, [batch_size, *@out_size, x.shape[3]])
      end

      # @return [Numo::SFloat] Convert weight to filter and return.
      def filters
        num_prev_filters = @input_shapes[0][2]
        @weight.data.reshape(*@filter_size, num_prev_filters, @num_filters)
      end

      # @param [Numo::SFloat] filters Convert weight to filters and set.
      def filters=(filters)
        num_prev_filters = @input_shapes[0][2]
        @weight.data = filters.reshape(@filter_size.reduce(:*) * num_prev_filters, @num_filters)
      end

      def to_hash
        super(num_filters: @num_filters,
              filter_size: @filter_size,
              strides: @strides,
              padding: @padding)
      end

      def load_hash(hash)
        initialize(hash[:num_filters], hash[:filter_size],
                   weight_initializer: Initializers::Initializer.from_hash(hash[:weight_initializer]),
                   bias_initializer: Initializers::Initializer.from_hash(hash[:bias_initializer]),
                   weight_regularizer: Regularizers::Regularizer.from_hash(hash[:weight_regularizer]),
                   bias_regularizer: Regularizers::Regularizer.from_hash(hash[:bias_regularizer]),
                   use_bias: hash[:use_bias],
                   strides: hash[:strides],
                   padding: hash[:padding])
      end
    end

    class Conv2DTranspose < Connection
      include Conv2DUtils

      attr_reader :num_filters
      attr_reader :filter_size
      attr_reader :strides
      attr_reader :padding

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
          raise DNNShapeError, "Input shape is #{input_shape}. But input shape must be 3 dimensional."
        end
        prev_h, prev_w, num_prev_filters = *input_shape
        @pad_size = if @padding == true
          calc_conv2d_transpose_padding_size(prev_h, prev_w, *@filter_size, @strides)
        elsif @padding.is_a?(Array)
          @padding
        else
          [0, 0]
        end
        @out_size = calc_conv2d_transpose_out_size(prev_h, prev_w, *@filter_size, *@pad_size, @strides)
        super
        @weight.data = Xumo::SFloat.new(@filter_size.reduce(:*) * @num_filters, num_prev_filters)
        @bias.data = Xumo::SFloat.new(@num_filters) if @bias
        init_weight_and_bias
      end

      def forward(x)
        x_shape = x.shape
        bsize = x.shape[0]
        x = x.reshape(x.shape[0..2].reduce(:*), x.shape[3])
        col = x.dot(@weight.transpose)
        img_shape = [bsize, @out_size[0] + @pad_size[0], @out_size[1] + @pad_size[1], @num_filters]
        y = Functions::Col2im.new(img_shape, x_shape[1..2], @filter_size, @strides).(col)
        y += @bias if @bias
        y = Functions::Cropping2D.new(@pad_size).(y) if @padding
        y
      end

      # @return [Numo::SFloat] Convert weight to filter and return.
      def filters
        num_prev_filters = @input_shapes[0][2]
        @weight.data.reshape(*@filter_size, @num_filters, num_prev_filters)
      end

      # @param [Numo::SFloat] filters Convert weight to filters and set.
      def filters=(filters)
        num_prev_filters = @input_shapes[0][2]
        @weight.data = filters.reshape(@filter_size.reduce(:*) * @num_filters, num_prev_filters)
      end

      def to_hash
        super(num_filters: @num_filters,
              filter_size: @filter_size,
              strides: @strides,
              padding: @padding)
      end

      def load_hash(hash)
        initialize(hash[:num_filters], hash[:filter_size],
                   weight_initializer: Initializers::Initializer.from_hash(hash[:weight_initializer]),
                   bias_initializer: Initializers::Initializer.from_hash(hash[:bias_initializer]),
                   weight_regularizer: Regularizers::Regularizer.from_hash(hash[:weight_regularizer]),
                   bias_regularizer: Regularizers::Regularizer.from_hash(hash[:bias_regularizer]),
                   use_bias: hash[:use_bias],
                   strides: hash[:strides],
                   padding: hash[:padding])
      end
    end

    # Super class of all pooling2D class.
    class Pool2D < Layer
      include Conv2DUtils

      attr_reader :pool_size
      attr_reader :strides
      attr_reader :padding

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
          raise DNNShapeError, "Input shape is #{input_shape}. But input shape must be 3 dimensional."
        end
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
        super
      end

      def to_hash
        super(pool_size: @pool_size,
              strides: @strides,
              padding: @padding)
      end

      def load_hash(hash)
        initialize(hash[:pool_size], strides: hash[:strides], padding: hash[:padding])
      end
    end

    class MaxPool2D < Pool2D
      def forward(x)
        batch_size = x.shape[0]
        ch = x.shape[3]
        x = Functions::ZeroPadding2D.new(@pad_size).(x) if @padding
        x = Functions::Im2col.new(@out_size, @pool_size, @strides).(x)
        x = x.reshape(batch_size * @out_size.reduce(:*), @pool_size.reduce(:*), ch)
        x = x.max(axis: 1, keepdims: true)
        x.reshape(batch_size, *@out_size, ch)
      end
    end

    class AvgPool2D < Pool2D
      def forward(x)
        batch_size = x.shape[0]
        ch = x.shape[3]
        x = Functions::ZeroPadding2D.new(@pad_size).(x) if @padding
        x = Functions::Im2col.new(@out_size, @pool_size, @strides).(x)
        x = x.reshape(batch_size * @out_size.reduce(:*), @pool_size.reduce(:*), ch)
        x = x.mean(axis: 1, keepdims: true)
        x.reshape(batch_size, *@out_size, ch)
      end
    end

    class GlobalAvgPool2D < Layer
      def build(input_shape)
        unless input_shape.length == 3
          raise DNNShapeError, "Input shape is #{input_shape}. But input shape must be 3 dimensional."
        end
        super
      end

      def forward(x)
        x = AvgPool2D.new(@input_shapes[0][0..1]).(x)
        x.reshape(x.shape[0], x.shape[1..-1].reduce(:*))
      end
    end

    class UnPool2D < Layer
      include Conv2DUtils

      attr_reader :unpool_size
      attr_reader :strides
      attr_reader :padding

      # @param [Array | Integer] unpool_size Unpooling size. unpooling size is of the form [height, width].
      # @param [Array | Integer | NilClass] strides Stride length. Stride length is of the form [height, width].
      #                                             If you set nil, treat pool_size as strides.
      # @param [Array | Boolean] padding Padding size or whether to padding. Padding size is of the form [height, width].
      def initialize(unpool_size, strides: nil, padding: false)
        super()
        @unpool_size = unpool_size.is_a?(Integer) ? [unpool_size, unpool_size] : unpool_size
        @strides = if strides
                     strides.is_a?(Integer) ? [strides, strides] : strides
                   else
                     @unpool_size.clone
                   end
        @padding = padding.is_a?(Integer) ? [padding, padding] : padding
      end

      def build(input_shape)
        unless input_shape.length == 3
          raise DNNShapeError, "Input shape is #{input_shape}. But input shape must be 3 dimensional."
        end
        prev_h, prev_w = input_shape[0..1]
        unpool_h, unpool_w = @unpool_size
        out_h = prev_h * unpool_h
        out_w = prev_w * unpool_w
        @out_size = [out_h, out_w]
        @num_channel = input_shape[2]
        @pad_size = if @padding == true
          calc_conv2d_padding_size(prev_h, prev_w, *@pool_size, @strides)
        elsif @padding.is_a?(Array)
          @padding
        else
          [0, 0]
        end
        super
      end

      def forward(x)
        fs = Functions::FunctionSpace
        x_shape = x.shape
        bsize = x.shape[0]
        num_filters = x.shape[3]
        x = x.reshape(x_shape[0..2].reduce(:*), 1, num_filters)
        x = fs.broadcast_to(x, [x_shape[0..2].reduce(:*), @unpool_size.reduce(:*), num_filters])
        col = x.reshape(x_shape[0..2].reduce(:*), @unpool_size.reduce(:*) * num_filters)
        img_shape = [bsize, @out_size[0] + @pad_size[0], @out_size[1] + @pad_size[1], num_filters]
        y = Functions::Col2im.new(img_shape, x_shape[1..2], @unpool_size, @unpool_size).(col)
        y = Functions::Cropping2D.new(@pad_size).(y) if @padding
        y
      end

      def to_hash
        super(unpool_size: @unpool_size)
      end

      def load_hash(hash)
        initialize(hash[:unpool_size])
      end
    end

  end
end
