module DNN
  module Functions

    module Conv2DFunctionUtils
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
    end

    class Conv2D < Function
      include Conv2DFunctionUtils

      def initialize(num_filters, out_size, filter_size, pad_size,
                     strides: 1,
                     padding: false)
        @num_filters = num_filters
        @out_size = out_size
        @filter_size = filter_size
        @pad_size = pad_size
        @strides = strides
        @padding = padding
        @trainable = true
      end

      def forward(x, weight, bias = nil)
        @weight = weight
        @bias = bias
        x = zero_padding(x, @pad_size) if @padding
        @x_shape = x.shape
        @col = im2col(x, *@out_size, *@filter_size, @strides)
        y = @col.dot(weight)
        y += bias if bias
        y.reshape(x.shape[0], *@out_size, y.shape[3])
      end

      def backward(dy)
        dy = dy.reshape(dy.shape[0..2].reduce(:*), dy.shape[3])
        if @trainable
          dweight = @col.transpose.dot(dy)
          dbias = dy.sum(0) if @bias
        end
        dcol = dy.dot(@weight.transpose)
        dx = col2im(dcol, @x_shape, *@out_size, *@filter_size, @strides)
        dx = @padding ? zero_padding_bwd(dx, @pad_size) : dx
        if @bias
          [dx, dweight, dbias]
        else
          [dx, dweight]
        end
      end
    end

    class Conv2DTranspose < Function
      include Conv2DFunctionUtils

      def initialize(num_filters, out_size, filter_size, pad_size,
                     strides: 1,
                     padding: false)
        @num_filters = num_filters
        @out_size = out_size
        @filter_size = filter_size
        @pad_size = pad_size
        @strides = strides
        @padding = padding
        @trainable = true
      end

      def forward(x, weight, bias = nil)
        @x_shape = x.shape
        @weight = weight
        @bias = bias
        bsize = x.shape[0]
        x = x.reshape(x.shape[0..2].reduce(:*), x.shape[3])
        @x = x
        col = x.dot(@weight.transpose)
        img_shape = [bsize, @out_size[0] + @pad_size[0], @out_size[1] + @pad_size[1], @num_filters]
        y = col2im(col, img_shape, *@x_shape[1..2], *@filter_size, @strides)
        y += @bias if @bias
        @padding ? zero_padding_bwd(y, @pad_size) : y
      end

      def backward(dy)
        dy = zero_padding(dy, @pad_size) if @padding
        col = im2col(dy, *@x_shape[1..2], *@filter_size, @strides)
        if @trainable
          dweight = col.transpose.dot(@x)
          dbias = col.reshape(col.shape[0] * @filter_size.reduce(:*), @num_filters).sum(0) if @bias
        end
        dx = col.dot(@weight)
        dx = dx.reshape(dy.shape[0], *@x_shape[1..-1])
        if @bias
          [dx, dweight, dbias]
        else
          [dx, dweight]
        end
      end
    end

    class Pool2D < Function
      include Conv2DFunctionUtils

      def initialize(pool_size, out_size, strides: nil, padding: false)
        super()
        @pool_size = pool_size.is_a?(Integer) ? [pool_size, pool_size] : pool_size
        @out_size = out_size = @out_size = out_size
        @strides = if strides
                     strides.is_a?(Integer) ? [strides, strides] : strides
                   else
                     @pool_size.clone
                   end
        @padding = padding.is_a?(Integer) ? [padding, padding] : padding
      end
    end

    class MaxPool2D < Pool2D
      def forward(x)
        x = zero_padding(x, @pad_size) if @padding
        @x_shape = x.shape
        col = im2col(x, *@out_size, *@pool_size, @strides)
        col = col.reshape(x.shape[0] * @out_size.reduce(:*), @pool_size.reduce(:*), x.shape[3])
        @max_index = col.max_index(1)
        col.max(1).reshape(x.shape[0], *@out_size, x.shape[3])
      end

      def backward(dy)
        dmax = Xumo::SFloat.zeros(dy.size * @pool_size.reduce(:*))
        dmax[@max_index.flatten] = dy.flatten
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
        col = col.reshape(x.shape[0] * @out_size.reduce(:*), @pool_size.reduce(:*), x.shape[3])
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

    class UnPool2D < Function
      def initialize(unpool_size)
        super()
        @unpool_size = unpool_size.is_a?(Integer) ? [unpool_size, unpool_size] : unpool_size
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
        in_size = @input_shape[0..1]
        col = im2col(dy, *in_size, *@unpool_size, @unpool_size)
        col = col.reshape(dy.shape[0] * in_size.reduce(:*), @unpool_size.reduce(:*), dy.shape[3])
        col.sum(1).reshape(dy.shape[0], *in_size, dy.shape[3])
      end
    end

    class ZeroPadding2D < Function
      include Conv2DFunctionUtils

      def initialize(pad_size)
        @pad_size = pad_size
      end

      def forward(x)
        zero_padding(x, @pad_size)
      end

      def backward(dy)
        zero_padding_bwd(dy, @pad_size)
      end
    end

    class Cropping2D < Function
      include Conv2DFunctionUtils

      def initialize(pad_size)
        @pad_size = pad_size
      end

      def forward(x)
        zero_padding_bwd(x, @pad_size)
      end

      def backward(dy)
        zero_padding(dy, @pad_size)
      end
    end

    class Im2col < Function
      include Conv2DFunctionUtils

      def initialize(out_size, filter_size, strides)
        @out_size = out_size
        @filter_size = filter_size
        @strides = strides
      end

      def forward(x)
        @x_shape = x.shape
        im2col(x, *@out_size, *@filter_size, @strides)
      end

      def backward(dy)
        col2im(dy, @x_shape, *@out_size, *@filter_size, @strides)
      end
    end

    class Col2im < Function
      include Conv2DFunctionUtils

      def initialize(img_shape, in_size, filter_size, strides)
        @img_shape = img_shape
        @in_size = in_size
        @filter_size = filter_size
        @strides = strides
      end

      def forward(x)
        col2im(x, @img_shape, *@in_size, *@filter_size, @strides)
      end

      def backward(dy)
        im2col(dy, *@in_size, *@filter_size, @strides)
      end
    end

  end
end
