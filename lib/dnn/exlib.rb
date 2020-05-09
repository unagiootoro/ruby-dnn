require_relative "../../ext/exlib/exlib"

module DNN
  module Exlib
    module_function

    def im2col(img, out_h, out_w, fil_h, fil_w, strides)
      bsize = img.shape[0]
      ch = img.shape[3]
      if DNN.use_cumo?
        col_data = _im2col_gpu(img, out_h, out_w, fil_h, fil_w, *strides)
      else
        col_data = _im2col_cpu(img, out_h, out_w, fil_h, fil_w, *strides)
      end
      col = col_data.reshape(bsize * out_h * out_w, fil_h * fil_w * ch)
      col
    end

    def col2im(col, img_shape, out_h, out_w, fil_h, fil_w, strides)
      pack_img_shape = img_shape.pack("q*")
      if DNN.use_cumo?
        img_data = _col2im_gpu(col, pack_img_shape, out_h, out_w, fil_h, fil_w, *strides)
      else
        img_data = _col2im_cpu(col, pack_img_shape, out_h, out_w, fil_h, fil_w, *strides)
      end
      img = img_data.reshape(*img_shape)
      img
    end
  end
end
