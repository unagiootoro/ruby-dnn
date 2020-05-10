#pragma once

extern void im2col_cpu(float* img, float* col, size_t bsize, size_t img_h, size_t img_w, size_t ch,
                       size_t out_h, size_t out_w, size_t fil_h, size_t fil_w, size_t stride_h, size_t stride_w);

extern void col2im_cpu(float* img, float* col, size_t bsize, size_t img_h, size_t img_w, size_t ch,
                       size_t out_h, size_t out_w, size_t fil_h, size_t fil_w, size_t stride_h, size_t stride_w);
