#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void im2col_gpu(float* img, float* col, size_t bsize, size_t img_h, size_t img_w, size_t ch,
                   size_t out_h, size_t out_w, size_t fil_h, size_t fil_w, size_t stride_h, size_t stride_w);

void col2im_gpu(float* img, float* col, size_t bsize, size_t img_h, size_t img_w, size_t ch,
                   size_t out_h, size_t out_w, size_t fil_h, size_t fil_w, size_t stride_h, size_t stride_w);
#ifdef __cplusplus
}
#endif
