#include <stdint.h>
#include <stdlib.h>

__global__ void im2col_kernel(float* img, float* col, size_t bsize, size_t img_h, size_t img_w, size_t ch,
                   size_t out_h, size_t out_w, size_t fil_h, size_t fil_w, size_t stride_h, size_t stride_w) {
  size_t n, i, j, k, l, m;
  size_t ofs1, ofs2;

  // batch loop
  for (n = 0; n < bsize; n++) {

    // stride loop
    for (i = 0; i < out_h * stride_h; i += stride_h) {
      for (j = 0; j < out_w * stride_w; j += stride_w) {

        // filter loop
        for (k = 0; k < fil_h; k++) {
          for (l = 0; l < fil_w; l++) {
            for (m = 0; m < ch; m++) {
              // compute img offset
              ofs1 = n * (img_h * img_w * ch);
              ofs1 += (i + k) * (img_w * ch);
              ofs1 += (j + l) * ch + m;

              // compute col offset
              ofs2 = n * (out_h * out_w * fil_h * fil_w * ch);
              ofs2 += (i / stride_h) * (out_w * fil_h * fil_w * ch);
              ofs2 += (j / stride_w) * (fil_h * fil_w * ch);
              ofs2 += k * (fil_w * ch);
              ofs2 += l * ch + m;

              col[ofs2] = img[ofs1];
            }
          }
        }

      }
    }

  }
}

__global__ void col2im_kernel(float* img, float* col, size_t bsize, size_t img_h, size_t img_w, size_t ch,
                   size_t out_h, size_t out_w, size_t fil_h, size_t fil_w, size_t stride_h, size_t stride_w) {
  size_t n, i, j, k, l, m;
  size_t ofs1, ofs2;

  // batch loop
  for (n = 0; n < bsize; n++) {

    // stride loop
    for (i = 0; i < out_h * stride_h; i += stride_h) {
      for (j = 0; j < out_w * stride_w; j += stride_w) {

        // filter loop
        for (k = 0; k < fil_h; k++) {
          for (l = 0; l < fil_w; l++) {
            for (m = 0; m < ch; m++) {
              // compute img offset
              ofs1 = n * (img_h * img_w * ch);
              ofs1 += (i + k) * (img_w * ch);
              ofs1 += (j + l) * ch + m;

              // compute col offset
              ofs2 = n * (out_h * out_w * fil_h * fil_w * ch);
              ofs2 += (i / stride_h) * (out_w * fil_h * fil_w * ch);
              ofs2 += (j / stride_w) * (fil_h * fil_w * ch);
              ofs2 += k * (fil_w * ch);
              ofs2 += l * ch + m;

              img[ofs1] += col[ofs2];
            }
          }
        }

      }
    }

  }
}

void im2col(float* img, float* col, size_t bsize, size_t img_h, size_t img_w, size_t ch,
                   size_t out_h, size_t out_w, size_t fil_h, size_t fil_w, size_t stride_h, size_t stride_w) {
  im2col_kernel<<<1,1>>>(img, col, bsize, img_h, img_w, ch, out_h, out_w, fil_h, fil_w, stride_h, stride_w);
}

void col2im(float* img, float* col, size_t bsize, size_t img_h, size_t img_w, size_t ch,
                   size_t out_h, size_t out_w, size_t fil_h, size_t fil_w, size_t stride_h, size_t stride_w) {
  col2im_kernel<<<1,1>>>(img, col, bsize, img_h, img_w, ch, out_h, out_w, fil_h, fil_w, stride_h, stride_w);
}
