#define ENABLE_CUMO

#include <ruby.h>
#include <stdint.h>
#include <stdlib.h>
#include "numo/narray.h"
#include "numo/intern.h"
#ifdef ENABLE_CUMO
#include "cumo/narray.h"
#include "cumo/intern.h"
#endif

#include "im2col.h"

static char* get_na_ptr(VALUE na);

static VALUE rb_im2col_cpu(VALUE self, VALUE rb_na_img, VALUE rb_out_h, VALUE rb_out_w,
                           VALUE rb_fil_h, VALUE rb_fil_w, VALUE rb_stride_h, VALUE rb_stride_w);

static VALUE rb_col2im_cpu(VALUE self, VALUE rb_na_col, VALUE rb_img_shape, VALUE rb_out_h, VALUE rb_out_w,
                           VALUE rb_fil_h, VALUE rb_fil_w, VALUE rb_stride_h, VALUE rb_stride_w);

#ifdef ENABLE_CUMO
static char* get_cumo_na_ptr(VALUE na);

static VALUE rb_im2col_gpu(VALUE self, VALUE rb_na_img, VALUE rb_out_h, VALUE rb_out_w,
                           VALUE rb_fil_h, VALUE rb_fil_w, VALUE rb_stride_h, VALUE rb_stride_w);

static VALUE rb_col2im_gpu(VALUE self, VALUE rb_na_col, VALUE rb_img_shape, VALUE rb_out_h, VALUE rb_out_w,
                           VALUE rb_fil_h, VALUE rb_fil_w, VALUE rb_stride_h, VALUE rb_stride_w);
#endif

static char* get_na_ptr(VALUE na) {
  VALUE clone;

  switch (RNARRAY_TYPE(na)) {
  case NARRAY_DATA_T:
    return RNARRAY_DATA_PTR(na);
  case NARRAY_VIEW_T:
    clone = rb_funcall(na, rb_intern("clone"), 0);
    return RNARRAY_DATA_PTR(clone);
  }
  return NULL;
}

#ifdef ENABLE_CUMO
static char* get_cumo_na_ptr(VALUE na) {
  VALUE clone;

  switch (CUMO_RNARRAY_TYPE(na)) {
  case CUMO_NARRAY_DATA_T:
    return CUMO_RNARRAY_DATA_PTR(na);
  case CUMO_NARRAY_VIEW_T:
    clone = rb_funcall(na, rb_intern("clone"), 0);
    return CUMO_RNARRAY_DATA_PTR(clone);
  }
  return NULL;
}
#endif

static VALUE rb_im2col_cpu(VALUE self, VALUE rb_na_img, VALUE rb_out_h, VALUE rb_out_w,
                           VALUE rb_fil_h, VALUE rb_fil_w, VALUE rb_stride_h, VALUE rb_stride_w) {
  float* img = (float*)get_na_ptr(rb_na_img);
  size_t* img_shape = RNARRAY_SHAPE(rb_na_img);
  size_t out_h = FIX2INT(rb_out_h);
  size_t out_w = FIX2INT(rb_out_w);
  size_t fil_h = FIX2INT(rb_fil_h);
  size_t fil_w = FIX2INT(rb_fil_w);
  size_t stride_h = FIX2INT(rb_stride_h);
  size_t stride_w = FIX2INT(rb_stride_w);
  size_t bsize = img_shape[0];
  size_t img_h = img_shape[1];
  size_t img_w = img_shape[2];
  size_t ch = img_shape[3];
  size_t col_size = bsize * out_h * out_w * fil_h * fil_w * ch;
  char script[64];
  VALUE rb_na_col;
  float* col;

  sprintf(&script[0], "Numo::SFloat.zeros(%ld)", col_size);
  rb_na_col = rb_eval_string(script);
  col = (float*)na_get_pointer(rb_na_col);

  im2col_cpu(img, col, bsize, img_h, img_w, ch, out_h, out_w, fil_h, fil_w, stride_h, stride_w);

  return rb_na_col;
}

static VALUE rb_col2im_cpu(VALUE self, VALUE rb_na_col, VALUE rb_img_shape, VALUE rb_out_h, VALUE rb_out_w,
                           VALUE rb_fil_h, VALUE rb_fil_w, VALUE rb_stride_h, VALUE rb_stride_w) {
  float* col = (float*)get_na_ptr(rb_na_col);
  size_t* img_shape = (size_t*)StringValuePtr(rb_img_shape);
  size_t out_h = FIX2INT(rb_out_h);
  size_t out_w = FIX2INT(rb_out_w);
  size_t fil_h = FIX2INT(rb_fil_h);
  size_t fil_w = FIX2INT(rb_fil_w);
  size_t stride_h = FIX2INT(rb_stride_h);
  size_t stride_w = FIX2INT(rb_stride_w);
  size_t bsize = img_shape[0];
  size_t img_h = img_shape[1];
  size_t img_w = img_shape[2];
  size_t ch = img_shape[3];
  size_t img_size = img_shape[0] * img_shape[1] * img_shape[2] * img_shape[3];
  char script[64];
  VALUE rb_na_img;
  float* img;

  sprintf(&script[0], "Numo::SFloat.zeros(%ld)", img_size);
  rb_na_img = rb_eval_string(script);
  img = (float*)na_get_pointer(rb_na_img);

  col2im_cpu(img, col, bsize, img_h, img_w, ch, out_h, out_w, fil_h, fil_w, stride_h, stride_w);

  return rb_na_img;
}

#ifdef ENABLE_CUMO
static VALUE rb_im2col_gpu(VALUE self, VALUE rb_na_img, VALUE rb_out_h, VALUE rb_out_w,
                       VALUE rb_fil_h, VALUE rb_fil_w, VALUE rb_stride_h, VALUE rb_stride_w) {
  float* img = (float*)get_cumo_na_ptr(rb_na_img);
  size_t* img_shape = CUMO_RNARRAY_SHAPE(rb_na_img);
  size_t out_h = FIX2INT(rb_out_h);
  size_t out_w = FIX2INT(rb_out_w);
  size_t fil_h = FIX2INT(rb_fil_h);
  size_t fil_w = FIX2INT(rb_fil_w);
  size_t stride_h = FIX2INT(rb_stride_h);
  size_t stride_w = FIX2INT(rb_stride_w);
  size_t bsize = img_shape[0];
  size_t img_h = img_shape[1];
  size_t img_w = img_shape[2];
  size_t ch = img_shape[3];
  size_t col_size = bsize * out_h * out_w * fil_h * fil_w * ch;
  char script[64];
  VALUE rb_na_col;
  float* col;

  sprintf(&script[0], "Cumo::SFloat.zeros(%ld)", col_size);
  rb_na_col = rb_eval_string(script);
  col = (float*)cumo_na_get_pointer(rb_na_col);

  im2col_cpu(img, col, bsize, img_h, img_w, ch, out_h, out_w, fil_h, fil_w, stride_h, stride_w);

  return rb_na_col;
}

static VALUE rb_col2im_gpu(VALUE self, VALUE rb_na_col, VALUE rb_img_shape, VALUE rb_out_h, VALUE rb_out_w,
                       VALUE rb_fil_h, VALUE rb_fil_w, VALUE rb_stride_h, VALUE rb_stride_w) {
  float* col = (float*)get_cumo_na_ptr(rb_na_col);
  size_t* img_shape = (size_t*)StringValuePtr(rb_img_shape);
  size_t out_h = FIX2INT(rb_out_h);
  size_t out_w = FIX2INT(rb_out_w);
  size_t fil_h = FIX2INT(rb_fil_h);
  size_t fil_w = FIX2INT(rb_fil_w);
  size_t stride_h = FIX2INT(rb_stride_h);
  size_t stride_w = FIX2INT(rb_stride_w);
  size_t bsize = img_shape[0];
  size_t img_h = img_shape[1];
  size_t img_w = img_shape[2];
  size_t ch = img_shape[3];
  size_t img_size = img_shape[0] * img_shape[1] * img_shape[2] * img_shape[3];
  char script[64];
  VALUE rb_na_img;
  float* img;

  sprintf(&script[0], "Cumo::SFloat.zeros(%ld)", img_size);
  rb_na_img = rb_eval_string(script);
  img = (float*)cumo_na_get_pointer(rb_na_img);

  col2im_cpu(img, col, bsize, img_h, img_w, ch, out_h, out_w, fil_h, fil_w, stride_h, stride_w);

  return rb_na_img;
}
#endif

void Init_exlib() {
  VALUE rb_dnn = rb_define_module("DNN");
  VALUE rb_exlib = rb_define_module_under(rb_dnn, "Exlib");

  rb_define_module_function(rb_exlib, "_im2col_cpu", rb_im2col_cpu, 7);
  rb_define_module_function(rb_exlib, "_col2im_cpu", rb_col2im_cpu, 8);
#ifdef ENABLE_CUMO
  rb_define_module_function(rb_exlib, "_im2col_gpu", rb_im2col_gpu, 7);
  rb_define_module_function(rb_exlib, "_col2im_gpu", rb_col2im_gpu, 8);
#endif
}
