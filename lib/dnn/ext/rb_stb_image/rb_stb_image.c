#include <ruby.h>
#include "numo/narray.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

//STBIDEF stbi_uc *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);
VALUE rb_stbi_load(VALUE self, VALUE rb_filename, VALUE rb_req_comp) {
  char* filename = StringValuePtr(rb_filename);
  int x, y, n;
  int req_comp = FIX2INT(rb_req_comp);
  unsigned char* pixels;
  narray_data_t* na_data;
  char script[64];
  int ch;
  VALUE rb_x, rb_y, rb_n;
  VALUE rb_pixels;

  pixels = stbi_load(filename, &x, &y, &n, req_comp);
  rb_x = INT2FIX(x);
  rb_y = INT2FIX(y);
  rb_n = INT2FIX(n);
  ch = req_comp == 0 ? n : req_comp;
  sprintf(script, "Numo::UInt8.zeros(%d, %d, %d)", y, x, ch);
  rb_pixels = rb_eval_string(&script[0]);
  na_data = RNARRAY_DATA(rb_pixels);
  memcpy(na_data->ptr, pixels, na_data->base.size);
  stbi_image_free(pixels);
  return rb_ary_new3(4, rb_pixels, rb_x, rb_y, rb_n);
}

//STBIWDEF int stbi_write_png(char const *filename, int w, int h, int comp, const void  *data, int stride_in_bytes);
VALUE rb_stbi_write_png(VALUE self, VALUE rb_filename, VALUE rb_w, VALUE rb_h, VALUE rb_comp, VALUE rb_pixels, VALUE rb_stride_in_bytes) {
  char* filename = StringValuePtr(rb_filename);
  int w = FIX2INT(rb_w);
  int h = FIX2INT(rb_h);
  int comp = FIX2INT(rb_comp);
  unsigned char* pixels;
  int stride_in_bytes = FIX2INT(rb_stride_in_bytes);
  narray_data_t* na_data;
  int result;

  na_data = RNARRAY_DATA(rb_pixels);
  pixels = (unsigned char*)malloc(na_data->base.size);
  memcpy(pixels, na_data->ptr, na_data->base.size);
  result = stbi_write_png(filename, w, h, comp, pixels, stride_in_bytes);
  stbi_image_free(pixels);
  return INT2FIX(result);
}

//STBIWDEF int stbi_write_bmp(char const *filename, int w, int h, int comp, const void  *data);
VALUE rb_stbi_write_bmp(VALUE self, VALUE rb_filename, VALUE rb_w, VALUE rb_h, VALUE rb_comp, VALUE rb_pixels) {
  char* filename = StringValuePtr(rb_filename);
  int w = FIX2INT(rb_w);
  int h = FIX2INT(rb_h);
  int comp = FIX2INT(rb_comp);
  unsigned char* pixels;
  narray_data_t* na_data;
  int result;

  na_data = RNARRAY_DATA(rb_pixels);
  pixels = (unsigned char*)malloc(na_data->base.size);
  memcpy(pixels, na_data->ptr, na_data->base.size);
  result = stbi_write_bmp(filename, w, h, comp, pixels);
  stbi_image_free(pixels);
  return INT2FIX(result);
}

//STBIWDEF int stbi_write_jpg(char const *filename, int x, int y, int comp, const void  *data, int quality);
VALUE rb_stbi_write_jpg(VALUE self, VALUE rb_filename, VALUE rb_w, VALUE rb_h, VALUE rb_comp, VALUE rb_pixels, VALUE rb_quality) {
  char* filename = StringValuePtr(rb_filename);
  int w = FIX2INT(rb_w);
  int h = FIX2INT(rb_h);
  int comp = FIX2INT(rb_comp);
  unsigned char* pixels;
  int quality = FIX2INT(rb_quality);
  narray_data_t* na_data;
  int result;

  na_data = RNARRAY_DATA(rb_pixels);
  pixels = (unsigned char*)malloc(na_data->base.size);
  memcpy(pixels, na_data->ptr, na_data->base.size);
  result = stbi_write_jpg(filename, w, h, comp, pixels, quality);
  stbi_image_free(pixels);
  return INT2FIX(result);
}

void Init_rb_stb_image() {
  VALUE rb_dnn = rb_define_module("DNN");
  VALUE rb_stb = rb_define_module_under(rb_dnn, "Stb");

  rb_define_module_function(rb_stb, "stbi_load", rb_stbi_load, 2);
  rb_define_module_function(rb_stb, "stbi_write_png", rb_stbi_write_png, 6);
  rb_define_module_function(rb_stb, "stbi_write_bmp", rb_stbi_write_bmp, 5);
  rb_define_module_function(rb_stb, "stbi_write_jpg", rb_stbi_write_jpg, 6);
}
