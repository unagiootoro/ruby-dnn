#include <ruby.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

//STBIDEF stbi_uc *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);
VALUE rb_stbi_load(VALUE self, VALUE rb_filename, VALUE rb_req_comp) {
  char* filename = StringValuePtr(rb_filename);
  int x, y, n;
  int req_comp = FIX2INT(rb_req_comp);
  unsigned char* data;
  int ch;
  VALUE rb_x, rb_y, rb_n;
  VALUE rb_data;

  data = stbi_load(filename, &x, &y, &n, req_comp);
  rb_x = INT2FIX(x);
  rb_y = INT2FIX(y);
  rb_n = INT2FIX(n);
  ch = req_comp == 0 ? n : req_comp;
  rb_data = rb_str_new((char*)data, y * x * ch);
  stbi_image_free(data);
  return rb_ary_new3(4, rb_data, rb_x, rb_y, rb_n);
}

//STBIWDEF int stbi_write_png(char const *filename, int w, int h, int comp, const void  *data, int stride_in_bytes);
VALUE rb_stbi_write_png(VALUE self, VALUE rb_filename, VALUE rb_w, VALUE rb_h, VALUE rb_comp, VALUE rb_data, VALUE rb_stride_in_bytes) {
  char* filename = StringValuePtr(rb_filename);
  int w = FIX2INT(rb_w);
  int h = FIX2INT(rb_h);
  int comp = FIX2INT(rb_comp);
  unsigned char* data = (unsigned char*)StringValuePtr(rb_data);
  int stride_in_bytes = FIX2INT(rb_stride_in_bytes);
  int result;

  result = stbi_write_png(filename, w, h, comp, data, stride_in_bytes);
  return INT2FIX(result);
}

//STBIWDEF int stbi_write_bmp(char const *filename, int w, int h, int comp, const void  *data);
VALUE rb_stbi_write_bmp(VALUE self, VALUE rb_filename, VALUE rb_w, VALUE rb_h, VALUE rb_comp, VALUE rb_data) {
  char* filename = StringValuePtr(rb_filename);
  int w = FIX2INT(rb_w);
  int h = FIX2INT(rb_h);
  int comp = FIX2INT(rb_comp);
  unsigned char* data = (unsigned char*)StringValuePtr(rb_data);
  int result;

  result = stbi_write_bmp(filename, w, h, comp, data);
  return INT2FIX(result);
}

//STBIWDEF int stbi_write_jpg(char const *filename, int x, int y, int comp, const void  *data, int quality);
VALUE rb_stbi_write_jpg(VALUE self, VALUE rb_filename, VALUE rb_w, VALUE rb_h, VALUE rb_comp, VALUE rb_data, VALUE rb_quality) {
  char* filename = StringValuePtr(rb_filename);
  int w = FIX2INT(rb_w);
  int h = FIX2INT(rb_h);
  int comp = FIX2INT(rb_comp);
  unsigned char* data = (unsigned char*)StringValuePtr(rb_data);
  int quality = FIX2INT(rb_quality);
  int result;

  result = stbi_write_jpg(filename, w, h, comp, data, quality);
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
