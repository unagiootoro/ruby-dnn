#include <ruby.h>
#include "numo/narray.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

VALUE image_io_read(VALUE self, VALUE rb_file_name) {
  char* file_name = StringValuePtr(rb_file_name);
  int width;
  int height;
  int bpp;
  unsigned char* pixels;
  char script[64];
  VALUE rb_na;
  narray_data_t* na_data;
  pixels = stbi_load(file_name, &width, &height, &bpp, 3);
  sprintf(script, "Xumo::UInt8.zeros(%d, %d, 3)", width, height);
  rb_na = rb_eval_string((char*)script);
  na_data = RNARRAY_DATA(rb_na);
  memcpy(na_data->ptr, pixels, na_data->base.size);
  stbi_image_free(pixels);
  return rb_na;
}

VALUE image_io_write_png(VALUE self, VALUE rb_file_name, VALUE rb_na) {
  char* file_name = StringValuePtr(rb_file_name);
  int width;
  int height;
  int bpp = 3;
  unsigned char* pixels;
  narray_data_t* na_data;
  na_data = RNARRAY_DATA(rb_na);
  pixels = (unsigned char*)malloc(na_data->base.size);
  memcpy(pixels, na_data->ptr, na_data->base.size);
  width = na_data->base.shape[0];
  height = na_data->base.shape[1];
  stbi_write_png(file_name, width, height, bpp, pixels, width * bpp);
  stbi_image_free(pixels);
  return Qnil;
}

VALUE image_io_write_bmp(VALUE self, VALUE rb_file_name, VALUE rb_na) {
  char* file_name = StringValuePtr(rb_file_name);
  int width;
  int height;
  int bpp = 3;
  unsigned char* pixels;
  narray_data_t* na_data;
  na_data = RNARRAY_DATA(rb_na);
  pixels = (unsigned char*)malloc(na_data->base.size);
  memcpy(pixels, na_data->ptr, na_data->base.size);
  width = na_data->base.shape[0];
  height = na_data->base.shape[1];
  stbi_write_bmp(file_name, width, height, bpp, pixels);
  stbi_image_free(pixels);
  return Qnil;
}

VALUE image_io_write_jpg(VALUE self, VALUE rb_file_name, VALUE rb_na, VALUE rb_quality) {
  char* file_name = StringValuePtr(rb_file_name);
  int width;
  int height;
  int bpp = 3;
  int quality = FIX2INT(rb_quality);
  unsigned char* pixels;
  narray_data_t* na_data;
  na_data = RNARRAY_DATA(rb_na);
  pixels = (unsigned char*)malloc(na_data->base.size);
  memcpy(pixels, na_data->ptr, na_data->base.size);
  width = na_data->base.shape[0];
  height = na_data->base.shape[1];
  stbi_write_jpg(file_name, width, height, bpp, pixels, quality);
  stbi_image_free(pixels);
  return Qnil;
}

void Init_image_io_ext() {
  VALUE rb_dnn;
  VALUE rb_image_io;
  rb_dnn = rb_define_module("DNN");
  rb_image_io = rb_define_module_under(rb_dnn, "ImageIO");
  rb_define_singleton_method(rb_image_io, "_read", image_io_read, 1);
  rb_define_singleton_method(rb_image_io, "_write_png", image_io_write_bmp, 2);
  rb_define_singleton_method(rb_image_io, "_write_bmp", image_io_write_png, 2);
  rb_define_singleton_method(rb_image_io, "_write_jpg", image_io_write_jpg, 3);
}
