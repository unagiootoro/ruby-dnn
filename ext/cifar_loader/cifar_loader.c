#include <ruby.h>
#include <stdint.h>
#include <stdlib.h>

#define CIFAR_WIDTH 32
#define CIFAR_HEIGHT 32
#define CIFAR_CHANNEL 3

static VALUE cifar_load_binary(VALUE self, VALUE rb_bin, VALUE rb_num_datas) {
  uint8_t* bin = (uint8_t*)StringValuePtr(rb_bin);
  int32_t num_datas = FIX2INT(rb_num_datas);
  VALUE rb_x_bin;
  VALUE rb_y_bin;
  int32_t i;
  int32_t j = 0;
  int32_t k = 0;
  int32_t size = CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNEL;
  int32_t x_bin_size = num_datas * size;
  int32_t y_bin_size = num_datas;
  uint8_t* x_bin;
  uint8_t* y_bin;

  x_bin = (uint8_t*)malloc(x_bin_size);
  y_bin = (uint8_t*)malloc(y_bin_size);
  for (i = 0; i < num_datas; i++) {
    y_bin[i] = bin[j];
    j++;
    memcpy(&x_bin[k], &bin[j], size);
    j += size;
    k += size;
  }
  rb_x_bin = rb_str_new((char*)x_bin, x_bin_size);
  rb_y_bin = rb_str_new((char*)y_bin, y_bin_size);
  free(x_bin);
  free(y_bin);
  return rb_ary_new3(2, rb_x_bin, rb_y_bin);
}

void Init_cifar_loader() {
  VALUE rb_dnn = rb_define_module("DNN");
  VALUE rb_cifar = rb_define_module_under(rb_dnn, "CIFAR");

  rb_define_singleton_method(rb_cifar, "load_binary", cifar_load_binary, 2);
}
