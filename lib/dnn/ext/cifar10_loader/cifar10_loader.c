#include <ruby.h>
#include <stdlib.h>

#define CIFAR10_WIDTH 32
#define CIFAR10_HEIGHT 32
#define CIFAR10_CHANNEL 3

VALUE cifar10_load_binary(VALUE self, VALUE rb_bin, VALUE rb_num_datas) {
  char* bin = StringValuePtr(rb_bin);
  int num_datas = FIX2INT(rb_num_datas);
  VALUE rb_x_bin;
  VALUE rb_y_bin;
  int i;
  int j = 0;
  int k = 0;
  int size = CIFAR10_WIDTH * CIFAR10_HEIGHT * CIFAR10_CHANNEL;
  int x_bin_size = num_datas * size;
  int y_bin_size = num_datas;
  char* x_bin;
  char* y_bin;

  x_bin = (char*)malloc(x_bin_size);
  y_bin = (char*)malloc(y_bin_size);
  for (i = 0; i < num_datas; i++) {
    y_bin[i] = bin[j];
    j++;
    memcpy(&x_bin[k], &bin[j], size);
    j += size;
    k += size;
  }
  rb_x_bin = rb_str_new(x_bin, x_bin_size);
  rb_y_bin = rb_str_new(y_bin, y_bin_size);
  free(x_bin);
  free(y_bin);
  return rb_ary_new3(2, rb_x_bin, rb_y_bin);
}

void Init_cifar10_loader() {
  VALUE rb_dnn = rb_define_module("DNN");
  VALUE rb_cifar10 = rb_define_module_under(rb_dnn, "CIFAR10");

  rb_define_singleton_method(rb_cifar10, "load_binary", cifar10_load_binary, 2);
}
