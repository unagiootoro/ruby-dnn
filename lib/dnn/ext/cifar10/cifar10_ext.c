#include <ruby.h>
#include <stdlib.h>
#include "numo/narray.h"

#define CIFAR10_WIDTH 32
#define CIFAR10_HEIGHT 32
#define CIFAR10_CHANNEL 3
#define CIFAR10_CLASSES 10

VALUE cifar10_load(VALUE self, VALUE rb_bin, VALUE rb_num_datas) {
  char* bin = StringValuePtr(rb_bin);
  int num_datas = FIX2INT(rb_num_datas);
  char script[64];
  VALUE rb_na_x;
  VALUE rb_na_y;
  VALUE rb_xy;
  narray_data_t* na_data_x;
  narray_data_t* na_data_y;
  int i;
  int j = 0;
  int k = 0;
  int size = CIFAR10_WIDTH * CIFAR10_HEIGHT * CIFAR10_CHANNEL;

  sprintf(script, "Numo::UInt8.zeros(%d, %d, %d, %d)", num_datas, CIFAR10_WIDTH, CIFAR10_HEIGHT, CIFAR10_CHANNEL);
  rb_na_x = rb_eval_string(&script[0]);
  na_data_x = RNARRAY_DATA(rb_na_x);
  for(i = 0; i < 64; i++) {
    script[i] = 0;
  }
  sprintf(script, "Numo::UInt8.zeros(%d, %d)", num_datas, CIFAR10_CLASSES);
  rb_na_y = rb_eval_string(&script[0]);
  na_data_y = RNARRAY_DATA(rb_na_y);

  for (i = 0; i < num_datas; i++) {
    na_data_y->ptr[i] = bin[j];
    j++;
    memcpy(&na_data_x->ptr[k], &bin[j], size);
    j += size;
    k += size;
  }

  rb_xy = rb_ary_new3(2, rb_na_x, rb_na_y);
  return rb_xy;
}

void Init_cifar10_ext() {
  VALUE rb_dnn;
  VALUE rb_cifar10;
  rb_dnn = rb_define_module("DNN");
  rb_cifar10 = rb_define_module_under(rb_dnn, "CIFAR10");
  rb_define_singleton_method(rb_cifar10, "_cifar10_load", cifar10_load, 2);
}
