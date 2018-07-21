#include <ruby.h>
#include <stdlib.h>
#include "numo/narray.h"

#define CIFAR10_WIDTH 32
#define CIFAR10_HEIGHT 32
#define CIFAR10_CHANNEL 3

VALUE mnist_load_images(VALUE self, VALUE rb_bin, VALUE rb_num_images, VALUE rb_cols, VALUE rb_rows) {
  char* bin = StringValuePtr(rb_bin);
  int num_images = FIX2INT(rb_num_images);
  int cols = FIX2INT(rb_cols);
  int rows = FIX2INT(rb_rows);
  int i;
  int j;
  char script[64];
  VALUE rb_na;
  narray_data_t* na_data;

  sprintf(script, "Numo::UInt8.zeros(%d, %d, %d)", num_images, cols, rows);
  rb_na = rb_eval_string(&script[0]);
  na_data = RNARRAY_DATA(rb_na);

  for (i = 0; i < num_images; i++) {
    j = i * cols * rows;
    memcpy(&na_data->ptr[j], &bin[j], cols * rows);
  }
  return rb_na;
}

VALUE mnist_load_labels(VALUE self, VALUE rb_bin, VALUE rb_num_labels) {
  char* bin = StringValuePtr(rb_bin);
  int num_labels = FIX2INT(rb_num_labels);
  char script[64];
  VALUE rb_na;
  narray_data_t* na_data;

  sprintf(script, "Numo::UInt8.zeros(%d)", num_labels);
  rb_na = rb_eval_string(&script[0]);
  na_data = RNARRAY_DATA(rb_na);

  memcpy(na_data->ptr, bin, num_labels);
  return rb_na;
}

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

  sprintf(script, "Numo::UInt8.zeros(%d, %d, %d, %d)", num_datas, CIFAR10_CHANNEL, CIFAR10_WIDTH, CIFAR10_HEIGHT);
  rb_na_x = rb_eval_string(&script[0]);
  na_data_x = RNARRAY_DATA(rb_na_x);
  for(i = 0; i < 64; i++) {
    script[i] = 0;
  }
  sprintf(script, "Numo::UInt8.zeros(%d)", num_datas);
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

void Init_dataset_loader() {
  VALUE rb_dnn = rb_define_module("DNN");
  VALUE rb_mnist = rb_define_module_under(rb_dnn, "MNIST");
  VALUE rb_cifar10 = rb_define_module_under(rb_dnn, "CIFAR10");

  rb_define_singleton_method(rb_mnist, "_mnist_load_images", mnist_load_images, 4);
  rb_define_singleton_method(rb_mnist, "_mnist_load_labels", mnist_load_labels, 2); 
  rb_define_singleton_method(rb_cifar10, "_cifar10_load", cifar10_load, 2);
}
