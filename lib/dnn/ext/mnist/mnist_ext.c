#include <ruby.h>
#include <stdlib.h>
#include "numo/narray.h"

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

void Init_mnist_ext() {
  VALUE rb_dnn;
  VALUE rb_mnist;
  rb_dnn = rb_define_module("DNN");
  rb_mnist = rb_define_module_under(rb_dnn, "MNIST");
  rb_define_singleton_method(rb_mnist, "_mnist_load_images", mnist_load_images, 4);
  rb_define_singleton_method(rb_mnist, "_mnist_load_labels", mnist_load_labels, 2);
}
