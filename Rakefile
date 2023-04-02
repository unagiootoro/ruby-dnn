require "bundler/gem_tasks"
require "rake/testtask"
require "rake/extensiontask"
require "yard"
require "yard/rake/yardoc_task"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "ext"
  t.libs << "lib"
  t.test_files = FileList["test/*_test.rb", "test/layers_test/*_test.rb"]
end

Rake::ExtensionTask.new "rb_stb_image" do |ext|
  ext.lib_dir = "lib/rb_stb_image"
end

Rake::ExtensionTask.new "exlib" do |ext|
  ext.lib_dir = "lib/exlib"
end

task :build_exlib do
  sh "cd ext/exlib; nvcc -dc -Xcompiler '-fPIC' im2col_gpu.cu"
  sh "cd ext/exlib; nvcc -dlink -Xcompiler '-fPIC' im2col_gpu.o -o link_im2col_gpu.o"
  sh "cd ext/exlib; ar rcs libim2col_gpu.a im2col_gpu.o link_im2col_gpu.o"

  sh "cd ext/exlib; ruby extconf.rb; make"
end

task :clean_exlib do
  targets = [
    "ext/exlib/im2col_gpu.o",
    "ext/exlib/link_im2col_gpu.o",
    "ext/exlib/libim2col_gpu.a",
  ]
  targets.each do |target|
    File.unlink(target) if File.exist?(target)
  end

  sh "cd ext/exlib; make clean; unlink Makefile"
end

task :build_rb_stb_image do
  sh "cd ext/rb_stb_image; ruby extconf.rb; make"
end

task :clean_rb_stb_image do
  sh "cd ext/rb_stb_image; make clean; unlink Makefile"
end

task :default => [:test, :build_rb_stb_image]

YARD::Rake::YardocTask.new do |t|
  t.files = [
    "lib/dnn.rb",
    "lib/dnn/core/*.rb",
    "lib/dnn/core/layers/*.rb",
    "lib/dnn/*.rb",
    "lib/dnn/datasets/*.rb",
  ]
end
