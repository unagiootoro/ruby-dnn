require "bundler/gem_tasks"
require "rake/testtask"
require "rake/extensiontask"
require "yard"
require "yard/rake/yardoc_task"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "ext"
  t.libs << "lib"
  t.test_files = FileList["test/*_test.rb", "test/functions_test/*_test.rb", "test/layers_test/*_test.rb"]
end

Rake::ExtensionTask.new "rb_stb_image" do |ext|
  ext.lib_dir = "lib/rb_stb_image"
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
