require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

task :build_mnist do
  sh "cd lib/dnn/ext/mnist; ruby extconf.rb; make"
end

task :build_cifar10 do
  sh "cd lib/dnn/ext/cifar10; ruby extconf.rb; make"
end

task :build_image_io do
  sh "cd lib/dnn/ext/image_io; ruby extconf.rb; make"
end

task :default => [:test, :build_mnist, :build_cifar10, :build_image_io]
