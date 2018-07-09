require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

task :build_dataset_loader do
  sh "cd lib/dnn/ext/dataset_loader; ruby extconf.rb; make"
end

task :build_image_io do
  sh "cd lib/dnn/ext/image_io; ruby extconf.rb; make"
end

task :default => [:test, :build_dataset_loader, :build_image_io]
