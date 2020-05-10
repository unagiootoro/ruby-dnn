require "rbconfig"
require "numo/narray"
require "cumo/narray"
require "mkmf"

$LOAD_PATH.each do |x|
  if File.exist? File.join(x,"numo/numo/narray.h")
    $INCFLAGS = "-I#{x}/numo " + $INCFLAGS
    break
  end
end

if !have_header("numo/narray.h")
  print <<EOL
  Header numo/narray.h was not found. Give pathname as follows:
  % ruby extconf.rb --with-narray-include=narray_h_dir
EOL
  exit(1)
end

$LOAD_PATH.each do |x|
  if File.exist? File.join(x,"include/cumo/narray.h")
    $INCFLAGS = "-I#{x}/include " + $INCFLAGS
    break
  end
end

# have_library("exlib_gpu_kernel", "im2col_gpu")
$libs += " -lim2col_gpu"

if !have_header("cumo/narray.h")
  print <<EOL
  Header cumo/narray.h was not found. Give pathname as follows:
  % ruby extconf.rb --with-narray-include=narray_h_dir
EOL
  exit(1)
end

$objs = %w[exlib.o im2col_cpu.o]

create_makefile("exlib")
