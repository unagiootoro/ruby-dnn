require "rbconfig"
require "numo/narray"
require "cumo/narray"
require "mkmf"

$CFLAGS="-O3"

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

# find_library("exlig_gpu_kernel", "im2col", ".")

if !have_header("cumo/narray.h")
  print <<EOL
  Header cumo/narray.h was not found. Give pathname as follows:
  % ruby extconf.rb --with-narray-include=narray_h_dir
EOL
  exit(1)
end

$objs = %w[exlib.o im2col.o]

create_makefile("exlib")
