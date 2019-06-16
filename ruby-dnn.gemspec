
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "dnn/version"

Gem::Specification.new do |spec|
  spec.name          = "ruby-dnn"
  spec.version       = DNN::VERSION
  spec.authors       = ["unagiootoro"]
  spec.email         = ["ootoro838861@outlook.jp"]

  spec.summary       = %q{ruby deep learning library.}
  spec.description   = %q{ruby-dnn is a ruby deep learning library.}
  spec.homepage      = "https://github.com/unagiootoro/ruby-dnn.git"
  spec.license       = "MIT"
  spec.extensions    = ["ext/cifar_loader/extconf.rb", "ext/rb_stb_image/extconf.rb"]

  spec.add_dependency "numo-narray"
  spec.add_dependency "archive-tar-minitar"

  # Prevent pushing this gem to RubyGems.org. To allow pushes either set the 'allowed_push_host'
  # to allow pushing to a single host or delete this section to allow pushing to any host.7
=begin
  if spec.respond_to?(:metadata)
    spec.metadata["allowed_push_host"] = "TODO: Set to 'http://mygemserver.com'"
  else
    raise "RubyGems 2.0 or newer is required to protect against " \
      "public gem pushes."
  end
=end

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files         = Dir.chdir(File.expand_path('..', __FILE__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "minitest", "~> 5.0"
end
