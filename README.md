# ruby-dnn
[![Gem Version](https://badge.fury.io/rb/ruby-dnn.svg)](https://badge.fury.io/rb/ruby-dnn)  
[![Build Status](https://travis-ci.org/unagiootoro/ruby-dnn.svg?branch=master)](https://travis-ci.org/unagiootoro/ruby-dnn)

ruby-dnn is a ruby deep learning library. This library supports full connected neural network and convolution neural network.
Currently, you can get 99% accuracy with MNIST and 74% with CIFAR 10.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'ruby-dnn'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install ruby-dnn

## Usage

### MNIST MLP example

```ruby
model = Sequential.new

model << InputLayer.new(784)

model << Dense.new(256)
model << ReLU.new

model << Dense.new(256)
model << ReLU.new

model << Dense.new(10)

model.setup(RMSProp.new, SoftmaxCrossEntropy.new)

model.train(x_train, y_train, 10, batch_size: 100, test: [x_test, y_test])
```

When create a model with 'define by run' style:  

```ruby
class MLP < Model
  def initialize
    super
    @l1 = Dense.new(256)
    @l2 = Dense.new(256)
    @l3 = Dense.new(10)
  end

  def call(x)
    x = InputLayer.new(784).(x)
    x = @l1.(x)
    x = ReLU.(x)
    x = @l2.(x)
    x = ReLU.(x)
    x = @l3.(x)
    x
  end
end

model = MLP.new

model.setup(Adam.new, SoftmaxCrossEntropy.new)

model.train(x_train, y_train, 10, batch_size: 100, test: [x_test, y_test])
```

Please refer to examples for basic usage.  
If you want to know more detailed information, please refer to the source code.

## Implemented
|| Implemented classes |
|:-----------|------------:|
| Connections | Dense, Conv2D, Conv2DTranspose, Embedding, SimpleRNN, LSTM, GRU |
| Layers | Flatten, Reshape, Dropout, BatchNormalization, MaxPool2D, AvgPool2D, UnPool2D |
| Activations | Sigmoid, Tanh, Softsign, Softplus, Swish, ReLU, LeakyReLU, ELU |
| Optimizers | SGD, Nesterov, AdaGrad, RMSProp, AdaDelta, RMSPropGraves, Adam, AdaBound |
| Losses | MeanSquaredError, MeanAbsoluteError, Hinge, HuberLoss, SoftmaxCrossEntropy, SigmoidCrossEntropy |

## TODO
● Write a test.  
● Write a document.  
● Support to GPU.  

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake "spec"` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/unagiootoro/ruby-dnn. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the ruby-dnn project’s codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/unagiootoro/ruby-dnn/blob/master/CODE_OF_CONDUCT.md).
