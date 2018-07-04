## ruby-dnnのAPIリファレンスです。

# module DNN
ruby-dnnの名前空間をなすモジュールです。


# class Model
 ニューラルネットワークのモデルを作成するクラスです。

## 【Singleton methods】

## def self.load(file_name)
marshalファイルを読み込み、モデルを作成します。
### arguments
* String file_name  
  読み込むmarshalファイル名。
### return
なし。

## 【Instance methods】

## def initialize
コンストラクタ。
### arguments
なし。

## def save(file_name)
モデルをmarshalファイルに保存します。
### arguments
* String file_name  
書き込むファイル名。
### return
なし。

## def <<(layer)
モデルにレイヤーを追加します。
### arguments
* Layer layer  
追加するレイヤー。
### return
Model  
自身のモデルのインスタンス。

## def compile(optimizer)
モデルをコンパイルします。
### arguments
* Optimizer optimizer
モデルが学習に使用するオプティマイザー。
### return
なし。

## def train(x, y, epochs, batch_size: 1, batch_proc: nil, verbose: true, &epoch_proc)
コンパイルしたモデルを用いて学習を行います。
### arguments
* SFloat x  
トレーニング用入力データ。
* SFloat y  
トレーニング用出力データ。
* epochs  
学習回数。
* Integer batch_size: 1  
学習に使用するミニバッチの数。
* Proc batch_proc: nil  
一度のバッチ学習が行われる前に呼び出されるprocを登録します。
* bool verbose: true
trueを設定すると、学習ログを出力します。
### block
epoch_proc  
1エポックの学習が終了するたびに呼び出されます。
### return
なし。

## def train_on_batch
入力されたバッチデータをもとに、一度だけ学習を行います。
### arguments
* SFloat x
トレーニング用入力バッチデータ。
* SFloat y
トレーニング用出力バッチデータ。
* Integer batch_size
学習に使用するミニバッチの数。
### block
一度のバッチ学習が行われる前に呼び出されます。
### return
Integer  
損失関数の値を返します。

## def test(x, y, batch_size = nil, &batch_proc)
学習結果をもとにテストを行います。
### arguments
* SFloat x  
テスト用入力データ。
* SFloat y  
テスト用出力データ。
* batch_size  
ミニバッチの数。学習を行っていないモデルのテストを行いたい場合等に使用します。
### block
一度のバッチ学習が行われる前に呼び出されます。
### return
Float  
テスト結果の認識率を返します。

## def accurate(x, y, batch_size = nil, &batch_proc)
学習結果をもとに認識を返します。
### arguments
* SFloat x  
テスト用入力データ。
* SFloat y  
テスト用出力データ。
* batch_size  
ミニバッチの数。学習を行っていないモデルのテストを行いたい場合等に使用します。
### block
一度のバッチ学習が行われる前に呼び出されます。
### return
Float
テスト結果の認識率を返します。

## def predict(x)
モデルを使用して、結果の推論を行います。
### arguments
* SFloat x  
推論用入力データ。
### return
SFloat
推論結果を返します。


# module Layers
レイヤーの名前空間をなすモジュールです。


# class Layer
全てのレイヤーのスーパークラスです。

## 【Instance methods】

## def init(model)
モデルのコンパイル時に、レイヤーを初期化するために使用されます。
### arguments
* Model model  
レイヤーを持つモデルを登録します。
### return
なし。

## abstruct def forward(x)  
順方向伝搬を行うメソッドです。Layerクラスを継承するクラスは、このメソッドを実装する必要があります。
### arguments
* SFloat x  
入力データ。
### return
SFloat  
出力データ。

## abstruct def backward(dout)
逆方向伝搬を行うメソッドです。Layerクラスを継承するクラスは、このメソッドを実装する必要があります。
### arguments
* SFloat dout  
逆方向から伝搬してきた微分値。
### return
SFloat  
逆方向に伝搬する微分値。

## def shape
レイヤーの形状を取得するメソッドです。
### arguments
なし。
### return
Array  
レイヤーの形状。Layerクラスのshapeメソッドでは、前レイヤーの形状を返却します。

## def prev_layer
前のレイヤーを取得します。
### arguments
なし。
### return
Layer  
前のレイヤー。


# class HasParamLayer < Layer
学習可能なパラメータを持つ全てのレイヤーのスーパークラスです。

## 【Instance methods】
## def initialize
コンストラクタ
### arguments
なし。

## override def init(model)
Layerクラスからオーバーライドされたメソッドです。
init_paramの呼び出しを行います。

## def update
オプティマイザーを用いてパラメータの更新を行います。
### arguments
なし。
### return
なし。

## private abstruct def init_params
更新可能なパラメータを初期化します。HasParamLayerクラスを継承するクラスは、このメソッドを実装する必要があります。
### arguments
なし。
### return
なし。


# class InputLayer < Layer
入力層に該当するレイヤーです。モデルの先頭レイヤーは、必ずこのクラスのインスタンスでなければなりません。

## 【Properties】
## attr_reaedr :shape  
SFloat shape  
コンストラクタで設定されたshapeを取得します。

## 【Instance methods】
## def initialize(dim_or_shape)
コンストラクタ
### arguments
* Integer|Array dim_or_shape  
入力層のdimentionまたはshapeを指定します。引数がIntegerだとdimentionとみなし、Arrayだとshapeとみなします。

## override def forward(x)
入力値をそのまま順方向に伝搬します。

## override def backward(dout)
逆方向から伝搬してきた微分値をそのまま逆方向に伝搬します。


# class Dense
全結合レイヤーを扱うクラスです。

## 【propaty】
## attr_reader :num_nodes
Integer  
レイヤーのノード数を取得します。

## attr_reader :weight_decay
Float  
重み減衰の係数を取得します。

## 【Instance methods】
## def initialize(num_nodes, weight_initializer: nil, bias_initializer: nil, weight_decay: 0)
コンストラクタ。
### arguments
* Integer num_nodes  
レイヤーのノード数を設定します。
* Initializer weight_initializer: nil  
重みの初期化に使用するイニシャライザーを設定します
nilを指定すると、RandomNormalイニシャライザーが使用されます。  
* Initializer bias_initializer: nil  
バイアスの初期化に使用するイニシャライザーを設定します。
nilを指定すると、Zerosイニシャライザーが使用されます。
* Float weight_decay: 0
重み減衰の係数を設定します。

## override def forward(x)
ノードを順方向に伝搬します。

## override def backward(dout)
ノードを逆方向に伝搬します。

## override def shape
[ノード数]をshapeとして返却します。


# class Conv2D < HasParamLayer
畳み込みレイヤーを扱うクラスです。

## 【Instance methods】
## def initialize(num_filters, filter_height, filter_width, weight_initializer: nil, bias_initializer: nil, strides: [1, 1], padding 0, weight_decay: 0)
コンストラクタ。
### arguments
* Integer num_filters  
出力するフィルターの枚数
* Integer filter_height  
フィルターの縦の長さ
* Integer filter_width
フィルターの横の長さ
* Initializer weight_initializer: nil  
重みの初期化に使用するイニシャライザーを設定します
nilを指定すると、RandomNormalイニシャライザーが使用されます。  
* Initializer bias_initializer: nil  
バイアスの初期化に使用するイニシャライザーを設定します。
* Array<Integer> strides: [1, 1]  
畳み込みを行う際のストライドの単位を指定します。配列の要素0でy軸方向のストライドを設定し、要素1でx軸方向のストライドを設定します。
* Integer padding: 0  
イメージに対してゼロパディングを行う単位を指定します。
* Float weight_decay: 0  
重み減衰を行うL2正則化項の強さを設定します。

## override def init(model)
モデルのコンパイル時に、レイヤーを初期化するために使用されます。

## override def forward(x)
イメージにフィルターを適用して順方向に伝搬します。

## override def backward(dout)
フィルターが適用されたイメージを変換して、逆方向に伝搬します。

## override def shape
畳み込み後のイメージの次元を返します。


# class MaxPool2D < Layer
maxプーリングを行うレイヤーです。

## 【Instance methods】
## def initialize(pool_height, pool_width, strides: nil, padding: 0)
コンストラクタ。
### arguments
* Integer pool_height  
プーリングを行う縦の長さ。
* Integer pool_width  
プーリングを行う横の長さ。
* Array<Integer> strides: nil  
畳み込みを行う際のストライドの単位を指定します。配列の要素0でy軸方向のストライドを設定し、要素1でx軸方向のストライドを設定します。なお、nilが設定された場合は、[pool_height, pool_width]がstridesの値となります。
* Integer padding: 0
イメージに対してゼロパディングを行う単位を指定します。

## override def init(model)
モデルのコンパイル時に、レイヤーを初期化するために使用されます。

## override def forward(x)
イメージにプーリングを行い、順方向に伝搬します。

## override def backward(dout)
プーリングされたイメージを変換し、逆方向に伝搬します。

## override def shape
プーリング後のイメージのshapeを返します。


# class Flatten
N次元のデータを平坦化します。

## 【Instance methods】
## override def forward(x)
データを平坦化して、順方向に伝搬します。

## override def backward(dout)
データを元の形状に戻し、逆方向に伝搬します。

## override def shape
前レイヤーの形状を平坦化して返します。


# class Reshape < Layer
データの形状を変更します。

## 【Instance methods】
## def initialize(shape)
コンストラクタ。
### arguments
* Array<Integer> shape  
データの形状を変更するshapeです。

## override def forward(x)
データをコンストラクタで指定したshapeにreshapeして、順方向に伝搬します。

## override def backward(dout)
データを元のshapeにreshapeして、逆方向に伝搬します。

# class OutputLayer < Layer
出力層に該当するレイヤーです。出力層の活性化関数は、全てこのクラスを継承する必要があります。

## abstruct def backward(y)
出力層の活性化関数と損失関数を合わせたものを微分した導関数を用いて、教師データの出力データを逆方向に伝搬します。
### arguments
SFloat y
出力データ。
### return
出力層の活性化関数と損失関数の微分値。

## abstruct def loss
損失関数の値を取得します。
### arguments
SFloat y  
出力データ。
### return
損失関数の値。

## def ridge
L2正則化係数を用いて、L2正則化項の値を計算して取得します。
### arguments
なし。
### return
SFloat  
L2正則化項の値を取得します。


# class Dropout
学習の際に、一部のノードを非活性化させるクラスです。

## def initialize(dropout_ratio)
コンストラクタ。
### arguments
* Float dropout_ration  
ノードを非活性にする割合。

## abstruct def forward(x)
一部のノードを非活性にした上で、順方向に伝搬します。

## abstruct def backward(dout)
一部の非活性のノード以外の全てのノードを逆方向に伝搬します。

# class BatchNormalization < HasParamLayer
ミニバッチ単位でのデータの正規化を行います。

## override def forward(x)
正規化したデータを順方向に伝搬します。

## override def backward(dout)
正規化したデータを微分して、逆方向に伝搬します。

# module Activations
活性化関数のレイヤーの名前空間をなすモジュールです。

# module SigmoidFunction
シグモイド関数を提供するモジュールです。

## def forward(x)
シグモイド関数の値を順方向に伝搬します。
### arguments
SFloat x  
シグモイド関数の引数。
### return
SFloat  
シグモイド関数の戻り値


# class Sigmoid < Layer
## include SigmoidFunction
シグモイド関数のレイヤーです。

## override def forward(x)
シグモイド関数の値を順方向に伝搬します。

## def backward(dout)
### arguments
SFloat dout  
シグモイド関数の導関数を適用した値を逆伝搬する。
### return
SFloat  
シグモイド関数の導関数を適用した逆伝搬の値。


# class Tanh < Layer
tanh関数のレイヤーです。
## def forward(x)
tanh関数の値を順方向に伝搬します。
### arguments
SFloat x  
tanh関数の引数。
### return
SFloat  
tanh関数の戻り値

## def backward(dout)
### arguments
SFloat dout  
tanh関数の導関数を適用した値を逆伝搬する。
### return
SFloat  
tanh関数の導関数を適用した逆伝搬の値。


# class ReLU < Layer
ランプ関数のレイヤーです。
## def forward(x)
ランプ関数の値を順方向に伝搬します。
### arguments
SFloat x  
ランプ関数の引数。
### return
SFloat  
ランプ関数の戻り値

## def backward(dout)
### arguments
SFloat dout  
ランプ関数の導関数を適用した値を逆伝搬する。
### return
SFloat  
ランプ関数の導関数を適用した逆伝搬の値。


# class LeakyReLU < Layer
LeakyReLU関数のレイヤーです。
## def forward(x)
LeakyReLU関数の値を順方向に伝搬します。
### arguments
SFloat x  
LeakyReLU関数の引数。
### return
SFloat  
LeakyReLU関数の戻り値

## def backward(dout)
### arguments
SFloat dout  
LeakyReLU関数の導関数を適用した値を逆伝搬する。
### return
SFloat  
LeakyReLU関数の導関数を適用した逆伝搬の値。


# class IdentityWithLoss < OutputLayer
恒等関数と二乗誤差関数を合わせた出力層のレイヤーです。
## override def forward(x)
データをそのまま順方向に伝搬します。
## override def backward(y)
恒等関数と二乗誤差関数を合わせたものを微分した導関数を用いて、教師データの出力データを逆方向に伝搬します。


# class SoftmaxWithLoss < OutputLayer
ソフトマックス関数とクロスエントロピー誤差関数を合わせた出力層のレイヤーです。
## override def forward(x)
ソフトマックス関数の値を順方向に伝搬します。
## override def backward(y)
ソフトマックス関数とクロスエントロピー誤差関数を合わせたものを微分した導関数を用いて、教師データの出力データを逆方向に伝搬します。


# class SigmoidWithLoss < OutputLayer
シグモイド関数とバイナリクロスエントロピー誤差関数を合わせた出力層のレイヤーです。
## override def forward(x)
シグモイド関数の値を順方向に伝搬します。
## override def backward(y)
シグモイド関数とバイナリクロスエントロピー誤差関数を合わせたものを微分した導関数を用いて、教師データの出力データを逆方向に伝搬します。


# module Initializers
全てのInitializerの名前空間をなすモジュールです。


# class Initializer
全てのInitializeクラスのスーパークラスです。

## def init_param(layer, param_key, param)
レイヤーの持つパラメータを更新します。
### arguments
* HasParamLayer layer  
更新対象のパラメータを持つレイヤーを指定します。
* Symbol param_key  
更新す対象のパラメータの名前を指定します。
* SFloat param  
更新するパラメータです。


# class Zeros < Initializer
パラメータを0で初期化します。

## override def init_param(layer, param_key)
レイヤーの持つパラメータを0で初期化します。

# class RandomNormal < Initializer
パラメータを正規分布による乱数で初期化します。

## def initialize(mean = 0, std = 0.05)
### arguments
Float mean = 0  
正規分布の平均。
Float std = 0.05  
正規分布の分散。

## override def init_param(layer, param_key)
レイヤーの持つパラメータを正規分布による乱数で初期化します。


# class Xavier < Initializer
パラメータをXavierの初期値で初期化します。

## override def init_param(layer, param_key)
レイヤーの持つパラメータをXavierの初期値で初期化します。


# class He < Initializer
パラメータをHeの初期値で初期化します。

## override def init_param(layer, param_key)
レイヤーの持つパラメータをHeの初期値で初期化します。


# module Optimizers
全てのOptimizerの名前空間をなすモジュールです。


# class Optimizer
全てのOptimizerのスーパークラスです。

## 【Properties】

## attr_accessor :learning_rate
Float learning_rate  
学習率のプロパティです。

## 【Instance methods】

## def initialize(learning_rate)
コンストラクタ。
### arguments
Float learning_rate  
Optimizerの学習率。

## abstruct def update(layer)
layerのgradsを元に、layerのparamsを更新します。
### arguments
Layer layer  
paramsを更新するレイヤー。
### return
なし。


# class SGD < Optimizer
SGDによるオプティマイザです。

## 【Properties】

## attr_accessor :momentum
Float momentum  
モーメンタム係数。

## 【Instance methods】

## override def initialize(learning_rate = 0.01, momentum: 0)
コンストラクタ。
### arguments
Float learning_rate  
学習率。
Float momentum  
モーメンタム係数。


# class AdaGrad < Optimizer
AdaGradによるオプティマイザです。


# class RMSProp < Optimizer
RMSPropによるオプティマイザです。

## 【Properties】

## attr_accessor :muse
Float muse  
指数平均移動のための係数。


# class Adam < Optimizer
Adamによるオプティマイザです。

## 【Properties】

## attr_accessor :beta1
Float beta1  
指数平均移動のための係数1。

## attr_accessor :beta2
Float beta2
指数平均移動のための係数2。


# module Util
ユーティリティ関数を提供します。

## 【Singleton methods】

## def self.get_minibatch(x, y, batch_size)
batch_size分のミニバッチを取得します。
### arguments
SFloat x  
教師データの入力データ。
SFloat y  
教師データの出力データ。
Integer batch_size  
ミニバッチのサイズ。
### return
Array  
[xのミニバッチ, yのミニバッチ]の形式の配列を返します。

## def self.to_categorical(y, num_classes, type = nil)
ラベルをnum_classesのベクトルにカテゴライズします。
### arguments
SFloat y  
教師データの出力データ。
Integer num_classes  
カテゴライズするクラス数。
NArray narray_type = nil  
カテゴライズしたNArrayデータの型。nilを指定すると、yの型を使用します。
### return
NArray  
カテゴライズされたNArrayのインスタンス。

## def self.numerical_grad(x, func)
引数で渡された関数を数値微分します。
### arguments
SFloat x  
funcの引数。
Proc|Method func  
数値微分を行う対象の関数。
### return
SFloat  
数値微分した結果の値。
