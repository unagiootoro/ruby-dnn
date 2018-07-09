# APIリファレンス
ruby-dnnのAPIリファレンスです。このリファレンスでは、APIを利用するうえで必要となるクラスとメソッドしか記載していません。
そのため、プログラムの詳細が必要な場合は、ソースコードを参照してください。

対応バージョン:0.1.7

# module DNN
ruby-dnnの名前空間をなすモジュールです。

## 【Constants】
## VERSION
ruby-dnnのバージョン。


# class Model
 ニューラルネットワークのモデルを作成するクラスです。

## 【Singleton methods】

## def self.load(file_name)
marshalファイルを読み込み、モデルを作成します。
### arguments
* String file_name  
  読み込むmarshalファイル名。
### return
Model  
生成したモデル。

## def self.load_json(json_str)
json文字列からモデルを作成します。
### arguments
* String json_str
  json文字列。
### return
Model  
生成したモデル。

## 【Instance methods】

## def load_json_params(json_str)
学習パラメータをjson文字列から取得し、モデルにセットします。
### arguments
* String json_str
学習パラメータを変換して生成したjson文字列。
### return
なし。

## def save(file_name)
モデルをmarshalファイルに保存します。
このmarshalファイルには、学習パラメータ及びOptimizerの状態が保存されるため、
読み込んだ後、正確に学習を継続することができます。
### arguments
* String file_name  
書き込むファイル名。
### return
なし。

## def to_json
モデルをjson文字列に変換します。
変換したjson文字列には学習パラメータの情報は含まれません。
学習パラメータの情報を取得したい場合は、params_to_jsonを使用してください。
### arguments
なし。
### return
String  
モデルを変換して生成したjson文字列。

## def params_to_json
学習パラメータをjson文字列に変換します。
### arguments
なし。
### return
String  
学習パラメータを変換して生成したjson文字列。

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

## abstruct def to_hash
レイヤーをハッシュに変換します。このメソッドは、モデルをjsonに変換するために使用されます。このメソッドが返すハッシュの要素には、{name: self.class.name}が含まれていなければなりません。
### arguments
なし。
### return
Hash  
レイヤーを変換したハッシュ。


# class HasParamLayer < Layer
学習可能なパラメータを持つ全てのレイヤーのスーパークラスです。

## 【Instance methods】

## private abstruct def init_params
更新可能なパラメータを初期化します。HasParamLayerクラスを継承するクラスは、このメソッドを実装する必要があります。
### arguments
なし。
### return
なし。


# class InputLayer < Layer
入力層に該当するレイヤーです。モデルの先頭レイヤーは、必ずこのクラスのインスタンスでなければなりません。

## 【Instance methods】
## def initialize(dim_or_shape)
コンストラクタ
### arguments
* Integer|Array dim_or_shape  
入力層のdimentionまたはshapeを指定します。引数がIntegerだとdimentionとみなし、Arrayだとshapeとみなします。


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


# class Flatten
N次元のデータを平坦化します。


# class Reshape < Layer
データの形状を変更します。

## 【Instance methods】
## def initialize(shape)
コンストラクタ。
### arguments
* Array<Integer> shape  
データの形状を変更するshapeです。


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


# class Dropout
学習の際に、一部のノードを非活性化させるクラスです。

## def initialize(dropout_ratio)
コンストラクタ。
### arguments
* Float dropout_ration  
ノードを非活性にする割合。


# class BatchNormalization < HasParamLayer
ミニバッチ単位でのデータの正規化を行います。


# module Activations
活性化関数のレイヤーの名前空間をなすモジュールです。


# class Sigmoid < Layer
シグモイド関数のレイヤーです。


# class Tanh < Layer
tanh関数のレイヤーです。


# class ReLU < Layer
ランプ関数のレイヤーです。


# class LeakyReLU < Layer
LeakyReLU関数のレイヤーです。

## 【Instance methods】

## def initialize(alpha)
コンストラクタ。
### arguments
* Float alpha  
出力値が負のときの傾き。


# class IdentityWithLoss < OutputLayer
恒等関数と二乗誤差関数を合わせた出力層のレイヤーです。


# class SoftmaxWithLoss < OutputLayer
ソフトマックス関数とクロスエントロピー誤差関数を合わせた出力層のレイヤーです。


# class SigmoidWithLoss < OutputLayer
シグモイド関数とバイナリクロスエントロピー誤差関数を合わせた出力層のレイヤーです。


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


# class RandomNormal < Initializer
パラメータを正規分布による乱数で初期化します。

## def initialize(mean = 0, std = 0.05)
### arguments
* Float mean = 0  
  正規分布の平均。
* Float std = 0.05  
  正規分布の分散。


# class Xavier < Initializer
パラメータをXavierの初期値で初期化します。


# class He < Initializer
パラメータをHeの初期値で初期化します。


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
* Float learning_rate  
  Optimizerの学習率。

## abstruct def update(layer)
layerのgradsを元に、layerのparamsを更新します。全てのOptimizerを継承するクラスは、このメソッドを実装する必要があります。
### arguments
* Layer layer  
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

## def initialize(learning_rate = 0.01, momentum: 0)
コンストラクタ。
### arguments
* Float learning_rate  
  学習率。
* Float momentum  
  モーメンタム係数。


# class AdaGrad < Optimizer
AdaGradによるオプティマイザです。


# class RMSProp < Optimizer
RMSPropによるオプティマイザです。

## 【Properties】

## attr_accessor :muse
Float muse  
指数平均移動のための係数。

## 【Instance methods】

## def initialize(learning_rate = 0.001, muse = 0.9)
コンストラクタ。
### arguments
* Float learning_rate  
  学習率。
* Float muse  
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

## 【Instance methods】

## def initialize(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999)
コンストラクタ。
### arguments
* Float beta1
  指数平均移動のための係数1。
* Float beta2
  指数平均移動のための係数2。


# module Util
ユーティリティ関数を提供します。

## 【Singleton methods】

## def self.get_minibatch(x, y, batch_size)
batch_size分のミニバッチを取得します。
### arguments
* SFloat x  
  教師データの入力データ。
* SFloat y  
  教師データの出力データ。
* Integer batch_size  
  ミニバッチのサイズ。
### return
Array  
[xのミニバッチ, yのミニバッチ]の形式の配列を返します。

## def self.to_categorical(y, num_classes, type = nil)
ラベルをnum_classesのベクトルにカテゴライズします。
### arguments
* SFloat y  
  教師データの出力データ。
*  Integer num_classes  
  カテゴライズするクラス数。
*  NArray narray_type = nil  
  カテゴライズしたNArrayデータの型。nilを指定すると、yの型を使用します。
### return
NArray  
カテゴライズされたNArrayのインスタンス。
