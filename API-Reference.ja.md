# APIリファレンス
ruby-dnnのAPIリファレンスです。このリファレンスでは、APIを利用するうえで必要となるクラスとメソッドしか記載していません。
そのため、プログラムの詳細が必要な場合は、ソースコードを参照してください。

最終更新バージョン:0.8.7

# module DNN
ruby-dnnの名前空間をなすモジュールです。

## 【Constants】

## VERSION
ruby-dnnのバージョン。

# class Model
 ニューラルネットワークのモデルを作成するクラスです。

## 【Properties】

## attr_accessor :layer
モデルに追加されたレイヤーの配列を取得します。

## attr_accessor :trainable
falseを設定すると、パラメータの学習を禁止します。


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
モデルにレイヤーまたはモデルを追加します。
### arguments
* Layer | Model layer  
追加するレイヤーまたはモデル。
### return
Model  
自身のモデルのインスタンス。

## def optimizer
モデルのオプティマイザーを取得します。
モデルにオプティマイザーが存在しない場合は、上位のモデルのオプティマイザーを取得します。
### arguments
なし。
### return
Optimizer  
モデルのオプティマイザー。

## def compile(optimizer)
モデルをコンパイルします。
### arguments
* Optimizer optimizer
モデルが学習に使用するオプティマイザー。
### return
なし。

## def compiled?
モデルがコンパイル済みであるか否かを取得します。
### arguments
なし。
### return
bool  
モデルがコンパイル済みであるか否か。

## def train(x, y, epochs, batch_size: 1, test: nil, verbose: true, batch_proc: nil, &epoch_proc)
コンパイルしたモデルを用いて学習を行います。
### arguments
* Numo::SFloat x  
トレーニング用入力データ。
* Numo::SFloat y  
トレーニング用出力データ。
* Integer epochs  
学習回数。
* Integer batch_size: 1  
学習に使用するミニバッチの数。
* Array test: nil  
[テスト用入力データ, テスト用出力データ]の形式で設定すると、1エポックごとにテストを行います。
* bool verbose: true
trueを設定すると、学習ログを出力します。
* Proc batch_proc: nil  
一度のバッチ学習が行われる前に呼び出されるprocを登録します。
### block
epoch_proc  
1エポックの学習が終了するたびに呼び出されます。
### return
なし。

## def train_on_batch(x, y, &batch_proc)
入力されたバッチデータをもとに、一度だけ学習を行います。
### arguments
* Numo::SFloat x
トレーニング用入力バッチデータ。
* Numo::SFloat y
トレーニング用出力バッチデータ。
### block
一度のバッチ学習が行われる前に呼び出されます。
### return
Integer  
損失関数の値を返します。

## def accurate(x, y, batch_size = 100, &batch_proc)
学習結果をもとに認識率を返します。
### arguments
* Numo::SFloat x  
テスト用入力データ。
* Numo::SFloat y  
テスト用出力データ。
* batch_size  
ミニバッチの数。
### block
一度のバッチ学習が行われる前に呼び出されます。
### return
Float
テスト結果の認識率を返します。

## def predict(x)
モデルを使用して、結果の推論を行います。
入力データは、バッチデータである必要があります。
### arguments
* Numo::SFloat x  
推論用入力データ。
### return
Numo::SFloat
推論結果を返します。

## def predict1(x)
モデルを使用して、結果の推論を行います。
predictとは異なり、一つの入力データに対して、一つの出力データを返します。
### arguments
* Numo::SFloat x  
推論用入力データ。
### return
Numo::SFloat
推論結果を返します。

## def copy
現在のモデルをコピーした新たなモデルを生成します。
### arguments
なし。
### return
Model  
コピーしたモデル。

## def get_layer(index)
indexのレイヤーを取得します。
### arguments
* Integer index  
取得するレイヤーのインデックス。
### return
Layer  
対象のレイヤーのインスタンス。

## def get_layer(layer_class, index)
layer_classで指定されたクラスのレイヤーをindexで取得します。
### arguments
* Layer layer_class  
取得するレイヤーのクラス。
* Integer index  
レイヤーのインデックス。例えば、layersが[InputLayer, Dense, Dense,SoftmaxWithLoss]のとき、
最初のDenseを取得したい場合、インデックスは0になります。
### return
Layer  
対象のレイヤーのインスタンス。

## def get_all_layers
モデルが持つ全てのレイヤー(モデルが持つ下位のモデルのレイヤーも含む)を取得します。
### arguments
なし。
### return
Array  
モデルの持つすべてのレイヤーの配列

# module Layers
レイヤーの名前空間をなすモジュールです。


# class Layer
全てのレイヤーのスーパークラスです。

## 【Instance methods】

## def build(model)
モデルのコンパイル時に、レイヤーをビルドするために使用されます。
### arguments
* Model model  
レイヤーを持つモデルを登録します。
### return
なし。

## def built?
レイヤーがビルド済みであるか否かを取得します。
### arguments
なし。
### return
bool  
レイヤーがビルド済みであるか否か。

## abstruct def forward(x)  
順方向伝搬を行うメソッドです。Layerクラスを継承するクラスは、このメソッドを実装する必要があります。
### arguments
* Numo::SFloat x  
入力データ。
### return
Numo::SFloat  
出力データ。

## abstruct def backward(dout)
逆方向伝搬を行うメソッドです。Layerクラスを継承するクラスは、このメソッドを実装する必要があります。
### arguments
* Numo::SFloat dout  
逆方向から伝搬してきた微分値。
### return
Numo::SFloat  
逆方向に伝搬する微分値。

## def shape
レイヤーの形状を取得するメソッドです。
### arguments
なし。
### return
Array  
レイヤーの形状。Layerクラスのshapeメソッドでは、前レイヤーの形状を返却します。

## abstruct def to_hash
レイヤーをハッシュに変換します。このメソッドは、モデルをjsonに変換するために使用されます。このメソッドが返すハッシュの要素には、{name: `self.class.name`}が含まれていなければなりません。
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
このメソッドは、レイヤーが初回ビルドされたときのみ実行されます。
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
* Integer | Array dim_or_shape  
入力層のdimentionまたはshapeを指定します。引数がIntegerだとdimentionとみなし、Arrayだとshapeとみなします。


# class Connection < HasParamLayer
ニューロンを接続するすべてのレイヤーのスーパークラスです。

## 【Properties】

## attr_reader :weight_initializer
Initializer  
重みの初期化に使用するイニシャライザーを取得します。

## attr_reader :bias_initializer
Initializer  
バイアスの初期化に使用するイニシャライザーを取得します。

## attr_reader :l1_lambda
Float  
重みのL1正則化の係数を取得します。

## attr_reader :l2_lambda
Float  
重みのL2正則化の係数を取得します。


# class Dense < Connection
全結合レイヤーを扱うクラスです。

## 【Properties】

## attr_reader :num_nodes
Integer  
レイヤーのノード数を取得します。

## 【Instance methods】

## def initialize(num_nodes, weight_initializer: Initializers::RandomNormal.new, bias_initializer: Initializers::Zeros.new, l1_lambda: 0, l2_lambda: 0)
コンストラクタ。
### arguments
* Integer num_nodes  
レイヤーのノード数を設定します。
* Initializer weight_initializer: Initializers::RandomNormal.new  
重みの初期化に使用するイニシャライザーを設定します。
* Initializer bias_initializer: Initializers::Zeros.new  
バイアスの初期化に使用するイニシャライザーを設定します。
* Float l1_lambda: 0  
重みのL1正則化の係数を設定します。
* Float l2_lambda: 0  
重みのL2正則化の係数を設定します。


# class Conv2D < Connection
畳み込みレイヤーを扱うクラスです。

## 【Properties】

## attr_reader :num_filters
Integer  
出力するフィルターの枚数。

## attr_reader :filter_size
Array   
フィルターの縦と横の長さ。  
[Integer height, Integer width]の形式で取得します。

## attr_reader :strides
Array  
畳み込みを行う際のストライドの単位。  
[Integer height, Integer width]の形式で取得します。

## 【Instance methods】

## def initialize(num_filters, filter_size, weight_initializer: Initializers::RandomNormal.new, bias_initializer: Initializers::Zeros.new, strides: 1, padding false, l1_lambda: 0, l2_lambda: 0)
コンストラクタ。
### arguments
* Integer num_filters  
出力するフィルターの枚数。
* Integer | Array filter_size  
フィルターの縦と横の長さ。
Arrayで指定する場合、[Integer height, Integer width]の形式で指定します。
* Initializer weight_initializer: Initializers::RandomNormal.new  
重みの初期化に使用するイニシャライザーを設定します。
* Initializer bias_initializer: Initializers::Zeros.new  
バイアスの初期化に使用するイニシャライザーを設定します。
* Array<Integer> strides: 1  
畳み込みを行う際のストライドの単位を指定します。
Arrayで指定する場合、[Integer height, Integer width]の形式で指定します。
* bool padding: true  
イメージに対してゼロパディングを行うか否かを設定します。trueを設定すると、出力されるイメージのサイズが入力されたイメージと同じになるように
ゼロパディングを行います。
* Float l1_lambda: 0  
重みのL1正則化の係数を設定します。
* Float l2_lambda: 0  
重みのL2正則化の係数を設定します。


# class Pool2D < Layer
全ての2Dプーリングレイヤーのスーパークラスです。

## 【Properties】

## attr_reader :pool_size
Array   
プーリングを行う縦と横の長さ。
[Integer height, Integer width]の形式で取得します。

## attr_reader :strides
Array  
畳み込みを行う際のストライドの単位。  
[Integer height, Integer width]の形式で取得します。

## 【Instance methods】

## def initialize(pool_size, strides: nil, padding: false)
コンストラクタ。
### arguments
* Integer | Array pool_size  
プーリングを行う縦と横の長さ。
Arrayで指定する場合、[Integer height, Integer width]の形式で指定します。
* Array<Integer> strides: nil  
畳み込みを行う際のストライドの単位を指定します。
Arrayで指定する場合、[Integer height, Integer width]の形式で指定します。
なお、nilが設定された場合は、pool_sizeがstridesの値となります。
* bool padding: true  
イメージに対してゼロパディングを行うか否かを設定します。trueを設定すると、出力されるイメージのサイズが入力されたイメージと同じになるように
ゼロパディングを行います。


# class MaxPool2D < Pool2D
maxプーリングを行うレイヤーです。


# class AvgPool2D < Pool2D
averageプーリングを行うレイヤーです。


# class UnPool2D < Layer
逆プーリングを行うレイヤーです。

## 【Properties】

## attr_reader :unpool_size
Array   
逆プーリングを行う縦と横の長さ。
[Integer height, Integer width]の形式で取得します。

## 【Instance methods】

## def initialize(unpool_size)
コンストラクタ。
### arguments
* Integer unpool_size  
逆プーリングを行う縦と横の長さ。
Arrayで指定する場合、[Integer height, Integer width]の形式で指定します。


# class RNN < Connection
全てのリカレントニューラルネットワークのレイヤーのスーパークラスです。

## attr_reader :num_nodes
Integer  
レイヤーのノード数を取得します。

## attr_reader :stateful
bool  
レイヤーがステートフルであるか否かを返します。

## 【Instance methods】

## def initialize(num_nodes, stateful: false, return_sequences: true, weight_initializer: Initializers::RandomNormal.new, bias_initializer: Initializers::Zeros.new, l1_lamda: 0, l2_lambda: 0)
コンストラクタ。
### arguments
* Integer num_nodes  
レイヤーのノード数を設定します。
* bool stateful  
trueを設定すると、一つ前に計算した中間層の値を使用して学習を行うことができます。
* bool return_sequences
trueを設定すると、時系列ネットワークの中間層全てを出力します。  
falseを設定すると、時系列ネットワークの中間層の最後のみを出力します。
* Initializer weight_initializer: Initializers::RandomNormal.new  
重みの初期化に使用するイニシャライザーを設定します。
* Initializer bias_initializer: Initializers::Zeros.new  
バイアスの初期化に使用するイニシャライザーを設定します。
* Float l1_lambda: 0  
重みのL1正則化の係数を設定します。
* Float l2_lambda: 0  
重みのL2正則化の係数を設定します。

## def reset_state
中間層のステートをリセットします。


# class SimpleRNN < RNN
シンプルなRNNレイヤーを扱うクラスです。

## 【Instance methods】

## def initialize(num_nodes, stateful: false, return_sequences: true,  activation: Tanh.new, weight_initializer: Initializers::RandomNormal.new, bias_initializer: Initializers::Zeros.new, l1_lamda: 0, l2_lambda: 0)
コンストラクタ。
### arguments
* Integer num_nodes  
レイヤーのノード数を設定します。
* bool stateful
trueを設定すると、一つ前に計算した中間層の値を使用して学習を行うことができます。
* bool return_sequences
trueを設定すると、時系列ネットワークの中間層全てを出力します。  
falseを設定すると、時系列ネットワークの中間層の最後のみを出力します。
* Layer activation
リカレントニューラルネットワークにおいて、使用する活性化関数を設定します。
* Initializer weight_initializer: nil  
重みの初期化に使用するイニシャライザーを設定します。
* Initializer bias_initializer: nil  
バイアスの初期化に使用するイニシャライザーを設定します。
* Float l1_lambda: 0  
重みのL1正則化の係数を設定します。
* Float l2_lambda: 0  
重みのL2正則化の係数を設定します。


# class LSTM < RNN
LSTMレイヤーを扱うクラスです。


# class GRU < RNN
GRUレイヤーを扱うクラスです。


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

## 【Instance methods】

## abstruct def backward(y)
出力層の活性化関数と損失関数を合わせたものを微分した導関数を用いて、教師データの出力データを逆方向に伝搬します。
### arguments
Numo::SFloat y
出力データ。
### return
出力層の活性化関数と損失関数の微分値。

## abstruct def loss
損失関数の値を取得します。
### arguments
Numo::SFloat y  
出力データ。
### return
損失関数の値。


# class Dropout
学習の際に、一部のノードを非活性化させるクラスです。

## 【Properties】

## attr_reader :dropout_ratio
ノードを非活性にする割合を取得します。

## 【Instance methods】

## def initialize(dropout_ratio = 0.5)
コンストラクタ。
### arguments
* Float dropout_ratio  
ノードを非活性にする割合。


# class BatchNormalization < HasParamLayer
ミニバッチ単位でのデータの正規化を行います。

## 【Properties】

## attr_reader :momentum
推論時に使用する平均と分散を求めるための指数平均移動の係数。

## 【Instance methods】

## def initialize(momentum: 0.9
コンストラクタ。
### arguments
* Float momenum: 0.9  
推論時に使用する平均と分散を求めるための指数平均移動の係数。


# module Activations
活性化関数のレイヤーの名前空間をなすモジュールです。


# class Sigmoid < Layer
シグモイド関数のレイヤーです。


# class Tanh < Layer
tanh関数のレイヤーです。


# class Softsign < Layer
softsign関数のレイヤーです。


# class Softplus < Layer
softplus関数のレイヤーです。


# class Swish < Layer
swish関数のレイヤーです。


# class ReLU < Layer
ランプ関数のレイヤーです。


# class LeakyReLU < Layer
LeakyReLU関数のレイヤーです。

## 【Properties】
## attr_reader :alpha
Float alpha  
出力値が負のときの傾き。

## 【Instance methods】

## def initialize(alpha)
コンストラクタ。
### arguments
* Float alpha  
出力値が負のときの傾き。


# class ELU < Layer
eLU関数のレイヤーです。

## 【Properties】
## attr_reader :alpha
Float alpha  
出力値が負のときの傾き。

## 【Instance methods】

## def initialize(alpha)
コンストラクタ。
### arguments
* Float alpha  
出力値が負のときの傾き。


# class IdentityMSE < OutputLayer
恒等関数と二乗誤差関数を合わせた出力層のレイヤーです。


# class IdentityMAE < OutputLayer
恒等関数と平均絶対誤差関数を合わせた出力層のレイヤーです。


# class SoftmaxWithLoss < OutputLayer
ソフトマックス関数とクロスエントロピー誤差関数を合わせた出力層のレイヤーです。


# class SigmoidWithLoss < OutputLayer
シグモイド関数とバイナリクロスエントロピー誤差関数を合わせた出力層のレイヤーです。


# module Initializers
全てのInitializerの名前空間をなすモジュールです。


# class Initializer
全てのInitializeクラスのスーパークラスです。

## 【Instance methods】

## def init_param(layer, param)
レイヤーの持つパラメータを更新します。
### arguments
* HasParamLayer layer  
更新対象のパラメータを持つレイヤーを指定します。
* Param param  
更新するパラメータです。


# class Zeros < Initializer
パラメータを0で初期化します。


# class Const < Initializer
パラメータを指定の定数で初期化します。

## 【Instance methods】
## def initialize(const)
### arguments
* Float const  
初期化する定数


# class RandomNormal < Initializer
パラメータを正規分布による乱数で初期化します。

## 【Properties】
## attr_reader :mean
Float mean  
正規分布の平均。
## attr_reader :std
Float std  
正規分布の分散。

## 【Instance methods】
## def initialize(mean = 0, std = 0.05)
### arguments
* Float mean = 0  
正規分布の平均。
* Float std = 0.05  
正規分布の分散。


# class RandomUniform < Initializer
パラメータを一様分布による乱数で初期化します。

## 【Properties】
## attr_reader :min
Float min  
一様分布の最小値。
## attr_reader :max
Float max  
一様分布の最大値。

## 【Instance methods】
## def initialize(min = -0.05, max = 0.05)
### arguments
* Float min = -0.05  
一様分布の最小値。
* Float max = 0.05  
一様分布の最大値。


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

## abstruct def update(params)
paramsが持つ全ての学習パラメータにおいて、gradを元に、dataを更新します。全てのOptimizerを継承するクラスは、このメソッドを実装する必要があります。
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


# class Nesterov < SGD
Nesterovによるオプティマイザです。

## 【Instance methods】

## def initialize(learning_rate = 0.01, momentum: 0.9)
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

## attr_accessor :alpha
Float alpha  
指数平均移動のための係数。

## 【Instance methods】

## def initialize(learning_rate = 0.001, alpha: 0.9)
コンストラクタ。
### arguments
* Float learning_rate  
  学習率。
* Float alpha  
  指数平均移動のための係数。


# class AdaDelta < Optimizer
AdaDeltaによるオプティマイザです。

## 【Properties】

## attr_accessor :rho
Float rho  
指数平均移動のための係数。

## 【Instance methods】

## def initialize(rho: 0.95)
コンストラクタ。
### arguments
* Float rho  
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

## def initialize(learning_rate = 0.001, beta1: 0.9, beta2: 0.999)
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
* Numo::SFloat x  
  教師データの入力データ。
* Numo::SFloat y  
  教師データの出力データ。
* Integer batch_size  
  ミニバッチのサイズ。
### return
Array  
[xのミニバッチ, yのミニバッチ]の形式の配列を返します。

## def self.to_categorical(y, num_classes, type = nil)
ラベルをnum_classesのベクトルにカテゴライズします。
### arguments
* Numo::SFloat y  
  教師データの出力データ。
*  Integer num_classes  
  カテゴライズするクラス数。
*  NArray narray_type = nil  
  カテゴライズしたNArrayデータの型。nilを指定すると、yの型を使用します。
### return
NArray  
カテゴライズされたNArrayのインスタンス。
