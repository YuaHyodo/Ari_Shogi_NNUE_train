# Ari_Shogi_NNUE_train
Pytorchで将棋のNNUE評価関数の学習をするプログラム
# 概要
- Pytorchで標準のNNUEができるプログラム
- NNUEの学習に関する各種実験用のプログラムのベースとして制作した
- 元々は個人で作って個人で使う予定の公開するつもりのないプログラムだったが、公開してほしいという要望があったので公開する事にした。なので、いつ書いたか覚えてもない非公開の個人用プログラム達から色々と流用して製作されている。(ので非常に非常に汚い)
- 非常に汚いし、バグが潜んでいる確率が物凄く高いので、そのまま利用したり、改造したりするのは推奨しない。
- この学習部のテストとして学習させた評価関数は https://github.com/YuaHyodo/Haojian_nnue で公開している。
- プログラムが汚いとか、変数名おかしいとか、バグがあったとか、そういうクレームに対応しない予定です。ただし、「ライセンス関連で重大な問題を発見した」といった場合には可能な限り早く対処するので、そういうのがある場合は教えてください。

# 色々
- 第34回世界コンピュータ将棋選手権( http://www2.computer-shogi.org/wcsc34/ )に出場する予定のAri Shogi and フレンズは、この学習部を発展させた学習部で作ったNNUE評価関数を搭載する予定
- ゼロからの学習は計算資源の都合上試せていない。
- NNUEの学習では、「ある局面を学習させる際、その局面ではなく、その局面から静止探索を行った際の末端の局面を学習させる」という事が行われる(らしい)が、この学習部にはその機能がない。そのため、そのような学習を行いたい場合は事前に教師データの局面を静止探索の末端の局面に入れ替えておく必要がある。

# ライセンスに関して
- ベースになっているpython-dlshogi2( https://github.com/TadaoYamaoka/python-dlshogi2 )、一部のコードを利用しているNNUE-Pytorchの将棋バージョン( https://github.com/nodchip/nnue-pytorch )がGPL-3.0なのでGPL-3.0に設定しています。詳細はLICENSEファイルを確認してください。
- 入力特徴量に関する部分で利用している https://github.com/bleu48/shogi-eval で公開されているプログラムにはライセンスが明記されていませんが、リポジトリ管理者に問い合わせたところ自由に利用して良いとの事だったので、一部改造して利用しています。
