# PyDepExtractor

## 概要
このスクリプトは、指定したディレクトリ内のPythonファイルを解析し、各ファイルで使用されている外部ライブラリを抽出します。その結果をまとめて、以下の形式でrequirements.txtファイルに出力します。

必要なライブラリ一覧（重複なし）：プロジェクト全体で使用されている外部ライブラリを重複なくリストアップします。

各ファイルで使用されているライブラリ：各Pythonファイルごとに、使用している外部ライブラリをリストアップします。

## 特徴
ローカルモジュールの除外：プロジェクト内のローカルモジュールやパッケージを除外し、外部ライブラリのみを検出します。

標準ライブラリの除外：Pythonの標準ライブラリを除外し、本当に必要な外部ライブラリのみを抽出します。

再帰的な解析：指定されたディレクトリ以下のすべてのPythonファイルを再帰的に解析します。

## 動作環境
Python 3.x

## 使い方
1. スクリプトの配置
このスクリプトを、解析したいPythonプロジェクトのディレクトリに保存します。例えば、dependency_extractor.pyという名前で保存します。

2. 必要なモジュールの確認
特別な外部モジュールは必要ありません。Python標準ライブラリのみで動作します。

3. スクリプトの実行
ターミナルやコマンドプロンプトで、スクリプトが保存されているディレクトリに移動し、以下のコマンドを実行します。

bash
python dependency_extractor.py
4. 解析するディレクトリの指定
スクリプトを実行すると、以下のように入力を求められます。解析したいプロジェクトのディレクトリパスを入力してください。

解析したいフォルダのパスを入力してください: your_project_directory_path
例：

解析したいフォルダのパスを入力してください: /Users/username/projects/my_python_project
5. 結果の確認
スクリプトの実行が完了すると、同じディレクトリ内にrequirements.txtファイルが生成されます。その内容は以下のようになります。
---
# 必要なライブラリ一覧（重複なし）
numpy
pandas
requests

# 各ファイルで使用されているライブラリ
## ファイル: main.py
numpy
pandas

## ファイル: utils/helper.py
requests
---

## 注意点
パッケージ名とインポート名の相違：

一部のライブラリでは、インポート名と実際のパッケージ名が異なる場合があります。例えば、bs4モジュールはbeautifulsoup4パッケージに含まれています。このような場合、requirements.txtの該当箇所を手動で修正してください。

バージョン情報の取得：

このスクリプトはライブラリのバージョン情報を取得しません。バージョン指定が必要な場合は、手動でrequirements.txtを編集するか、追加の機能を実装してください。

特殊なインポート文への対応：

ダイナミックインポートや、文字列によるモジュールの読み込みには対応していません。AST解析で検出可能な範囲のインポート文のみが対象です。

## カスタマイズ方法
バージョン情報の追加
ライブラリのバージョン情報を取得してrequirements.txtに反映させたい場合、以下のようにpkg_resourcesを使用して拡張できます。

python
import pkg_resources

def get_package_version(package_name):
    try:
        version = pkg_resources.get_distribution(package_name).version
        return version
    except pkg_resources.DistributionNotFound:
        return None
上記の関数を活用して、バージョン情報を取得し、requirements.txtに書き込む際に反映させます。

## コマンドライン引数のサポート
argparseモジュールを使用することで、コマンドライン引数から解析対象のディレクトリを指定できるようになります。

python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Pythonプロジェクトの依存関係を解析します。')
    parser.add_argument('directory', help='解析したいフォルダのパス')
    args = parser.parse_args()

    file_imports = get_imports_by_file(args.directory)
    # 以下、既存のコードを続けます
## 出力ファイル名の指定
出力ファイル名を変更したい場合、main関数内で変更できます。

python
output_file = 'my_requirements.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    # 以下、既存のコードを続けます
## トラブルシューティング
依存関係が正しく取得できない場合：

スクリプトが解析できるのは、静的に記述されたインポート文のみです。__import__やimportlibを使用した動的なモジュール読み込みには対応していません。

標準ライブラリが含まれてしまう場合：

環境によっては、標準ライブラリの判定が正確でない場合があります。その際は、exclude_standard_libs関数のロジックを調整してください。

## ライセンス
MITライセンスの下で提供されています。詳細はLICENSEファイルを参照してください。

## 貢献
バグ報告や機能改善の提案は大歓迎です。GitHubリポジトリのIssuesやPull Requestsを通じてご連絡ください。

## 作者
名前：Taiki Akama with Copilot

## 更新履歴
v1.0：初期リリース