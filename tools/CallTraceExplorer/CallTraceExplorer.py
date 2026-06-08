import os
import ast
import sys
import threading
import inspect

# グローバル変数として関数の呼び出し順序を記録
call_sequence = []

# トレース関数を定義
def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        func_name = code.co_name
        module_name = frame.f_globals.get('__name__', '')
        filename = frame.f_code.co_filename
        # ビルトイン関数や標準ライブラリを除外
        if not filename.startswith('<') and 'site-packages' not in filename:
            call_sequence.append((module_name, func_name))
    return trace_calls

# フォルダ内の全ての関数をリストアップ
def list_functions(directory):
    functions = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        module_name = os.path.splitext(os.path.relpath(filepath, directory))[0].replace(os.sep, '.')
                        full_name = f"{module_name}.{func_name}"
                        functions[full_name] = {'module': module_name, 'name': func_name, 'filepath': filepath}
    return functions

# モジュールを動的にインポート
def import_modules(directory):
    sys.path.insert(0, directory)
    modules = {}
    for root, _, files in os.walk(directory):
        package = os.path.relpath(root, directory).replace(os.sep, '.')
        if package == '.':
            package = ''
        for file in files:
            if file.endswith('.py'):
                mod_name = os.path.splitext(file)[0]
                full_module_name = f"{package}.{mod_name}" if package else mod_name
                try:
                    modules[full_module_name] = __import__(full_module_name, fromlist=[''])
                except Exception as e:
                    print(f"モジュール '{full_module_name}' のインポートに失敗しました: {e}")
    return modules

# 関数呼び出しの出力と可視化
def visualize_call_sequence(call_sequence):
    # 呼び出された関数のペアを作成
    call_pairs = []
    for i in range(len(call_sequence)-1):
        caller = call_sequence[i]
        callee = call_sequence[i+1]
        call_pairs.append((caller, callee))

    # グラフの作成
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("networkx および matplotlib が必要です。インストールしてください。")
        return

    G = nx.DiGraph()
    for caller, callee in call_pairs:
        G.add_edge(f"{caller[0]}.{caller[1]}", f"{callee[0]}.{callee[1]}")

    plt.figure(figsize=(12,8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=1500)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.title("関数呼び出し順序のグラフ")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    directory = input("解析するフォルダのパスを入力してください: ")
    print("関数を解析中...")
    functions = list_functions(directory)
    print("検出された関数の一覧:")
    for i, func in enumerate(functions.keys()):
        print(f"{i}: {func}")
    func_index = int(input("実行したい関数の番号を入力してください: "))
    selected_func_name = list(functions.keys())[func_index]
    selected_func_info = functions[selected_func_name]

    # 引数を取得
    args_input = input("引数をカンマ区切りで指定してください（例: arg1, 2, 'text'): ")
    args = eval(f"({args_input},)")

    # モジュールをインポート
    modules = import_modules(directory)

    # 選択された関数を取得
    target_module = modules.get(selected_func_info['module'])
    if not target_module:
        print(f"モジュール {selected_func_info['module']} が見つかりません。")
        return
    target_func = getattr(target_module, selected_func_info['name'], None)
    if not target_func:
        print(f"関数 {selected_func_info['name']} が見つかりません。")
        return

    # トレースを開始して関数を実行
    global call_sequence
    call_sequence = []
    sys.settrace(trace_calls)
    try:
        threading.Thread(target=target_func, args=args).start()
    except Exception as e:
        print(f"関数の実行中にエラーが発生しました: {e}")
    sys.settrace(None)

    # 呼び出し順序を表示
    print("呼び出された関数の順序:")
    for i, (mod_name, func_name) in enumerate(call_sequence):
        print(f"{i+1}: {mod_name}.{func_name}")

    # 可視化
    visualize_call_sequence(call_sequence)

if __name__ == '__main__':
    main()
