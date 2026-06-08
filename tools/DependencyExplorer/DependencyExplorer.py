import os
import ast
import networkx as nx
import matplotlib.pyplot as plt

def get_python_files(directory):
    py_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory)
                py_files[relative_path] = full_path
    return py_files

def extract_imports(file_path, current_file, project_files):
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            root = ast.parse(file.read(), filename=file_path)
        except SyntaxError as e:
            print(f"構文エラーが発生しました: {file_path}\n{e}")
            return set()

    imports = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                module_file = module_name.replace('.', os.sep) + '.py'
                if module_file in project_files and module_file != current_file:
                    imports.add(module_file)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module
                module_file = module_name.replace('.', os.sep) + '.py'
                if module_file in project_files and module_file != current_file:
                    imports.add(module_file)
            else:
                # 相対インポートの処理
                level = node.level
                module_file = resolve_relative_import(current_file, level)
                if module_file and module_file != current_file:
                    imports.add(module_file)
    return imports

def resolve_relative_import(current_file, level):
    parts = current_file.split(os.sep)
    if level > len(parts):
        return None
    return os.sep.join(parts[:-level] + ['__init__.py'])

def build_dependency_graph(py_files):
    project_files = set(py_files.keys())
    graph = nx.DiGraph()

    # ノードの追加
    for file_name in py_files.keys():
        graph.add_node(file_name)

    # エッジの追加
    for file_name, file_path in py_files.items():
        imports = extract_imports(file_path, file_name, project_files)
        for imported_file in imports:
            if imported_file in project_files:
                graph.add_edge(file_name, imported_file)
    return graph

def get_directory_depths(py_files):
    depths = {}
    for file_name in py_files.keys():
        depth = file_name.count(os.sep)
        depths[file_name] = depth
    return depths

def compute_positions(graph, depths):
    positions = {}
    # 深さの最大値を取得
    max_depth = max(depths.values())
    # 各深さごとにノードを集める
    depth_nodes = {}
    for node, depth in depths.items():
        depth_nodes.setdefault(depth, []).append(node)
    # ノードの位置を設定
    y_gap = 1.0
    y_pos = max_depth * y_gap
    for depth in range(max_depth + 1):
        nodes = depth_nodes.get(depth, [])
        x_gap = 1.0 / (len(nodes) + 1)
        x_pos = x_gap
        for node in nodes:
            positions[node] = (x_pos, y_pos)
            x_pos += x_gap
        y_pos -= y_gap
    return positions

def set_japanese_font():
    # 日本語フォントの設定
    import matplotlib.font_manager as fm
    from matplotlib import rcParams

    rcParams['font.family'] = 'Yu Gothic'  # お使いのシステムの日本語フォントに変更してください

def visualize_graph(graph, py_files):
    set_japanese_font()  # フォントの設定

    depths = get_directory_depths(py_files)
    pos = compute_positions(graph, depths)

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=1500, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.title("Pythonファイル間の依存関係グラフ")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    directory = input("解析したいフォルダのパスを入力してください: ")
    if not os.path.isdir(directory):
        print("有効なフォルダパスを入力してください。")
        return

    print("Pythonファイルを収集しています...")
    py_files = get_python_files(directory)
    if not py_files:
        print("指定されたフォルダにPythonファイルが見つかりませんでした。")
        return

    print("依存関係グラフを構築しています...")
    graph = build_dependency_graph(py_files)

    print("依存関係グラフを可視化しています...")
    visualize_graph(graph, py_files)

if __name__ == '__main__':
    main()
