import os
import ast
from collections import defaultdict

def extract_imports(file_path, project_root):
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            node = ast.parse(file.read(), filename=file_path)
        except SyntaxError:
            return set()
    imports = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                module_name = alias.name.split('.')[0]
                if not is_local_module(module_name, project_root):
                    imports.add(module_name)
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                module_name = n.module.split('.')[0]
                # 相対インポートを除外
                if n.level == 0 and not is_local_module(module_name, project_root):
                    imports.add(module_name)
    return imports

def is_local_module(module_name, project_root):
    # ローカルモジュールかどうかを判定
    for root, dirs, files in os.walk(project_root):
        if f"{module_name}.py" in files or module_name in dirs:
            return True
    return False

def get_imports_by_file(directory):
    file_imports = defaultdict(set)
    for root, _, files in os.walk(directory):
        py_files = [file for file in files if file.endswith('.py')]
        for file in py_files:
            file_path = os.path.join(root, file)
            imports = extract_imports(file_path, directory)
            if imports:
                relative_path = os.path.relpath(file_path, directory)
                file_imports[relative_path].update(imports)
    return file_imports

def exclude_standard_libs(modules):
    import sys
    import pkgutil
    std_libs = set()
    prefixes = [sys.base_prefix, sys.prefix]
    for module_info in pkgutil.iter_modules():
        if any(module_info.module_finder.path.startswith(prefix) for prefix in prefixes):
            std_libs.add(module_info.name)
    return modules - std_libs

def main():
    directory = input("解析したいフォルダのパスを入力してください: ")
    file_imports = get_imports_by_file(directory)

    # 全ファイルの外部ライブラリを収集
    all_imports = set()
    for imports in file_imports.values():
        all_imports.update(imports)
    external_imports = exclude_standard_libs(all_imports)

    with open('requirements.txt', 'w', encoding='utf-8') as f:
        # 重複なしの必要なライブラリ一覧を出力
        f.write("# 必要なライブラリ一覧（重複なし）\n")
        for module in sorted(external_imports):
            f.write(f"{module}\n")
        f.write("\n")
        # 各ファイルごとのライブラリを出力
        f.write("# 各ファイルで使用されているライブラリ\n")
        for file, imports in file_imports.items():
            external_imports = exclude_standard_libs(imports)
            if external_imports:
                f.write(f"## ファイル: {file}\n")
                for module in sorted(external_imports):
                    f.write(f"{module}\n")
                f.write("\n")
    print("requirements.txtが生成されました。")

if __name__ == '__main__':
    main()
