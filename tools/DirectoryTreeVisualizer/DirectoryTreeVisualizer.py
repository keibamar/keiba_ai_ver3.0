import os
import re

def list_directory_contents(directory, prefix='', data_mode=False):
    structure = ''
    items = sorted(os.listdir(directory), key=lambda x: (not os.path.isdir(os.path.join(directory, x)), x.lower()))
    pointers = ['|--- '] * (len(items) - 1) + ['|--- ']
    date_pattern = re.compile(r'^\d{8}$')

    for pointer, item in zip(pointers, items):
        path = os.path.join(directory, item)
        
        if item == '.git' or item.endswith('.gitignore') or item.endswith('.csv') or item.endswith('.pickle') or item.endswith('.txt'):
            continue

        if date_pattern.match(item):
            continue
        
        if not data_mode or os.path.isdir(path):
            structure += prefix + pointer + item + '\n'
        
        if os.path.isdir(path):
            extension = '|   ' if pointer == '|-- ' else '    '
            structure += list_directory_contents(path, prefix + extension, data_mode=(item == 'data'))
    
    return structure

def write_structure_to_file(directory, output_file):
    structure = list_directory_contents(directory)
    
    with open(output_file, 'w') as file:
        file.write(structure)

if __name__ == '__main__':
    directory_to_scan = input('ディレクトリパスを入力してください（未入力の場合は現在のディレクトリを使用します）: ').strip()
    
    if not directory_to_scan:
        directory_to_scan = os.getcwd()
        print(f'入力がなかったため、現在のディレクトリを使用します: {directory_to_scan}')
    
    output_file = 'DirectoryTree.txt'
    
    write_structure_to_file(directory_to_scan, output_file)
    print(f'フォルダ構成が {output_file} に書き出されました。')
