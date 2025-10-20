# !/usr/bin/env python
# encoding: utf-8

"""
    Cleanup scripts.
"""

import os, shutil


def remove(base_dir: str, name: str = '.DS_Store'):
    """
        Recursively remove all files or directories named `name` under `base_dir`.
    """
    print('Removing "{}" in {}'.format(name, base_dir))
    for root, dirs, files in os.walk(base_dir):
        # 先处理目录（含同名目录与目录符号链接）
        for d in list(dirs):
            if d == name:
                abs_path = os.path.join(root, d)
                try:
                    if os.path.islink(abs_path):
                        os.unlink(abs_path)
                    else:
                        shutil.rmtree(abs_path)
                    print(f'\tRemoved dir: {abs_path}')
                except Exception as e:
                    print(f'\tFailed dir: {abs_path} -> {e}')
                # 避免继续遍历已删除目录
                if d in dirs:
                    dirs.remove(d)

        # 再处理文件
        for f in files:
            if f == name:
                abs_path = os.path.join(root, f)
                try:
                    os.remove(abs_path)
                    print(f'\tRemoved file: {abs_path}')
                except Exception as e:
                    print(f'\tFailed file: {abs_path} -> {e}')


def clean_dot_files(base_dir: str):
    """
        Recursively delete AppleDouble resource fork files (`._filename`) corresponding to files under `base_dir`.
    """
    print('Cleaning AppleDouble resource fork files in {}'.format(base_dir))
    if not os.path.isdir(base_dir):
        print(f'Path does not exist or is not a directory: {base_dir}')
        return

    for root, _, files in os.walk(base_dir):
        for filename in files:
            # Skip files that are already AppleDouble resource forks
            if filename.startswith("._"):
                continue

            normal_file = os.path.join(root, filename)
            dot_file = os.path.join(root, "._" + filename)

            if os.path.isfile(normal_file) and os.path.exists(dot_file):
                try:
                    os.remove(dot_file)
                    print(f"\tDeleted: {dot_file}")
                except Exception as e:
                    print(f"\tDelete error {dot_file}: {e}")


def main():
    """
        Remove __pycache__, .git directories and .DS_Store files, recursively.
    """
    base_dir = '..'
    remove(base_dir, '.DS_Store')
    remove(base_dir, '__pycache__')
    # remove(base_dir, '.git')
    clean_dot_files(base_dir)

    # base_dir = 'G:/data/time_series'
    # remove(base_dir, '.DS_Store')
    # remove(base_dir, '__pycache__')
    # # remove(base_dir, '.git')
    # clean_dot_files(base_dir)


if __name__ == '__main__':
    main()
