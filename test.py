import tree_sitter
from tree_sitter import Language

print(Language)          # 打印 Language 类信息，确认是否来自 tree_sitter
print(Language.__module__)  # 应该输出 tree_sitter._tree_sitter

import os
print(os.path.exists('parser/my-languages.so'))
