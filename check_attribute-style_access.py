import ast
import os
import sys

class DataFrameAttributeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.dataframe_vars = set()
        self.accesses = []

    def visit_Assign(self, node):
        # Check if the assigned value is an instance of DataFrame or Series
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if (node.value.func.attr == 'DataFrame'
                 or node.value.func.attr == 'Series'
                 or node.value.func.attr == "GeoDataFrame"
                 or node.value.func.attr == "Dataset"
                 or node.value.func.attr == "DataArray"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.dataframe_vars.add(target.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Check if the attribute access is on a variable that is a DataFrame or Series
        if isinstance(node.value, ast.Name) and node.value.id in self.dataframe_vars:
            self.accesses.append((node.lineno, node.col_offset, node.value.id, node.attr))
        self.generic_visit(node)

def find_dataframe_attribute_accesses(directory):
    visitor = DataFrameAttributeVisitor()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=file_path)
                        visitor.visit(tree)
                        if visitor.accesses:
                            print(f'In file {file_path}:')
                            for lineno, col_offset, var_name, attr in visitor.accesses:
                                print(f'  Line {lineno}, Column {col_offset}: {var_name}.{attr}')
                        visitor.accesses = []  # Reset for the next file
                    except SyntaxError as e:
                        print(f'Syntax error in file {file_path}: {e}')

# Usage
path_to_check = sys.argv[1]
find_dataframe_attribute_accesses(path_to_check)