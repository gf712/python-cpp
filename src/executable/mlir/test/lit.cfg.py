# -*- Python -*-
# Lit configuration for the python-cpp MLIR test suite.
#
# Read by lit when discovering tests under src/executable/mlir/test/.
# Substitutes the python-mlir-opt and FileCheck paths into test
# RUN: lines so tests can be written as
#
#     // RUN: python-mlir-opt %s | FileCheck %s

import os
import lit.formats

config.name = "MLIR-Python"
config.test_format = lit.formats.ShTest(execute_external=False)

# File extensions lit will pick up as tests.
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.python_mlir_obj_root, "test")

# Tool substitutions.
tool_dirs = [config.python_mlir_tools_dir, config.llvm_tools_dir]
tools = ["python-mlir-opt", "FileCheck"]

import lit.llvm
lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config
llvm_config.add_tool_substitutions(tools, tool_dirs)
