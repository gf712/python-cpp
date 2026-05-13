// RUN: python-mlir-opt %s -verify-diagnostics -split-input-file

// Build{List,Tuple,Set} verifiers require requires_expansion.size()
// to equal elements.size(); BuildDict requires it to equal keys.size()
// (== values.size() via SameVariadicOperandSize).

module {
  func.func @list_size_mismatch(%a: !python.object) -> !python.object {
    // expected-error @+1 {{requires_expansion has 2 entries but op has 1 elements}}
    %0 = "python.build_list"(%a) {requires_expansion = array<i1: false, false>} : (!python.object) -> !python.object
    return %0 : !python.object
  }
}

// -----

module {
  func.func @tuple_size_mismatch(%a: !python.object, %b: !python.object) -> !python.object {
    // expected-error @+1 {{requires_expansion has 1 entries but op has 2 elements}}
    %0 = "python.build_tuple"(%a, %b) {requires_expansion = array<i1: false>} : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }
}

// -----

module {
  func.func @set_size_mismatch(%a: !python.object) -> !python.object {
    // expected-error @+1 {{requires_expansion has 0 entries but op has 1 elements}}
    %0 = "python.build_set"(%a) {requires_expansion = array<i1>} : (!python.object) -> !python.object
    return %0 : !python.object
  }
}

// -----

module {
  func.func @dict_size_mismatch(%k: !python.object, %v: !python.object) -> !python.object {
    // expected-error @+1 {{requires_expansion has 2 entries but op has 1 key/value pairs}}
    %0 = "python.build_dict"(%k, %v) {operandSegmentSizes = array<i32: 1, 1>, requires_expansion = array<i1: false, false>} : (!python.object, !python.object) -> !python.object
    return %0 : !python.object
  }
}
