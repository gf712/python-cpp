// RUN: python-mlir-opt %s -verify-diagnostics -split-input-file

// The NamedOp trait rejects an empty 'name' attribute on the
// Load{Name,Fast,Global,Deref} / Store... / Delete... ops. Pick a
// representative from each direction; the trait is shared so testing
// one of each (Load/Store/Delete) is sufficient signal.

module {
  func.func @bad_load() -> !python.object {
    // expected-error @+1 {{requires a non-empty 'name' attribute}}
    %0 = "python.load_fast"() {name = ""} : () -> !python.object
    return %0 : !python.object
  }
}

// -----

module {
  func.func @bad_store(%v: !python.object) -> !python.object {
    // expected-error @+1 {{requires a non-empty 'name' attribute}}
    "python.store_fast"(%v) {name = ""} : (!python.object) -> ()
    return %v : !python.object
  }
}

// -----

module {
  func.func @bad_delete() {
    // expected-error @+1 {{requires a non-empty 'name' attribute}}
    "python.delete_fast"() {name = ""} : () -> ()
    return
  }
}
