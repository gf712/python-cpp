# Module support

Currently all modules written in C++ are statically linked to libpython. 

However, in the long term this will be replaced with a plugin architecture, so that C++ libraries can be directly imported into python and use a symbol table to load Python objects to the runtime PyModule symbol table.

The API to register Python objects in a shared library will be similar to the one provided by pybind/embind/PyTorch/...

```C++
DEFINE_MODULE(test, m) { 
    class_<NativeTest>(m, "Test")
        .def("foo", +[](){ std::cout << "Hello, world!\n"; })
        .def("echo", +[](PyObject* obj){ std::cout << obj->to_string() << '\n';})
        .def("test", &Test::test)
        ;
}
```
