## Business Card Raytracer

Inspired by this [post](http://fabiensanglard.net/rayTracing_back_of_business_card/index.php), I wanted to explore porting the code from C++ to a variety of other languages, benchmarking them, and documenting the process.

### Benchmarks

| Language            | SLoC | Time   | Comparison | Compile Flags | Compiler / Runtime Version                |
|---------------------|------|--------|------------|---------------|-------------------------------------------|
| C++ (Original Post) | 35   | 11.845 | 1.0        | clang++ -O3   | Apple LLVM version 8.1.0 (clang-802.0.42) |
| CPython             |      |        |            |               | CPython 3.6.1                             |
| PyPy                |      |        |            |               |                                           |
| Cython              |      |        |            |               |                                           |
| Nim                 |      |        |            |               |                                           |
| Crystal             |      |        |            |               |                                           |
