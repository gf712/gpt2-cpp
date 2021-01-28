GPT2
------

This is a simple repo with an implementation of the GPT2 tokenizer + execution of the GPT2 ONNX model provided [here](https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2).

## Installation
The cmake script will look for [onnxruntime](https://github.com/microsoft/onnxruntime) header files and dynamic library using the repo structure based from the $HOME directory. After compiling make sure that the dynamic library can be found by the runtime, ie. set `LD_LIBRARY_PATH` accordingly.

The other three dependencies are already included in this project:
  * [simdjson](https://github.com/simdjson/simdjson)
  * [ctre](https://github.com/hanickadot/compile-time-regular-expressions)
  * [cxxopts](https://github.com/jarro2783/cxxopts/)

Compiling the binary requires a C++17 compliant compiler.
Additionally the ONNX model is downloaded during the build process from the [ONNX Model Zoo](https://github.com/onnx/models) repo. The model is 634 MB large, so it may take a while to download it :)

The vocabulary and merges files are provided in this repository, but were originally obtained from the [transformers](https://github.com/huggingface/transformers) repo.

```bash
mkdir build && cd build
cmake ..
make
```

## Usage
```bash
cd build
./gpt2-generate -t "I've got a q" -n 5
Prediction: "I've got a qwerty knife!!"
```