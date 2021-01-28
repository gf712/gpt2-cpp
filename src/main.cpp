#include "config.hpp"
#include "GPT2Tokenizer.hpp"

#include <cmath>
#include <iostream>

#include <onnxruntime_cxx_api.h>
#include "cxxopts.hpp"

static constexpr size_t BATCH_SIZE = 1;

template <typename T>
T unwrap(std::optional<T>&& value, const std::string& error_msg) {
    if (value.has_value()) {
        return value.value();
    }
    else {
        throw std::runtime_error(error_msg);
    }
} 

template <typename T>
struct view {
    typename std::vector<T>::iterator _start;
    typename std::vector<T>::iterator _end;

    auto begin() const {
        return _start;
    }
    auto end() const {
        return _end;
    }
};

size_t next_token_prediction(const std::unique_ptr<Ort::Session>& session, 
                             std::vector<int64_t>& token_ids, // cannot be const otherwise will confuse templated CreateTensor
                             const size_t vocab_size) {
    Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    // model from https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2
    // input:
    //  - input1: long tensor (?, batch_size, sequence_length)
    std::vector<const char*> input_names = {"input1"};
    // CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len);
    std::vector<int64_t> input_ids_shape {BATCH_SIZE, BATCH_SIZE,  static_cast<int64_t>(token_ids.size())};
    std::vector<Ort::Value> input_values;
    input_values.push_back(
        Ort::Value::CreateTensor(info, token_ids.data(), token_ids.size(), input_ids_shape.data(), input_ids_shape.size()));
    size_t input_count = 1;
    // output:
    //  - output1: float tensor (?, batch_size, sequence_length, vocab_size)
    std::vector<const char*> output_names = {"output1"};
    std::vector<int64_t> prediction_scores_shape {BATCH_SIZE, BATCH_SIZE, static_cast<int64_t>(token_ids.size()), static_cast<int64_t>(vocab_size)};
    std::vector<float> prediction_scores(BATCH_SIZE * token_ids.size() * vocab_size);
    std::vector<Ort::Value> output_values;
    output_values.push_back(
        Ort::Value::CreateTensor(info, prediction_scores.data(), prediction_scores.size(), prediction_scores_shape.data(), prediction_scores_shape.size()));
    size_t output_count = 1;

    session->Run(
        Ort::RunOptions{},
        input_names.data(),
        input_values.data(),
        input_count,
        output_names.data(),
        output_values.data(),
        output_count
    );

    const auto new_word_prediction = view<float>{prediction_scores.begin() + (token_ids.size()-1) * vocab_size, 
                                                 prediction_scores.end()};

    auto softmax = [](view<float> vec) { 
        std::transform(vec.begin(), vec.end(), vec.begin(), [](const float& el){ return std::exp(el); });
        const float sum = std::accumulate(vec.begin(), vec.end(), 0.f);
        std::transform(vec.begin(), vec.end(), vec.begin(), [sum](const float& el){ return el/sum; });
    };

    softmax(new_word_prediction);

    const auto next_token_iter = std::max_element(new_word_prediction.begin(), new_word_prediction.end());
    return std::distance(new_word_prediction.begin(), next_token_iter);
}


int main(int argc, char *argv[]) {

    cxxopts::Options options("GPT2", "GPT2 implementation in C++ using Ort");

    options.add_options()
        ("t,text", "Initial text for GPT2", cxxopts::value<std::string>())
        ("n,number", "Number of new words to generate from initial text", cxxopts::value<size_t>()->default_value("1"))
        ("h,help", "Print usage")
    ;
    cxxopts::ParseResult result;
    
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::OptionException& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("text") == 0) {
        std::cout << "Expected text input!\n\n";
        std::cout << options.help() << std::endl;
        exit(0);
    }
    const std::string text = result["text"].as<std::string>();
    const size_t generate = result["number"].as<size_t>();

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    auto session = std::make_unique<Ort::Session>(env, model_file.data(), session_options);

    auto tokenizer = unwrap(GPT2Tokenizer::load(vocab_file, merges_file), "Error initialising GPT2 tokenizer\n");

    auto token_ids = tokenizer.encode(text);

    for (size_t i = 0; i < generate; ++i) {
        token_ids.push_back(
            next_token_prediction(session, token_ids, tokenizer.vocab_size()));
    }    

    std::cout << "Prediction: \"" << tokenizer.decode(token_ids) << '\"' << '\n';
}
