#include <torch/script.h> // One-stop header.
#include <iostream>

void model(torch::jit::script::Module module, torch::Tensor intput) {
    try {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        // Run inference
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.slice(/*dim=*/1, 0, 5) << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "\nInference took " << elapsed.count() << " seconds" << std::endl;

    }
    catch (const c10::Error& e) {
        std::cerr << "error: " << e.what()<< "\n";
        return -1;
    }
    return 0;
}

int main() {
    torch::jit::script::Module module = torch::jit::load("script.pt");
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor input = torch::rand({1, 48, 128});

    model(module, input);


    return 0;
}
