//
// Created by yshhuang on 2020-02-12.
//

#include "jni.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <torch/script.h> // One-stop header.

#include <memory>
#include "com_quickcan_image_watermarkdetection_component_WatermarkDetector.h"


JNIEXPORT jboolean JNICALL Java_com_quickcan_image_watermarkdetection_component_WatermarkDetector_ifContainsDouyinLogo
        (JNIEnv *env, jobject, jstring file) {
    const char *str;
    str = env->GetStringUTFChars(file, JNI_FALSE);
    if (str == NULL) {
        return false; /* OutOfMemoryError already thrown */
    }
    std::cout << str << std::endl;

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("/Volumes/develop/code-repository/practice-code/c++/pytorch/traced_resnet_model.pt");
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return JNI_FALSE;
    }

    std::cout << "ok\n";

    std::vector <torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

// Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    output.s
    return true;
};
