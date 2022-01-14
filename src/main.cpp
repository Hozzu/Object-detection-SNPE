#include <iostream>
#include <cstring>
#include <string>

#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/RuntimeList.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <DlSystem/TensorShape.hpp>


bool run_qcarcam(zdl::SNPE::SNPE * snpe_arg, char * label_path, char * display_path);
bool run_image(zdl::SNPE::SNPE * snpe, char * label_path, char * directory_path, char * result_path);

int main(int argc, char ** argv){
    //usage guide
    if(strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "--h") == 0){
        std::cout << "Usage: pkshin_detect camera [MODEL] [LABEL] [DISPLAY] [ACCELERATOR]\n";
        std::cout << "camera mode runs the object detection using qcarcam API.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[DISPLAY] is path of the file defining the display setting.\n\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        std::cout << "Usage: pkshin_detect image [MODEL] [LABEL] [IMG_DIR] [RESULT] [ACCELERATOR]\n";
        std::cout << "image mode runs the object detection with jpeg images.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[IMG_DIR] is path of the directory containing images.\n";
        std::cout << "[RESULT] is path of the result json file.\n\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        return true;
    }

    // Argument error checking
    if( (strcmp(argv[1], "camera") != 0 && strcmp(argv[1], "image") != 0) || (strcmp(argv[1], "camera") == 0 && argc < 5) || (strcmp(argv[1], "image") == 0 && argc < 6) ){
        std::cout << "ERROR: The first argument must be camera or image. camera mode requires at least 3 more arguments and image mode requires at least 4 more arguments\n\n";
        std::cout << "Usage: pkshin_detect camera [MODEL] [LABEL] [DISPLAY] [ACCELERATOR]\n";
        std::cout << "camera mode runs the object detection using qcarcam API.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[DISPLAY] is path of the file defining the display setting.\n\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        std::cout << "Usage: pkshin_detect image [MODEL] [LABEL] [IMG_DIR] [RESULT] [ACCELERATOR]\n";
        std::cout << "image mode runs the object detection with jpeg images.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[IMG_DIR] is path of the directory containing images.\n";
        std::cout << "[RESULT] is path of the result json file.\n\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        return false;
    }

    // Load the model
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(std::string(argv[2]));

    if(container == NULL){
        std::cout << "ERROR: Model load failed. Check the model name.\n";
        return false;
    }

    // Set the runtime
    zdl::DlSystem::Runtime_t runtime;
    zdl::DlSystem::RuntimeList runtime_list;

    if( (strcmp(argv[1], "camera") == 0 && argc == 5) || (strcmp(argv[1], "image") == 0 && argc == 6) || (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "CPU") == 0 || strcmp(argv[5], "cpu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "CPU") == 0 || strcmp(argv[6], "cpu") == 0)) ){
        std::cout << "INFO: Run with CPU only.\n";
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }
    else if( (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "GPU") == 0 || strcmp(argv[5], "gpu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "GPU") == 0 || strcmp(argv[6], "gpu") == 0)) ){
        if(zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)){
            std::cout << "INFO: Run with GPU runtime.\n";
            runtime = zdl::DlSystem::Runtime_t::GPU;
        }
        else{
            std::cout << "WARNING: Cannot create GPU runtime. Run with CPU only.\n";
            runtime = zdl::DlSystem::Runtime_t::CPU;
        }
    }
    else if( (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "NPU") == 0 || strcmp(argv[5], "npu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "NPU") == 0 || strcmp(argv[6], "npu") == 0)) ){
        if(zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)){
            std::cout << "INFO: Run with DSP runtime.\n";
            runtime = zdl::DlSystem::Runtime_t::DSP;
        }
        else{
            std::cout << "WARNING: Cannot create DSP runtime. Check whether the ADSP_LIBRARY_PATH is correct. Run with CPU only.\n";
            runtime = zdl::DlSystem::Runtime_t::CPU;
        }
    }
    else{
        std::cout << "WARNING: [ACCELERATOR] should be CPU, GPU, or NPU. Run with CPU only.\n";
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }

    if(runtime_list.empty()){
        runtime_list.add(runtime);
    }

    // Specify the output tensors
    zdl::DlSystem::StringList output_tensor_lists;
    output_tensor_lists.append("Postprocessor/BatchMultiClassNonMaxSuppression_classes");
    output_tensor_lists.append("Postprocessor/BatchMultiClassNonMaxSuppression_boxes");
    output_tensor_lists.append("Postprocessor/BatchMultiClassNonMaxSuppression_scores");

    // Build the interpreter
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setOutputTensors(output_tensor_lists).setRuntimeProcessorOrder(runtime_list).setUseUserSuppliedBuffers(false).setInitCacheMode(false).build();

    if(snpe == NULL){
        std::cout << "ERROR: Interpreter build failed. Check that the output node name is appropriate. Check the runtime compatibility of graph.\n";
        return false;
    }

    // Check input tensor
    zdl::DlSystem::StringList input_tensor_names = snpe->getInputTensorNames();

    for(int i = 0; i < input_tensor_names.size(); i++){
        std::cout << "INFO: Graph input " << i << ": " << input_tensor_names.at(i) << std::endl;
    }

    // Check output tensors
    zdl::DlSystem::StringList output_tensor_names = snpe->getOutputTensorNames();

    for(int i = 0; i < output_tensor_names.size(); i++){
        std::cout << "INFO: Graph output " << i << ": " << output_tensor_names.at(i) << std::endl;
    }

    // Call the appropriate functons for mode
    if(strcmp(argv[1], "camera") == 0){
        std::cout << "INFO: Running the object detection using qcarcam API.\n";
        return run_qcarcam(snpe.get(), argv[3], argv[4]);
    }
    else if(strcmp(argv[1], "image") == 0){
        std::cout << "INFO: Running the object detection with jpeg images.\n";
        return run_image(snpe.get(), argv[3], argv[4], argv[5]);
    }
}