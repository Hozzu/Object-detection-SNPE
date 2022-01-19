#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>
#include <vector>

#include <SNPE/SNPE.hpp>
#include <DlSystem/TensorShape.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <DlSystem/ITensorFactory.hpp>

#include <opencv2/opencv.hpp>
#include <fastcv/fastcv.h>
#include <qcarcam_client.h>

static zdl::SNPE::SNPE * snpe;

static std::string labels[200]; // label id must be less than 200
static int label_num = 0;

static unsigned long average_inference_time = 0;
static unsigned int num_inference = 0;

static int input_height;
static int input_width;
static int input_channel;

std::unique_ptr<zdl::DlSystem::ITensor> input_tensor;

// Callback function called when the camera frame is refreshed
void qcarcam_event_handler(qcarcam_input_desc_t input_id, unsigned char* buf_ptr, size_t buf_len){
    // Get the camera info
    unsigned int queryNumInputs = 0, queryFilled = 0;
    qcarcam_input_t * pInputs;

    if(qcarcam_query_inputs(NULL, 0, &queryNumInputs) != QCARCAM_RET_OK || queryNumInputs == 0){
        std::cout << "ERROR: The camera is not found.\n";
        exit(-1);
    }

    pInputs = (qcarcam_input_t *)calloc(queryNumInputs, sizeof(*pInputs));       
    if(!pInputs){
        std::cout << "ERROR: Failed to calloc\n";
        exit(-1);
    }

    if(qcarcam_query_inputs(pInputs, queryNumInputs, &queryFilled) != QCARCAM_RET_OK || queryFilled != queryNumInputs){
        std::cout << "ERROR: Failed to get the camera info\n";
        exit(-1);
    }

    int camera_height = pInputs[input_id].res[0].height;
    int camera_width = pInputs[input_id].res[0].width;

    free(pInputs);

    // Change color format from uyuv to rgb
    uint8_t * uv = (uint8_t *)fcvMemAlloc(camera_width * camera_height, 16);
    uint8_t * y = (uint8_t *)fcvMemAlloc(camera_width * camera_height, 16);
    if(uv == NULL || y == NULL){
        std::cout << "ERROR: Failed to fcvMemAlloc\n";
        exit(-1);
    }

    uint8_t * rgb_buf_ptr = new unsigned char[camera_height * camera_width * 3];
    if(rgb_buf_ptr == NULL){
        std::cout << "ERROR: Failed memory allocation\n";
        exit(-1);
    }

    fcvDeinterleaveu8(buf_ptr, camera_width, camera_height, camera_width * 2, (uint8_t *)uv, camera_width, (uint8_t *)y, camera_width);
    fcvColorYCbCr422PseudoPlanarToRGB888u8((uint8_t *)y, (uint8_t *)uv, camera_width, camera_height, camera_width, camera_width, (uint8_t *)rgb_buf_ptr, camera_width * 3);

    // Resize image
    uint8_t * r_buf_ptr = new unsigned char[camera_height * camera_width];
    uint8_t * g_buf_ptr = new unsigned char[camera_height * camera_width];
    uint8_t * b_buf_ptr = new unsigned char[camera_height * camera_width];

    for(int i = 0; i < camera_height * camera_width; i++){
        r_buf_ptr[i] = rgb_buf_ptr[3 * i];
        g_buf_ptr[i] = rgb_buf_ptr[3 * i + 1];
        b_buf_ptr[i] = rgb_buf_ptr[3 * i + 2];
    }

    unsigned char * resize_img_ptr = new unsigned char[input_height * input_width * input_channel];
    unsigned char * r_resize_img_ptr = new unsigned char[input_height * input_width];
    unsigned char * g_resize_img_ptr = new unsigned char[input_height * input_width];
    unsigned char * b_resize_img_ptr = new unsigned char[input_height * input_width];
    fcvScaleu8(r_buf_ptr, camera_width, camera_height, camera_width, r_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);
    fcvScaleu8(g_buf_ptr, camera_width, camera_height, camera_width, g_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);
    fcvScaleu8(b_buf_ptr, camera_width, camera_height, camera_width, b_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);

    for(int i = 0; i < input_height * input_height; i++){
        resize_img_ptr[3 * i] = r_resize_img_ptr[i];
        resize_img_ptr[3 * i + 1] = g_resize_img_ptr[i];
        resize_img_ptr[3 * i + 2] = b_resize_img_ptr[i];
    }

    delete r_buf_ptr;
    delete g_buf_ptr;
    delete b_buf_ptr;
    delete r_resize_img_ptr;
    delete g_resize_img_ptr;
    delete b_resize_img_ptr;

    // Inference
    std::vector<float> f_resize_img_vec;

    for(int i = 0; i < input_height * input_width * input_channel; i++){
        double value = ((double)resize_img_ptr[i] - 127.5) / 127.5;
        f_resize_img_vec.push_back((float)value);
    }

    std::copy(f_resize_img_vec.begin(), f_resize_img_vec.end(), input_tensor->begin());

    zdl::DlSystem::TensorMap output_tensor_map;

    auto start = std::chrono::high_resolution_clock::now();
    if(snpe->execute(input_tensor.get(), output_tensor_map) != true){
        std::cout << "ERROR: Model execute failed\n";
        exit(-1);
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    average_inference_time = (average_inference_time * num_inference + std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()) / (num_inference + 1);
    num_inference++;

    // Output of inference
    std::vector<float> output_locations;
    std::vector<int> output_classes;
    std::vector<float> output_scores;
    int output_nums = 0;

    zdl::DlSystem::StringList output_tensor_names = output_tensor_map.getTensorNames();

    for(int i = 0; i < output_tensor_names.size(); i++){
        const char * name = output_tensor_names.at(i);
        zdl::DlSystem::ITensor * output_tensor = output_tensor_map.getTensor(name);

        for(auto it = output_tensor->begin(); it != output_tensor->end(); it++){                        
            float tensor_data = *it;

            if(strstr(name, "class")){
                output_classes.push_back((int)tensor_data);
                output_nums++;
            }
            else if(strstr(name, "score")){
                output_scores.push_back(tensor_data);
            }
            else if(strstr(name, "box")){
                output_locations.push_back(tensor_data);
                it++;
                tensor_data = *it;
                output_locations.push_back(tensor_data);
                it++;
                tensor_data = *it;
                output_locations.push_back(tensor_data);
                it++;
                tensor_data = *it;
                output_locations.push_back(tensor_data);
            }
        }
    }

    // Change color format from rgb to bgr
    for(int i = 0; i < camera_width * camera_height * 3; i += 3){
        unsigned char tmp = rgb_buf_ptr[i];
        rgb_buf_ptr[i] = rgb_buf_ptr[i + 2];
        rgb_buf_ptr[i + 2] = tmp;
    }

    // Draw rectangles
    cv::Mat cvimg(camera_height, camera_width, CV_8UC3, rgb_buf_ptr);

    for (int i = 0; i < output_nums; i++){
        //std::cout << i <<  ": , output_classes: " << output_classes[i] << ", output_scores: " << output_scores[i] << ", output_locations: [" << output_locations[i * 4] << "," << output_locations[i * 4 + 1] << "," << output_locations[i * 4 + 2] << ","<< output_locations[i * 4 + 3] << "]\n";

        float score =  output_scores[i];
        if(score < 0.5)
            continue;

        int ymin = output_locations[i * 4] * camera_height;
        int xmin = output_locations[i * 4 + 1] * camera_width;
        int ymax = output_locations[i * 4 + 2] * camera_height;
        int xmax = output_locations[i * 4 + 3] * camera_width;

        int id =  (int)(output_classes[i]) + 1;

        char str[100]; 
        sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
        cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
        cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 0, 255), 2);
    }

    memcpy(rgb_buf_ptr, cvimg.data, camera_width * camera_height * 3 * sizeof(unsigned char));

    // Change color format from bgr to uyuv
    fcvColorRGB888ToYCbCr422PseudoPlanaru8(rgb_buf_ptr, camera_width, camera_height, camera_width * 3, y, uv, camera_width, camera_width);
    fcvInterleaveu8(uv, y, camera_width, camera_height, camera_width, camera_width, buf_ptr, camera_width * 2);

    // Free memory
    delete rgb_buf_ptr;
    delete resize_img_ptr;
    fcvMemFree(uv);
    fcvMemFree(y);
}

bool run_qcarcam(zdl::SNPE::SNPE * snpe_arg, char * label_path, char * display_path){
    // Parse the label
    std::ifstream labelfile(label_path);
    if(!labelfile.is_open()){
        std::cout << "ERROR: Cannot open the label file.\n";
        return false;
    }

    while(true){
        label_num++;
        std::string line_string;

        if(std::getline(labelfile, line_string)){
            labels[label_num] = line_string;
        }
        else
            break;
    }

    labelfile.close();

    // Get the input tensor info
    snpe = snpe_arg;

    zdl::DlSystem::TensorShape input_shape = snpe->getInputDimensions();

    input_height = input_shape.getDimensions()[1];
    input_width = input_shape.getDimensions()[2];
    input_channel = input_shape.getDimensions()[3];

    // Allocate input tensor
    input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(input_shape);

    if(input_tensor == NULL){
        std::cout << "ERROR: Cannot allocate input tensor\n";
        return false;
    }

    // Run qcarcam
    if(qcarcam_client_start_preview(display_path, qcarcam_event_handler) != QCARCAM_RET_OK){
        std::cout << "ERROR: Cannot connect to the qcarcam. Please check the display setting file.\n";
        return false;
    }

    // Wait the exit
    std::cout << "\nPress ctrl+c to exit.\n\n";
    int secs = 0;
    while (true){
        sleep(10);
        secs += 10;
        std::cout << std::fixed;
        std::cout.precision(3);
        std::cout << "Average inference speed(0~" << secs << "s): " << 1000000.0 / average_inference_time << "fps\n";
        std::cout << "Average inference speed(0~" << secs << "s): " << average_inference_time << "us\n\n";
    }

    // Stop qcarcam
    if(qcarcam_client_stop_preview() != QCARCAM_RET_OK){
        std::cout << "ERROR: Cannot disconnect the qcarcam.\n";
        return false;
    }

    return true;
}
