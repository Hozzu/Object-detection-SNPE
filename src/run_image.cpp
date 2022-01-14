#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>

#include <SNPE/SNPE.hpp>
#include <DlSystem/TensorShape.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <DlSystem/ITensorFactory.hpp>

#include <opencv2/opencv.hpp>
#include <fastcv/fastcv.h>
#include <dirent.h>
#include <jpeglib.h>
#include <json-c/json.h>

static std::string labels[200]; // label id must be less than 200
static int label_num = 0;

static unsigned long average_inference_time = 0;
static unsigned int num_inference = 0;

bool run_image(zdl::SNPE::SNPE * snpe, char * label_path, char * directory_path, char * result_path){
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

    // Create json object
    json_object * json_result = json_object_new_object();
    json_object * json_categories = json_object_new_array();
    json_object * json_images = json_object_new_array();
    json_object * json_annotations = json_object_new_array();

    // Write json categories from label
    for(int i = 1; i <= label_num; i++){
        if(labels[i] == "???" || labels[i] == "")
            continue;

        json_object * json_category = json_object_new_object();

        json_object_object_add(json_category, "id", json_object_new_int(i));
        json_object_object_add(json_category, "name", json_object_new_string(labels[i].c_str()));

        json_object_array_add(json_categories, json_category);
    }

    // Get the input tensor info
    zdl::DlSystem::TensorShape input_shape = snpe->getInputDimensions();

    int input_height = input_shape.getDimensions()[1];
    int input_width = input_shape.getDimensions()[2];
    int input_channel = input_shape.getDimensions()[3];

    // Allocate input tensor
    std::unique_ptr<zdl::DlSystem::ITensor> input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(input_shape);

    if(input_tensor == NULL){
        std::cout << "ERROR: Cannot allocate input tensor\n";
        return false;
    }

    // Find images in the directory
    DIR * dir;
    struct dirent * ent;

    dir = opendir(directory_path);
    if(dir == NULL){
        std::cout << "ERROR: Cannot open the directory.\n";
        return false;
    }

    char current_dir[500];
    getcwd(current_dir, 500);
    chdir(directory_path);

    int image_id = 0;
    while((ent = readdir(dir)) != NULL){
        char * filename = ent->d_name;
        if(strstr(filename, ".jpg")){
            //std::cout << "Detecting " << filename << "..\n";
            image_id++;

            // Decode the image            
            struct jpeg_decompress_struct cinfo;
            struct jpeg_error_mgr jerr;

            cinfo.err = jpeg_std_error(&jerr);
            jpeg_create_decompress(&cinfo);

            FILE * fp = fopen(filename, "rb");
            if(fp == NULL) {
                std::cout << "ERROR: Cannot open the image: " << filename << std::endl;
                return false;
            }
            jpeg_stdio_src(&cinfo, fp);

            jpeg_read_header(&cinfo, TRUE);

            cinfo.out_color_space = JCS_RGB;
            cinfo.output_components = 3;

            jpeg_start_decompress(&cinfo);

            // Get the image data
            int img_height = cinfo.output_height;
            int img_width = cinfo.output_width;
            int row_stride = cinfo.output_width * 3;

            unsigned char * rgb_buf_ptr = new unsigned char[img_height * img_width * 3];
            while(cinfo.output_scanline < cinfo.output_height){
                unsigned char * rowptr = (unsigned char *)rgb_buf_ptr + row_stride * cinfo.output_scanline; 
                jpeg_read_scanlines(&cinfo, &rowptr, 1);
            }

            // Write json images
            json_object * json_image = json_object_new_object();

            json_object_object_add(json_image, "id", json_object_new_int(image_id));
            json_object_object_add(json_image, "file_name", json_object_new_string(filename));
            json_object_object_add(json_image, "width", json_object_new_int(img_width));
            json_object_object_add(json_image, "height", json_object_new_int(img_height));

            json_object_array_add(json_images, json_image);

            // Resize image
            uint8_t * r_buf_ptr = new unsigned char[img_height * img_width];
            uint8_t * g_buf_ptr = new unsigned char[img_height * img_width];
            uint8_t * b_buf_ptr = new unsigned char[img_height * img_width];

            for(int i = 0; i < img_height * img_width; i++){
                r_buf_ptr[i] = rgb_buf_ptr[3 * i];
                g_buf_ptr[i] = rgb_buf_ptr[3 * i + 1];
                b_buf_ptr[i] = rgb_buf_ptr[3 * i + 2];
            }

            unsigned char * resize_img_ptr = new unsigned char[input_height * input_width * input_channel];
            unsigned char * r_resize_img_ptr = new unsigned char[input_height * input_width];
            unsigned char * g_resize_img_ptr = new unsigned char[input_height * input_width];
            unsigned char * b_resize_img_ptr = new unsigned char[input_height * input_width];
            fcvScaleu8(r_buf_ptr, img_width, img_height, img_width, r_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);
            fcvScaleu8(g_buf_ptr, img_width, img_height, img_width, g_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);
            fcvScaleu8(b_buf_ptr, img_width, img_height, img_width, b_resize_img_ptr, input_width, input_height, input_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);

            for(int i = 0; i < input_height * input_height; i++){
                resize_img_ptr[3 * i] = r_resize_img_ptr[i];
                resize_img_ptr[3 * i + 1] = g_resize_img_ptr[i];
                resize_img_ptr[3 * i + 2] = b_resize_img_ptr[i];
            }

            delete r_buf_ptr, g_buf_ptr, b_buf_ptr, r_resize_img_ptr, g_resize_img_ptr, b_resize_img_ptr;

            // Inference
            std::vector<float> f_resize_img_vec;

            for(int i = 0; i < input_height * input_width * input_channel; i++){
                double value = ((double)resize_img_ptr[i] - 127.5) / 127.5;
                f_resize_img_vec.push_back((float)value);
            }

            zdl::DlSystem::TensorMap output_tensor_map;

            std::copy(f_resize_img_vec.begin(), f_resize_img_vec.end(), input_tensor->begin());

            auto start = std::chrono::high_resolution_clock::now();
            if(snpe->execute(input_tensor.get(), output_tensor_map) != true){
                std::cout << "ERROR: Model execute failed\n";
                return false;
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

            for(int i = 0; i < output_nums; i++){
                //std::cout << i <<  ": , output_classes: " << output_classes[i] << ", output_scores: " << output_scores[i] << ", output_locations: [" << output_locations[i * 4] << "," << output_locations[i * 4 + 1] << "," << output_locations[i * 4 + 2] << ","<< output_locations[i * 4 + 3] << "]\n";

                float score =  output_scores[i];
                if(score < 0.3)
                    continue;

                float ymin = output_locations[i * 4] * img_height;
                float xmin = output_locations[i * 4 + 1] * img_width;
                float ymax = output_locations[i * 4 + 2] * img_height;
                float xmax = output_locations[i * 4 + 3] * img_width;
                float width = xmax - xmin;
                float height = ymax - ymin;

                int id =  (int)(output_classes[i]) + 1;

                // Write json annotations
                json_object * json_annotation = json_object_new_object();

                json_object * json_bbox = json_object_new_array();

                json_object_array_add(json_bbox, json_object_new_double(xmin));
                json_object_array_add(json_bbox, json_object_new_double(ymin));
                json_object_array_add(json_bbox, json_object_new_double(width));
                json_object_array_add(json_bbox, json_object_new_double(height));

                json_object_object_add(json_annotation, "image_id", json_object_new_int(image_id));
                json_object_object_add(json_annotation, "bbox", json_bbox);
                json_object_object_add(json_annotation, "category_id", json_object_new_int(id));
                json_object_object_add(json_annotation, "score", json_object_new_double(score));

                json_object_array_add(json_annotations, json_annotation);

                // Display image
                /* TO DO */
            }

            // Free memory
            delete rgb_buf_ptr;
            delete resize_img_ptr;

            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);

            fclose(fp);
        }
    }

    std::cout << std::fixed;
    std::cout.precision(3);
    std::cout << "\nAverage inference speed(" << image_id << " images): " << 1000000.0 / average_inference_time << "fps\n";
    std::cout << "Average inference speed(" << image_id << " images): " << average_inference_time << "us\n\n";

    // Write json file
    chdir(current_dir);

    json_object_object_add(json_result, "categories", json_categories);
    json_object_object_add(json_result, "images", json_images);
    json_object_object_add(json_result, "annotations", json_annotations);

    json_object_to_file(result_path, json_result);

    json_object_put(json_result);

    closedir(dir);

    return true;
}