// TCV_cpp_libTorch.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include<torch/script.h>
#include<torch/torch.h>
#include<torch/csrc/cuda/device_set.h>

#include<opencv2/core.hpp>
void checkVersions()
{    
    std::cout << "LibTorch-"<< std::endl;
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "DeviceCount: " << torch::cuda::device_count() << std::endl;
    std::cout << "CUDNN available: " << torch::cuda::cudnn_is_available() << std::endl;
    std::cout << "LibTorch Version: " << TORCH_VERSION_MAJOR << "."
        << TORCH_VERSION_MINOR << "."
        << TORCH_VERSION_PATCH << std::endl;

    std::cout << "OpenCV version : " << CV_VERSION << std::endl;   
     

    //std::cout << torch::show_config() << std::endl;
    //torch::Tensor a = torch::ones({ 2, 2 }).to(torch::kCUDA);
    //torch::Tensor b = torch::randn({ 2, 2 }).to(torch::kCUDA);
    //torch::Tensor c = a + b;
    //std::cout << c << std::endl;

}
class TCV_test 
{
private:
    int value = 0;
    torch::jit::Module network;
    torch::DeviceType device_type;

    // 
    int _real_net_width;
    int _real_net_height;

    //Original frame size

    int _ori_height = 960;
    int _ori_width = 1280;

    // Model thresholds

    float _score_thre = 0.5f; //confidence
    float _iou_thre = 0.5f; //IOU

    // Detection results
    std::vector<torch::Tensor> dets_vec;

    // Model name
    std::string _modelName;

    // Tensor holding the predicted locations
    torch::Tensor predLoc;

    // Structure of detected objects
    struct Object {
        float left;
        float top;
        float right;
        float bottom;
        float score;
        int classId;
    };
    // Colors for visualization
    std::vector<cv::Vec3b> colors;



public:
    // Constructor
    TCV_test(std::string model, int real_net_width = 640, int real_net_height = 640, int frame_h = 960, int frame_w = 1280, float conf= 0.6, float iou= 0.6, int device=0)
    {
        int dev = device;
        SetDevice(dev);
        _score_thre = conf;
        _iou_thre = iou;
        _real_net_height = real_net_height;
        _real_net_width = real_net_width;
        _ori_height = frame_h;
        _ori_width = frame_w;

        colors = {
            cv::Vec3b(0, 0, 255),  // Red
            cv::Vec3b(0, 255, 0),  // Green
            cv::Vec3b(255, 0, 0),  // Blue
        };

    }

    void SetDevice(int deviceNum)
    {
        if (deviceNum == 0)
        {
            device_type = torch::kCPU; //use cpu
        
        }
        else {
            if (torch::cuda::is_available())
            {
                device_type = torch::kCUDA; // use gpu
            }
            else {
                device_type = torch::kCPU; // Fall back to CPU
                std::cout << "NO CUDA AVIALVE" << std::endl;
            }     
        }
        std::cout <<"Device set to: "  << (device_type == torch::kCPU ? "CPU" : "GPU") << std::endl;
    }

    int LoadModel(char* modelPath)
        try {
        network = torch::jit::load(modelPath, device_type);
        network.eval();

    }
    catch (const c10::Error& e)
    {
        std::cout << "Model reading failed .. " << std::endl;
    }
    
};


int main()
{

    std::cout << "Hello World!\n";
    checkVersions();
    TCV_test tcv("te", 640, 640, 960, 1280, 0.5, 0.6, 1);

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
