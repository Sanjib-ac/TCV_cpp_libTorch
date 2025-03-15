// TCV_cpp_libTorch.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <filesystem>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/tls.hpp>
#include <vector>
#include <memory>
#include <string>

#include<torch/script.h>
#include<torch/torch.h>
#include<torch/csrc/cuda/device_set.h>

#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include "opencv2/core/core_c.h"
static void checkVersions()
{    
    std::cout << "LibTorch-"<< std::endl;
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "DeviceCount: " << torch::cuda::device_count() << std::endl;
    std::cout << "CUDNN available: " << torch::cuda::cudnn_is_available() << std::endl;
    std::cout << "LibTorch Version: " << TORCH_VERSION_MAJOR << "."
        << TORCH_VERSION_MINOR << "."
        << TORCH_VERSION_PATCH << std::endl;

    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    //std::cout << "OpenCV is using: " << cv::getThreadStrategyName() << std::endl;

   
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
    int device;
    std::vector<std::string> class_labels; // Classes

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
        int left;
        int top;
        int right;
        int bottom;
        float score;
        int classId;
    };

    //
    std::vector<std::shared_ptr<Object>> detections;

    // Colors for visualization
    std::vector<cv::Vec3b> colors;



public:
    // Constructor
    TCV_test(): _modelName(""), _real_net_width(1280), _real_net_height(960), _ori_height(960), _ori_width(1280), _score_thre(0.6), _iou_thre(0.6), device(0)
    {        
        SetDevice(device);
        //std::cout << "Constructor1";
        colors = {
            cv::Vec3b(0, 0, 255),  // Red
            cv::Vec3b(0, 255, 0),  // Green
            cv::Vec3b(255, 0, 0),  // Blue
        };

    }
    //Constructor 2
    TCV_test(std::string model, int real_net_height = 640, int real_net_width = 640, int frame_h = 960, int frame_w = 1280, float conf = 0.6, float iou = 0.6, int device = 0)
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
                //device_type = at::kCUDA;
                std::cout << "Setting device_type: " << device_type << std::endl;
            }
            else {
                //device_type = at::kCPU; 
                device_type = torch::kCPU; // Fall back to CPU
                std::cout << "NO CUDA AVIALVE" << std::endl;
            }     
        }
        std::cout <<"Device set to: "  << (device_type == torch::kCPU ? "CPU" : "GPU") << std::endl;
    }

    int LoadModel(const std::filesystem::path& modelPath)
    { 
        try
            {
            //std::cout << "Loading the model";
            

           // Ensure model path is valid
            if (modelPath.empty())
            {
                std::cerr << "Error MOdel path is empty!" << std::endl;
            }
            std::cout << "Model absolute path: " << std::filesystem::absolute(modelPath) << std::endl;
            network = torch::jit::load(modelPath.string());            
            network.to(device_type);
            network.eval();
            if (!network.find_method("forward"))
                {
                std::cerr << "Model loaded, but forward method is missing!" << std::endl;
                }
        
            std::cout << "Loaded the model successfully.";
            return 0;
            }
        catch (const c10::Error& e)
            {
            std::cout << "Model reading failed .. " << e.what() << std::endl;
            return -1;
            }
    }
    int loadClassLabels(const std::filesystem::path& class_names)
    {
        std::ifstream file(class_names);
        if (!file) return -1;
        class_labels.assign((std::istream_iterator<std::string>(file)), std::istream_iterator<std::string>());
        std::cout<<'\n' << class_labels << std::endl;
        return 0;

    }
    int loadClassLabels1(const std::filesystem::path& class_names)
    {
        std::ifstream file(class_names);
        if (!file.is_open())
        {
        std::cerr << "Error loading class name file";
        return -1;
        }
        std::string line;
        while(std::getline(file, line))
        {
            class_labels.push_back(line);
        }
        std::cout << class_labels << std::endl;
        return 0;
    }
    void nonMaxSuppression(torch::Tensor preds)
    {
        dets_vec.clear();


    }
    void postProcess(const torch::Tensor& detections, int img_width, int img_height)
    {
        //detections.to(device_type);
        this->detections.clear();

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        for (int i = 0; i < detections.size(0); i++) 
        {
            float conf = detections[i][4].item<float>();
            if (conf < _score_thre) continue;

            int center_x = static_cast<int>(detections[i][0].item<float>() * img_width);
            int center_y = static_cast<int>(detections[i][1].item<float>() * img_height);
            int width = static_cast<int>(detections[i][2].item<float>() * img_width);
            int height = static_cast<int>(detections[i][3].item<float>() * img_height);

            int left = center_x - width / 2;
            int top = center_y - height / 2;
            int right = center_x + width / 2;
            int bottom = center_y + height / 2;

            boxes.emplace_back(left, top, width, height);
            scores.push_back(conf);
            class_ids.push_back(detections[i][5].item<int>());
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, _score_thre, _iou_thre, indices);

        for (int idx : indices) 
        {
            auto obj = std::make_shared<Object>(Object{
                boxes[idx].x,
                boxes[idx].y,
                boxes[idx].x + boxes[idx].width,
                boxes[idx].y + boxes[idx].height,
                scores[idx],
                class_ids[idx]
                });

            this->detections.push_back(obj);
        }
        std::cout << detections << std::endl;
    }
    const std::vector<std::shared_ptr<Object>>& getDetections() const
    {
        return detections;
    }
    void detect(const cv::Mat& image)
    {
        if (image.empty()) {
            std::cerr << "Error: Input image is empty!" << std::endl;
            return;
        }
        cv::Mat img_resized;
        // Resize the image to match the input dimensions of the model
        cv::resize(image, img_resized, cv::Size(_real_net_width, _real_net_height));

        if (img_resized.empty()) {
            std::cerr << "Error: Resized image is empty!" << std::endl;
            return;
        }

        std::cout << "Input Frame Shape: " << img_resized.size() << std::endl;

        // Convert the color space from BGR to RGB (the model expects RGB input)
        cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);

        // Normalize the image to [0, 1]
        img_resized.convertTo(img_resized, CV_32FC3, 1.0f / 255.0f);

        if (!img_resized.isContinuous()) {
            img_resized = img_resized.clone();
        }
        /* 
        torch::Tensor tensor_img = torch::from_blob(img_resized.data, { _real_net_width, _real_net_height, 3 }, torch::kFloat);
            //.permute({ 0, 3, 1, 2 })
            //.to(device_type);
        // Permute dimensions to match the model's expected input [C, H, W] format
        tensor_img = tensor_img.permute({ 2, 0, 1 });
        //Add batch dimension
        tensor_img = tensor_img.unsqueeze(0);
        */
        
        torch::Tensor tensor_img =torch::from_blob(img_resized.data, { 1, 3, _real_net_height, _real_net_width}, torch::kFloat)
            //.permute({2, 0, 1})
            //.unsqueeze(0)
            .to(device_type);
        //tensor_img = tensor_img.to(at::kCUDA);
        std::cout << "Input Tensor Shape: " << tensor_img.sizes() << std::endl;

        if (img_resized.empty()) {
            std::cerr << "Error: Resized image is empty!" << std::endl;
            return;
        }

        //tensor_img = tensor_img.to(at::kCUDA);

        // Create a vector of IValue to pass the tensor to the model
        std::vector<torch::jit::IValue> inputs{tensor_img};

        //inputs.push_back(std::move(tensor_img));  // Add the input tensor to the vector
        // Enable inference mode for efficiency.
        //torch::InferenceMode guard(true);
        //network.to(at::kCUDA);
        if (!network.find_method("forward")) {
            std::cerr << "Error: Model is not loaded correctly!" << std::endl;
            return;
        }
        try 
        {
            torch::jit::IValue output = network.forward(inputs);
            //std::cout << "OUT:" << output << std::endl;

             // Handle the output based on its type
            if (output.isTensor()) {
                std::cout << "Predicted result: ";
                /// If the output is a tensor, convert it to a std::vector of floats
                torch::Tensor output_tensor = output.toTensor();

                // Convert the tensor to a list (std::vector in this case)
                //std::vector<float> output_list(output_tensor.data_ptr<float>(), output_tensor.data_ptr<float>() + output_tensor.numel());

                // Print the result (for example, first 5 elements)
                /*std::cout << "Predicted result: ";
                for (size_t i = 0; i < std::min(output_list.size(), size_t(5)); ++i) {
                    std::cout << output_list[i] << " ";
                }
                std::cout << std::endl;*/
            }
            else if (output.isTuple()) {
                // If output is a tuple
                auto output_tuple = output.toTuple();
                auto preds = output_tuple->elements()[0].toTensor();
               // std::cout << "Predictions tensor from tuple: " << preds << std::endl;
                postProcess(preds, image.cols, image.rows);
            }
            else {
                std::cerr << "Error: Output is neither a tensor nor a tuple!" << std::endl;
            }
            
            
        }
        catch (const std::runtime_error& e) {
            std::cerr << "Runtime error: " << e.what() << std::endl;
        }

        catch (const c10::Error& e) 
        {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            // Additional information about inputs and output
            //std::cerr << "Inputs size: " << inputs.size() << std::endl;            
        }
        
    }
    void CaptureCam(int index, int h = 736, int w = 1280)
    {
        cv::VideoCapture cap(index,cv::CAP_MSMF);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open the camera.\n";
            return;
        }
        // Get Frame size
        double frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // Display the frame size
        std::cout << "Before Frame width: " << frameWidth << std::endl;
        std::cout << "Before Frame height: " << frameHeight << std::endl;

        // Set the frame size using the `set` function
        cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);

        // Get the actual frame size to verify the change
        double NframeWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double NframeHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        std::cout << "After frame width: " << NframeWidth << std::endl;
        std::cout << "After frame height: " << NframeHeight << std::endl;

        cv::Mat frame;

        while (cap.read(frame)) 
        {
            std::cout << "Reading frame" << std::endl;
             
            detect(frame);  // Detect objects in the frame
            /*
            // Draw bounding boxes on the frame
            for (const auto& obj : detections) {
                cv::rectangle(frame, cv::Point(obj->left, obj->top), cv::Point(obj->right, obj->bottom), cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, class_labels[obj->classId], cv::Point(obj->left, obj->top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
            */
            // Display the resulting frame
            cv::imshow("YOLOv11 - Webcam Detection", frame);

            // Exit when 'q' is pressed
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        cap.release();  // Release the webcam
        cv::destroyAllWindows();  // Destroy all OpenCV windows


    }
    int Image()
    {
        // Path to the image you want to read
        std::string imagePath = "D:/VS/TCV_cpp_libTorch/model/test_img.jpg";

        // Read the image
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);  // Read image in color mode

        // Check if the image is loaded successfully
        if (image.empty()) {
            std::cerr << "Could not open or find the image at: " << imagePath << std::endl;
            return -1;  // Exit if the image is not loaded
        }
        detect(image);
        // Create a window to display the image
        cv::namedWindow("Image", cv::WINDOW_NORMAL);

        // Display the image in the created window
        cv::imshow("Image", image);

        // Wait indefinitely for any key press
        cv::waitKey(0);

        // Close the window when a key is pressed
        cv::destroyAllWindows();

        return 0;
    }
};


int main()
{
    //cv::setNumThreads(0);  // Disable OpenCV parallel threading
    //cv::ocl::setUseOpenCL(false);  // Disable OpenCL (optional)
    //torch::set_num_threads(1);
    //_putenv("OMP_NUM_THREADS=1");
    //_putenv("OPENCV_ENABLE_PLUGINS=0");
    //cv::setNumThreads(0);
    std::cout << "Hello World!\n";
    //cv::setNumThreads(0);
    checkVersions();
    std::cout << '\n';
    //TCV_test tcv;
    std::cout << '\n';
    TCV_test tcv2("te", 736, 1280, 736, 1280, 0.5, 0.6, 1);
    //tcv2.SetDevice(1);
    //std::filesystem::path modelPath = "D:/VS/TCV_cpp_libTorch/model/best.pt";
    std::filesystem::path modelPath = "D:/VS/TCV_cpp_libTorch/model/nws_736_1280_b1.torchscript";

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "Error: Model file not found at " << modelPath << std::endl;
        
    }
    else {
        std::cout << "Model file found, now loading..."<< std::endl;
        tcv2.LoadModel(modelPath);
    }
    std::filesystem::path classNamelPath = "./model/classes.names";
    //tcv2.loadClassLabels(classNamelPath);
    tcv2.CaptureCam(0);
    tcv2.Image();

    return 0;

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
