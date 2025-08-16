# PaddleOCR C++ on Windows (CPU)

This project demonstrates how to run PaddleOCR using C++ on a Windows machine with CPU. The instructions below guide you through building and running the OCR system.

## Prerequisites

Before you begin, ensure that the following dependencies are installed:

- **Paddle Inference**: The PaddleOCR inference engine.
- **OpenCV**: Required for image processing.
- **ONNX Runtime**: Required for running ONNX models.

## ⚠️ Important: update config.txt paths

Before building, open the config file and set paths to your local model folders and label list:

File: projects\cpp\PaddleOCR\deploy\cpp_infer\tools\config.txt

Update these entries (examples shown—use your actual drive/paths):

# --- paths (EDIT THESE) ---
det_model_dir = F:/projects/cpp/PaddleOCR/deploy/cpp_infer/models/en_PP-OCRv3_det_infer/
rec_model_dir = F:/projects/cpp/PaddleOCR/deploy/cpp_infer/models/en_PP-OCRv3_rec_infer/
char_list_file = F:/projects/cpp/PaddleOCR/deploy/cpp_infer/models/en_PP-OCRv3_rec_infer/label_list.txt
# optional if you use angle classifier:
cls_model_dir = ./   # or leave as-is if not used (use_angle_cls = 0)

# --- runtime (typical CPU settings) ---
use_gpu = 0
use_mkldnn = 1
cpu_math_library_num_threads = 10


Tip: Forward slashes (D:/...) avoid escaping issues in configs. Make sure the models exist under .../cpp_infer/models/ or adjust the paths accordingly.

## Build Steps

1. **Clone this repository:**

    ```bash
    git clone https://github.com/waleed-222/Paddel-OCR-C-
    cd Paddel-OCR-C-
    ```

2. **Navigate to the project directory:**

    ```bash
    cd projects\cpp\PaddleOCR\deploy\cpp_infer\build
    ```

3. **Run `cmake` with the necessary paths:**

    ```bash
    cmake .. ^
      -DPADDLE_LIB="F:/projects/cpp/paddle_inference" ^
      -DOpenCV_DIR="F:/projects/cpp/opencv/build/x64/vc16/lib" ^
      -DONNXRUNTIME_DIR="D:/onnxruntime-win-x64-1.14.1"
    ```

    This configures the project with paths to Paddle Inference, OpenCV, and ONNX Runtime libraries.

4. **Build the project:**

    ```bash
    cmake --build . --config Release
    ```

    This will compile the project in Release mode.

## Running the Project

Once the project is built, you can run the OCR system.

1. **Navigate to the `Release` directory:**

    ```bash
    cd Release
    ```

2. **Run the OCR system:**

    ```bash
    .\ocr_system.exe <path_to_image>
    ```

    For example, to run OCR on an image located at `F:\projects\cpp\PaddleOCR\deploy\cpp_infer\imgs\new.png`, run:

    ```bash
    .\ocr_system.exe F:\projects\cpp\PaddleOCR\deploy\cpp_infer\imgs\new.png
    ```

    This will display the following output:

    ```
    =======Paddle OCR inference config======
    char_list_file : F:/projects/cpp/PaddleOCR/deploy/cpp_infer/models/en_PP-OCRv3_rec_infer/label_list.txt
    cls_model_dir : ./
    cls_thresh : 0.9
    cpu_math_library_num_threads : 10
    det_db_box_thresh : 0.5
    det_db_thresh : 0.3
    det_db_unclip_ratio : 1.6
    det_model_dir : F:/projects/cpp/PaddleOCR/deploy/cpp_infer/models/en_PP-OCRv3_det_infer/
    gpu_id : 0
    gpu_mem : 4000
    max_side_len : 960
    rec_model_dir : F:/projects/cpp/PaddleOCR/deploy/cpp_infer/models/en_PP-OCRv3_rec_infer/
    use_angle_cls : 0
    use_fp16 : 0
    use_gpu : 0
    use_mkldnn : 1
    use_polygon_score : 1
    use_tensorrt : 0
    visualize : 1
    =======End of Paddle OCR inference config======
    ```

3. **Output:**

    The system will perform OCR on the input image and display the results. For example:

    ```
    The detection visualized image saved in ./ocr_vis.png
    Number of detected boxes: 2
    Running recognition on box #0
    resize_img shape: 174 x 32
    Prediction shape: 1 22 97
    Prediction result: development. Score: 0.993475
    Running recognition on box #1
    resize_img shape: 679 x 32
    Prediction shape: 1 85 97
    Prediction result: A good project will save a lot of time before starting       Score: 0.951571
    A good project will save a lot of time before starting
    development.
    Cost  0.538542s
    ```

    Additionally, a visualized image (`ocr_vis.png`) will be saved, showing the original image with bounding boxes around the detected text.

## Troubleshooting

- Ensure all paths are correctly set in the `cmake` command.
- Verify the correct version of ONNX Runtime and OpenCV is being used.
- If the output is not as expected, check the `ocr_system.exe` logs for any errors or misconfigurations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Developed by

This repository was developed by [Waleed Ebrahem Mohamed](https://www.linkedin.com/in/waleed-ebrahem-46624a1b2/).

## Acknowledgements

This project is built upon the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) repository.
