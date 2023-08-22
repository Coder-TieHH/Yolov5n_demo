
#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo.hpp"
#include <chrono> 

/////////////////////////////////



#define MODE_WIDTH 640
#define MODE_HEIGHT 640
// mean ,scale 分别表示归一化时需要减去的均值和除以的尺度参数，这里的数组中存储的是 R、G、B 三个通道的均值和尺度参数
std::array<float, 3> image_scale = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
std::array<float, 3> image_shift = {0.f, 0.f, 0.f};

const std::string model_path = "/home/rpdzkj/Desktop/yolov5n_demo/models/yolov5n_uint8.tmfile";
std::vector<Object> objects;
YOLO detector(model_path, MODE_WIDTH, MODE_HEIGHT, image_scale, image_shift);

int main(int argc, char **argv)
{
  cv::VideoCapture capture;
  cv::Mat frame;
  capture.open("v4l2src device=/dev/video0 ! video/x-raw, format=RGB, width=1920, height=1080, framerate=1000/30 ! videoconvert ! appsink",cv::CAP_GSTREAMER);
  
  if(capture.isOpened())
  {
    while (capture.read(frame))
    {


      detector.detect(frame, objects);
      for (auto &object : objects)
      {
        // box
        cv::Rect2f rect(object.box.x, object.box.y, object.box.width, object.box.height);

        const unsigned char *color = colors[object.label];

        cv::rectangle(frame, rect, cv::Scalar(color[0], color[1], color[2]), 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[object.label], object.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = rect.x;
        int y = rect.y - label_size.height - baseLine;
        if (y < 0)
          y = 0;
        if (x + label_size.width > frame.cols)
          x = frame.cols - label_size.width;
        cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(frame, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
      }
      // int width = frame.cols;
      // int  height = frame.rows;
      // float scaleFactor = static_cast<float>(MODE_WIDTH) / std::max(width, height);
      // int newWidth = static_cast<int>(width * scaleFactor);
      // int newHeight = static_cast<int>(height * scaleFactor);
      // cv::resize(frame, frame, cv::Size(newWidth, newHeight));
      // int top = (MODE_WIDTH - newHeight) / 2;
      // int bottom = MODE_WIDTH - newHeight - top;
      // int left = (MODE_WIDTH - newWidth) / 2;
      // int right = MODE_WIDTH - newWidth - left;
      // cv::Mat paddedImage;
      // cv::copyMakeBorder(frame, paddedImage, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

      cv::imshow("YOLOV5n", frame);
      cv::waitKey(30);
    }
  }
  return 0;
}
