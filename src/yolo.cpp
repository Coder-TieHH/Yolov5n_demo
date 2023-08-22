/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include "yolo.hpp"

#include "timer.hpp"

//#define DEBUG
#define RUNTIME_DEVICE "TIMVX"

const int layer_type = 0;
const int num_box = 3;
const int num_anchor = 9;
const int num_classes = 80;

const float thresh = 0.5;
const float hier_thresh = 0.5;
const float nms_threshold = 0.7f;
const int relative = 1;

YOLO::YOLO(const std::string& model, const int& w, const int& h, const std::array<float, 3>& model_scale, const std::array<float, 3>& model_bias)
{
    this->scale = model_scale;
    this->bias = model_bias;
    this->width = w;
    this->height = h;
    this->init_done = false;


    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());


    /* create VeriSilicon TIM-VX backend */
    this->context = create_context("timvx", 1);
    set_context_device(this->context, RUNTIME_DEVICE, nullptr, 0);

    this->graph = create_graph(this->context, "tengine", model.c_str());
    if (nullptr == this->graph)
    {
        fprintf(stderr, "Load Tengine model failed.\n");
    }

    /* get the quantizaiont params of input/output tensors */
    // get_graph_output_node_number(this->graph) 是一个函数调用，用于获取当前模型的输出节点数目。
    // 接着，代码通过调用 std::vector 对象的 resize() 函数对输出数据的缓存空间进行设置。
    // this->output_float.resize(branch_count) 表示将 output_float 的长度设置为 branch_count，
    // 即该向量可以存储 branch_count 个浮点数。this->out_scale.resize(branch_count) 和
    // this->out_zp.resize(branch_count) 分别表示将 out_scale 和 out_zp 的长度也设置为 branch_count，即它们两个可以分别存储 branch_count 个浮点数和整数。
    const auto branch_count = get_graph_output_node_number(this->graph);

    this->output_float.resize(branch_count);
    this->out_scale.resize(branch_count);
    this->out_zp.resize(branch_count);

    tensor_t input_tensor = get_graph_input_tensor(this->graph, 0, 0);
    if (nullptr == input_tensor)
    {
        fprintf(stderr, "Get input tensor failed.\n");
    }

    get_tensor_quant_param(input_tensor, &this->in_scale, &this->in_zp, 1);

    for (int i = 0; i < branch_count; i++)
    {
        tensor_t output_tensor = get_graph_output_tensor(this->graph, i, 0);
        if (nullptr == output_tensor)
        {
            fprintf(stderr, "Get model output tensor(%d/%d) failed.\n", i, branch_count);
        }

        get_tensor_quant_param(output_tensor, &this->out_scale[i], &this->out_zp[i], 1);
    }
}

YOLO::~YOLO()
{
    postrun_graph(this->graph);
    destroy_graph(this->graph);
    destroy_context(this->context);
}


// 就相当于get_input_data_focus_uint8()有focus就是5.0版本
int YOLO::detect(const cv::Mat& image, std::vector<Object>& objects)
{
    if (nullptr == this->graph)
    {
        fprintf(stderr, "Graph was not ready.\n");
        return -1;
    }

    /* initial the graph and prerun graph */
    if (!this->init_done)
    {
        int ret = this->init();
        if (0 != ret)
        {
            fprintf(stderr, "Init graph failed.\n");
            return -1;
        }
        this->init_done = true;
#ifdef DEBUG
        dump_graph(this->graph);
#endif
    }

    /* prepare process, letterbox */
    Timer prepare_timer;

    cv::Mat sample = image;
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    int letterbox_rows = this->height;
    int letterbox_cols = this->width;
    if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / img.rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    img.convertTo(img, CV_32FC3);

    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_32FC3, cv::Scalar(0, 0, 0));
    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    img_new.convertTo(img_new, CV_32FC3);

    //    cv::imwrite("process_img.jpg", img_new);

    float* img_data = (float*)img_new.data;
    // std::vector<float> input_temp(3 * letterbox_cols * letterbox_rows);
    uint8_t* input_data = this->input_uint8.data();

  /* nhwc to nchw */
    for (int h = 0; h < letterbox_rows; h++)
    {
        for (int w = 0; w < letterbox_cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * letterbox_cols * 3 + w * 3 + c;
                int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                input_data[out_index] = round((img_data[in_index] - this->bias[c]) * this->scale[c] / this->in_scale + (float)this->in_zp);

            }
        }
    }

    float prepare_cost = (float)prepare_timer.Cost();
    fprintf(stdout, "Prepare cost   %.2fms.\n", prepare_cost);
    // fouce结束

    /* network inference */
    Timer model_timer;
    int ret = run_graph(this->graph, 1);

    float top_model_cost = (float)model_timer.Cost();
    fprintf(stdout, "Run graph cost %.2fms.\n", top_model_cost);

    if (0 != ret)
    {
        fprintf(stderr, "Run graph failed.\n");
        return -1;
    }

    /* post process */
    Timer post_timer;
    this->run_post(image.cols, image.rows, objects);

    float post_cost = (float)post_timer.Cost();
    fprintf(stdout, "Post cost      %.2fms.\n", post_cost);

    return 0;
}

// 验证结束
int YOLO::init()
{
    Timer timer;
    this->input_uint8.resize(this->width * this->height * 3);

    tensor_t input_tensor = get_graph_input_tensor(this->graph, 0, 0);
    if (nullptr == input_tensor)
    {
        fprintf(stderr, "Get input tensor was failed.\n");
        return -1;
    }

    int dims[] = {1, 3, this->height, this->width};

    // int dims[] = {1, 12, int(this->height / 2), int(this->width / 2)};

    int ret = set_tensor_shape(input_tensor, dims, 4);
    if (0 != ret)
    {
        fprintf(stderr, "Set input tensor shape failed.\n");
        return -1;
    }

    ret = set_tensor_buffer(input_tensor, this->input_uint8.data(), this->input_uint8.size());
    if (0 != ret)
    {
        fprintf(stderr, "Set input tensor buffer failed.\n");
        return -1;
    }

    ret = prerun_graph(this->graph);
    if (0 != ret)
    {
        fprintf(stderr, "Pre-run graph failed.\n");
        return -1;
    }

    auto time_cost = (float)timer.Cost();
    fprintf(stdout, "Init cost %.2fms.\n", time_cost);

    return 0;
}


void YOLO::run_post(int image_width, int image_height, std::vector<Object>& boxes)
{
    pose_process(this->graph, image_width, image_height, this->width, this->height, boxes);
}
