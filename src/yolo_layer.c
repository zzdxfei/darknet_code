#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

// 初始化yolo层
// @param batch  网络的batch size
// @param w  网络输入图片的宽度
// @param h  网络输入图片的高度
// @param n  voc中为3
// @param total  voc中为9
// @param *mask  voc中为6,7,8
// @param classes  voc中为20
// 
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    // 属于某一类的概率，包围盒回归，包含目标的概率
    // n为anchors的个数
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    // 存放anchors
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        // 默认对应于n个anchor
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    // 输出的长度
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;

    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

// 输入输出长度重新计算，输出重新分配内存
void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

/**
 * @brief anchors是相对于网络输入大小的尺寸
 *
 * @param x
 * @param biases
 * @param n
 * @param index
 * @param i
 * @param j
 * @param lw  该层的输入的宽
 * @param lh  该层的输入的高
 * @param w  网络输入的宽
 * @param h  网络输入的高
 * @param stride  该层的输入一个通道的长度
 *
 * @return 
 */
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    // x中存放的是相对于anchor要做出的改变量
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

/**
 * 对于特定的anchor,在进行训练时，首先获得gt相对于anchor的改变量，作为回归目标。
 * 在进行预测时，网络输出该回归目标，再对anchor做相应的变化，得到预测的box
 * @param scale  delta放缩的比例
 */
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, 
        int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    // truth.x, truth.y在[0, 1]内，相对于原图和特征图的比例不变
    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    // truth.w * w得到gt相对于原始图片的真实宽度，所以biases也是相对于原始图片的
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

/*
 * @brief 
 * 
 * @param output  yolo层的输出
 * @param delta  损失数组
 * @param index  class在output中的位置索引
 * @param class  类别索引
 * @param classes  总的类别个数
 * @param stride  l.w * l.h
 * @param avg_cat  对ground truth类进行概率相加
 */
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        // 概率越大越好
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];  // 对ground truth类进行概率相加
        return;
    }
    
    // ground truth类的权重大
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    // 将net.input拷贝到l.output
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    // (------------- batch -----------)
    // ((--anchor--),--------------)
    // ((x, y, w, h, obj, class1, ... classn),--------------)
    // w, h通道没有使用LOGISTIC
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    // 损失 set to zero
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

    if(!net.train) return;
    float avg_iou = 0;  // iou的平均值
    float recall = 0;  // iou > 0.5的召回
    float recall75 = 0;  // iou > 0.75的召回
    float avg_cat = 0;  // 对每一个ground truth，其对应的anchor预测正确类的概率的均值
    float avg_obj = 0;  // 对于每一个ground truth，其对应的anchor包含目标的均值
    float avg_anyobj = 0;  // 每一个anchor包含目标的概率的均值
    int count = 0;
    int class_count = 0;
    // 损失函数置0
    *(l.cost) = 0;

    for (b = 0; b < l.batch; ++b) {  // 对每一张图片
        // ------------------ 第一阶段 ----------------------
        // 对是否含有目标的损失进行计算
        // --------------------------------------------------
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                // 对图片上的每一个像素点
                for (n = 0; n < l.n; ++n) {
                    // 对于每一个anchor进行处理
                    // 包围盒存放位置
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    // 获取该位置对应的box
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);

                    // 计算和ground truth中iou最大的ground truth的索引
                    float best_iou = 0;  // iou的最高得分
                    int best_t = 0;  // 最高得分的ground truth的索引
                    for(t = 0; t < l.max_boxes; ++t){
                        // 获得ground truth
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        // 遇到不足max_boxes时退出
                        if(!truth.x) break;
                        // 计算iou
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }

                    // 包含目标的概率
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];

                    // 包含目标的概率越大，
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    // 如果和ground truth的iou大于ignore_thresh，忽略其损失
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }

                    // 默认不会进行该部分 truth_thresh = 1
                    // 如果和ground truth的iou大，则其包含目标的概率要尽量大
                    if (best_iou > l.truth_thresh) {
                        // 包含目标的概率越大，delta越小
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        // ground truth的类别
                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];

                        // 可以对类别进行映射
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);

                        // 计算类别的损失
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        // 计算包围盒的损失
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        // truth的面积越大，delta扩大的比例越小 (2-truth.w*truth.h)
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }

        // ------------------ 第二阶段 ----------------------
        // 对包围盒和类别损失进行计算
        // --------------------------------------------------
        // 对ground truth中的每一个，查找预测该ground truth的anchor
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            // 要预测该ground truth的anchor的图像坐标
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;

            // 尝试所有的anchors，找到最适合的anchor的索引
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                // pred的x, y也是0
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            // 判断最佳的anchor索引是否在该层
            // 如果不在l.mask中，返回-1；否则返回l.mask的索引
            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){
                // 进行box和class损失的计算
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    // mag_array 计算数组的平方根
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    /********************************************************************** 
     * 出现-nan的情况，是因为最佳的anchor并不在本层
     *********************************************************************/
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n",
       net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_yolo_layer(const layer l, network net)
{
    // net.delta += l.delta
    // TODO(zzdxfei) ???
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

// 对所有的anchor，获取其预测目标的概率大于thresh的个数
int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    // 翻转数据的起始位置
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

/**
 * @brief 
 *
 * @param l  YOLO层
 * @param w  原始图片的高度
 * @param h  原始图片的宽度
 * @param netw  网络输入宽度
 * @param neth  网络输入高度
 * @param thresh
 * @param map
 * @param relative
 * @param dets
 *
 * @return 
 */
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        // anchor中心的坐标
        int row = i / l.w;
        int col = i % l.w;
        // 遍历每种anchor
        for(n = 0; n < l.n; ++n){
            // anchor包含目标的概率的存放位置
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];  // 包含目标的概率
            if(objectness <= thresh) continue;

            // 存储box
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            // 相对于网络输入的真实大小
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }

    // 在此处使用到了原始图片的尺寸，和预处理为逆过程
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

