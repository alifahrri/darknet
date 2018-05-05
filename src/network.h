// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU
void pull_network_output(network *net);
#endif

void compare_networks(network *n1, network *n2, data d);
char *get_layer_string(LAYER_TYPE a);

float network_accuracy_multi(network *net, data d, int n);
int get_predicted_class_network(network *net);
void print_network(network *net);
int resize_network(network *net, int w, int h);
void calc_network_cost(network *net);

load_args get_base_args(network *net);
void load_network_into(network **net_, char *cfg, char *weights, int clear);
network *load_network(char *cfg, char *weights, int clear);
size_t get_current_batch(network *net);
void reset_network_state(network *net, int b);
void reset_rnn(network *net);
float get_current_rate(network *net);
// char *get_layer_string(LAYER_TYPE a);
void make_network_into(network *net, int n);
network *make_network(int n);
void forward_network(network *netp);
void update_network(network *netp);
void calc_network_cost(network *netp);
int get_predicted_class_network(network *net);
void backward_network(network *netp);
float train_network_datum(network *net);
float train_network_sgd(network *net, data d, int n);
float train_network(network *net, data d);
void set_temp_network(network *net, float t);
void set_batch_network(network *net, int b);
int resize_network(network *net, int w, int h);
layer get_network_detection_layer(network *net);
image get_network_image_layer(network *net, int i);
image get_network_image(network *net);
void visualize_network(network *net);
void top_predictions(network *net, int k, int *index);
float *network_predict(network *net, float *input);
int num_detections(network *net, float thresh);
detection *make_network_boxes(network *net, float thresh, int *num);
void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
float *network_predict_image(network *net, image im);
// int network_width(network *net){return net->w;}
// int network_height(network *net){return net->h;}
matrix network_predict_data_multi(network *net, data test, int n);
matrix network_predict_data(network *net, data test);
void print_network(network *net);
void compare_networks(network *n1, network *n2, data test);
matrix network_predict_data(network *net, data test);
void print_network(network *net);
void compare_networks(network *n1, network *n2, data test);
float network_accuracy(network *net, data d);
float *network_accuracies(network *net, data d, int n);
layer get_network_output_layer(network *net);
float network_accuracy_multi(network *net, data d, int n);
void free_network(network *net);
layer network_output_layer(network *net);
int network_inputs(network *net);
int network_outputs(network *net);
float *network_output(network *net);

#ifdef GPU

void forward_network_gpu(network *netp);
void backward_network_gpu(network *netp);
void update_network_gpu(network *netp);
void harmless_update_network_gpu(network *netp);
void *train_thread(void *ptr);
pthread_t train_network_in_thread(network *net, data d, float *err);
void merge_weights(layer l, layer base);
void scale_weights(layer l, float s);
void pull_weights(layer l);
void push_weights(layer l);
void distribute_weights(layer l, layer base);
void sync_layer(network **nets, int n, int j);
void *sync_layer_thread(void *ptr);
pthread_t sync_layer_in_thread(network **nets, int n, int j);
void sync_nets(network **nets, int n, int interval);
float train_networks(network **nets, int n, data d, int interval);
void pull_network_output(network *net);

#endif

#ifdef __cplusplus
}
#endif

#endif

