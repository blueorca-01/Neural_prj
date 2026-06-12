#include <stdio.h>
#include <math.h>
#include <stdlib.h>


typedef struct {
    int num_layers;
    int* layer_sizes;

    float*** weights;
    float** activations;
    float** biases;
    float** deltas;
} Network;


/* 1차원 환경의 현재 상태 */
float position = 0.0f;

float random_weight_he(int in){
    float raw_weight = (float)rand()/(float)RAND_MAX;
    float L = sqrtf(6.0f / (float)in);
    float weight = -L + 2.0f * L * raw_weight;

    return weight;
}

void create_network(Network* net, int num_layers, const int* layer_sizes){
    net->num_layers = num_layers;

    for (int i = 0; i < num_layers; i++){
        net->layer_sizes[i] = layer_sizes[i];
    }

    net->weights = malloc((num_layers - 1) * sizeof(float**));
    net->biases = malloc((num_layers - 1) * sizeof(float*));
    net->activations = malloc((num_layers) * sizeof(float*));
    net->deltas = malloc((num_layers) * sizeof(float*));

    if(net->weights == NULL || net->biases == NULL || net->activations == NULL || net->deltas == NULL) {
        fprintf(stderr, "오류: Network 최상위 메모리 할당에 실패했습니다.\n");
        exit(EXIT_FAILURE);
    }

    for (int layer = 0; layer < num_layers - 1; layer++) {
        net->weights[layer] = malloc((layer_sizes[layer + 1]) * sizeof(float*));
        net->biases[layer] = malloc((layer_sizes[layer + 1]) * sizeof(float));

        if(net->weights[layer] == NULL || net->biases[layer] == NULL) {
            fprintf(stderr, "오류: layer %d의 weights 또는 biases 메모리 할당에 실패했습니다.\n", layer);
            exit(EXIT_FAILURE);
        }

        for (int out = 0; out < layer_sizes[layer + 1]; out++) {
            net->weights[layer][out] = malloc(layer_sizes[layer] * sizeof(float));
                if(net->weights[layer][out] == NULL) {
                    fprintf(stderr, "오류: weights[%d][%d] 메모리 할당에 실패했습니다.\n", layer, out);
                    exit(EXIT_FAILURE);
                }
        }
    }

    for(int layer = 0; layer < num_layers; layer++) { 
        net->activations[layer] = malloc((layer_sizes[layer]) * sizeof(float));
        net->deltas[layer] = malloc((layer_sizes[layer]) * sizeof(float));
        if(net->activations[layer] == NULL || net->deltas[layer] == NULL) {
            fprintf(stderr, "오류: layer %d의 activations 또는 deltas 메모리 할당에 실패했습니다.\n", layer);
            exit(EXIT_FAILURE);
        }
    }

    for (int layer = 0; layer < num_layers - 1; layer++) {
        for (int out = 0; out < layer_sizes[layer + 1]; out++) {
            for (int in = 0; in < layer_sizes[layer]; in++) {
                net->weights[layer][out][in] = random_weight_he(layer_sizes[layer]);
            }
            net->biases[layer][out] = 0.0f;
        }
    }

    for (int layer = 0; layer < num_layers; layer++) {
        for (int neuron = 0; neuron < layer_sizes[layer]; neuron++) {
            net->activations[layer][neuron] = 0.0f;
            net->deltas[layer][neuron] = 0.0f;
        }
    }
}

float environment(float move)
{
    position = position + 0.1f * move;

    return position;
}

int main(void)
{
    Network net = {0};

    int num_layers = 3;
    int layer_sizes[num_layers]={1, 4, 4};

    create_network(&net, num_layers, layer_sizes);
}