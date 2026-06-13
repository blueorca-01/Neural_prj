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

    net->layer_sizes = malloc((num_layers-1) * sizeof(int));

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

void print_network(const Network* net)
{
    printf("====================================\n");
    printf("Network structure\n");
    printf("====================================\n");

    printf("num_layers: %d\n", net->num_layers);

    printf("layer_sizes: ");
    for (int layer = 0; layer < net->num_layers; layer++) {
        printf("%d ", net->layer_sizes[layer]);
    }
    printf("\n\n");

    printf("====================================\n");
    printf("Weights and biases\n");
    printf("====================================\n");

    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int input_size = net->layer_sizes[layer];
        int output_size = net->layer_sizes[layer + 1];

        printf("Layer connection %d -> %d\n", layer, layer + 1);

        for (int out = 0; out < output_size; out++) {
            printf("  Output neuron %d\n", out);

            for (int in = 0; in < input_size; in++) {
                printf("    weights[%d][%d][%d] = %f\n",
                       layer,
                       out,
                       in,
                       net->weights[layer][out][in]);
            }

            printf("    biases[%d][%d] = %f\n",
                   layer,
                   out,
                   net->biases[layer][out]);
        }

        printf("\n");
    }

    printf("====================================\n");
    printf("Activations and deltas\n");
    printf("====================================\n");

    for (int layer = 0; layer < net->num_layers; layer++) {
        printf("Layer %d\n", layer);

        for (int neuron = 0; neuron < net->layer_sizes[layer]; neuron++) {
            printf("  activations[%d][%d] = %f, deltas[%d][%d] = %f\n",
                   layer,
                   neuron,
                   net->activations[layer][neuron],
                   layer,
                   neuron,
                   net->deltas[layer][neuron]);
        }

        printf("\n");
    }
}

void free_network(Network* net) {
    
    for(int layer = 0; layer < net->num_layers - 1; layer++) {
        for(int out = 0; out < net->layer_sizes[layer + 1]; out++) {
            free(net->weights[layer][out]);
        }
        free(net->weights[layer]);
        free(net->biases[layer]);
    }

    for(int layer = 0; layer < net->num_layers; layer++) {
        free(net->activations[layer]);
        free(net->deltas[layer]);
    }

    free(net->weights);
    free(net->biases);
    free(net->activations);
    free(net->deltas);
    free(net->layer_sizes);

    net->weights = NULL;
    net->biases = NULL;
    net->activations = NULL;
    net->deltas = NULL;
    net->layer_sizes = NULL;
    net->num_layers = 0;
}

float ReLU(float x) {
    float y = 0.0f;
    if (x < 0) {
        y = 0.0f;
    }
    else {
        y = x;
    }
    return y;
}

void forward(Network* net, const float* first_input, int input_layer_size) {

    printf("forward input = %f\n", first_input[0]);

    if (input_layer_size != net->layer_sizes[0]) {
        fprintf(stderr, "오류: 입력 크기가 입력층 뉴런 수와 다릅니다.\n");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < input_layer_size; i++) {
        net->activations[0][i] = first_input[i];
    }    
    printf("activation[0][0] = %f\n", net->activations[0][0]);

    for (int layer = 0; layer < ((net->num_layers) - 1); layer++) {
        for (int out = 0; out < (net->layer_sizes[layer + 1]); out++) {
            float sum_weight_total = 0.0f;
            for (int in = 0; in < (net->layer_sizes[layer]); in ++) {
                sum_weight_total += net->weights[layer][out][in] * net->activations[layer][in];
            }
            sum_weight_total += net->biases[layer][out];
            net->activations[layer + 1][out] = ReLU(sum_weight_total);
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
    printf("LATEST CODE\n");
    srand(1);

    float input[] = {0.5f};
    int input_layer_size = sizeof(input) / sizeof(input[0]);
    int layer_sizes[] = {input_layer_size, 4, 4, 2};
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

    Network net = {0};

    create_network(&net, num_layers, layer_sizes);
    forward(&net, input, input_layer_size);
    print_network(&net);
    free_network(&net);
    return 0;
}