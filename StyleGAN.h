#include "tensor4.h"


struct Block
{
	t4::tensor4f noise_weight_1;
	t4::tensor4f noise_weight_2;
	t4::tensor4f conv_1_weight;
	t4::tensor4f conv_2_weight;
	t4::tensor4f bias_1;
	t4::tensor4f bias_2;
	t4::tensor2f style_1_weight;
	t4::tensor1f style_1_bias;
	t4::tensor2f style_2_weight;
	t4::tensor1f style_2_bias;
	t4::tensor4f to_rgb_weight;
	t4::tensor1f to_rgb_bias;
};

struct StyleGAN
{
	t4::tensor2f mapping_block_weight[8];
	t4::tensor1f mapping_block_bias[8];
	t4::tensor1f dlatent_avg;
	t4::tensor4f block_0_const;
	t4::tensor2f latents;
	Block block[9];
};


StyleGAN StyleGANLoad(const char* filename);

