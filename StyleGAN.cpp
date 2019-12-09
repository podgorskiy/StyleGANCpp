#include "StyleGAN.h"
#include "zfp.h"

namespace t4
{
	inline t4::model_dict decompress(const t4::model_dict md, int compression)
	{
		t4::model_dict cd;

		for (auto it = md.m_parameters.cbegin(); it != md.m_parameters.cend(); ++it)
		{
			auto entry = *it;
			model_dict::Entry e;
			printf("%s\n", entry.first.c_str());
			e = entry.second;
			zfp_type type = zfp_type_float;
			zfp_field* field = nullptr;
			std::vector<uint64_t> sshape;
			assert(e.compressed_size != 0);

			auto compressed_data = e.ptr;
			e.ptr.reset(new uchar[e.size]);

			int nel = 1;
			for (int i = 0; i < e.ndim; ++i)
			{
				nel *= e.shape[i];
			}

			if (e.ndim == 4 && e.shape[3] > 1 && e.shape[2] > 1)
			{
				field = zfp_field_2d(e.ptr.get(), type, e.shape[3], nel / e.shape[3]);
			}
			else
			{
				field = zfp_field_1d(e.ptr.get(), type, nel);
			}

			zfp_stream* zfp = zfp_stream_open(nullptr);
			zfp_stream_set_precision(zfp, compression);
			size_t bufsize = zfp_stream_maximum_size(zfp, field);

			bitstream* stream = stream_open(compressed_data.get(), bufsize);
			zfp_stream_set_bit_stream(zfp, stream);

			int64_t size = zfp_decompress(zfp, field);
			assert(size != 0);
			e.compressed_size = 0;

			cd.m_parameters[entry.first] = e;
		}
		return cd;
	}
}

StyleGAN StyleGANLoad(const char* filename)
{
	StyleGAN ctx;
	t4::model_dict dict = decompress(t4::load(filename), 12);
	//t4::model_dict dict = t4::load(filename);
	char wname[1024];
	for (int i = 0; i < 8; ++i)
	{
		sprintf(wname, "mapping_block_%d_weight", i + 1);
		dict.load(ctx.mapping_block_weight[i], wname, 512, 512);
		sprintf(wname, "mapping_block_%d_bias", i + 1);
		dict.load(ctx.mapping_block_bias[i], wname, 512);
	}
	dict.load(ctx.dlatent_avg, "dlatent_avg", 512);
	dict.load(ctx.block_0_const, "block_0_const", 1, 512, 4, 4);
	dict.load(ctx.latents, "latents", 1, 512);

	int fmap = 8192 / 2;
	int fmap_max = 512;
	for (int i = 0; i < 9; ++i)
	{
		int nf = std::min(fmap, fmap_max);
		fmap /= 2;

		sprintf(wname, "block_%d_noise_weight_1", i);
		dict.load(ctx.block[i].noise_weight_1, wname, 1, nf, 1, 1);
		sprintf(wname, "block_%d_noise_weight_2", i);
		dict.load(ctx.block[i].noise_weight_2, wname, 1, nf, 1, 1);
		if (i != 0)
		{
			sprintf(wname, "block_%d_conv_1_weight", i);
			if (i < 5)
			{
				dict.load(ctx.block[i].conv_1_weight, wname, nf, std::min(2 * nf, fmap_max), 3, 3);
			}
			else
			{
				dict.load(ctx.block[i].conv_1_weight, wname, std::min(2 * nf, fmap_max), nf, 4, 4);
			}
		}
		sprintf(wname, "block_%d_conv_2_weight", i);
		dict.load(ctx.block[i].conv_2_weight, wname, nf, nf, 3, 3);
		sprintf(wname, "block_%d_bias_1", i);
		dict.load(ctx.block[i].bias_1, wname, 1, nf, 1, 1);
		sprintf(wname, "block_%d_bias_2", i);
		dict.load(ctx.block[i].bias_2, wname, 1, nf, 1, 1);
		sprintf(wname, "block_%d_style_1_weight", i);
		dict.load(ctx.block[i].style_1_weight, wname, 2 * nf, 512);
		sprintf(wname, "block_%d_style_1_bias", i);
		dict.load(ctx.block[i].style_1_bias, wname, 2 * nf);
		sprintf(wname, "block_%d_style_2_weight", i);
		dict.load(ctx.block[i].style_2_weight, wname, 2 * nf, 512);
		sprintf(wname, "block_%d_style_2_bias", i);
		dict.load(ctx.block[i].style_2_bias, wname, 2 * nf);
		sprintf(wname, "block_%d_to_rgb_weight", i);
		dict.load(ctx.block[i].to_rgb_weight, wname, 3, nf, 1, 1);
		sprintf(wname, "block_%d_to_rgb_bias", i);
		dict.load(ctx.block[i].to_rgb_bias, wname, 3);
	}

	return ctx;
}


