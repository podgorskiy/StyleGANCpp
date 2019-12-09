//#define T4_USE_OMP 1
//#define USE_MKLDNN
#include "tensor4.h"
#include "StyleGAN.h"
#include "image_io.h"


t4::tensor2f MappingForward(const StyleGAN& model, t4::tensor2f x)
{
	for (int i = 0; i < 8; ++i)
	{
		x = t4::Linear(x, model.mapping_block_weight[i], model.mapping_block_bias[i]);
		x = t4::LeakyReluInplace(x, 0.2);
	}
	return x;
}


t4::tensor4f IN(t4::tensor4f x)
{
	t4::tensor4f e = t4::GlobalAveragePool2d(x);
	x = x - e;
	t4::tensor4f var= t4::GlobalAveragePool2d(x * x);
	var = t4::Pow(var + 1e-8f, 0.5f);
	return x / var;
}


t4::tensor4f style_mod(t4::tensor4f x, t4::tensor2f style)
{
	t4::tensor1i shape = t4::Shape(style);
	t4::tensor1i count = t4::Unsqueeze<0>(t4::Gather(shape, t4::Constant<t4::int64>(0)));
	t4::tensor1i component = t4::Unsqueeze<0>(t4::Gather(shape, t4::Constant<t4::int64>(1)));
	component = component / int64_t(2);
	t4::tensor1i chunks = t4::Unsqueeze<0>(t4::Constant<t4::int64>(2));
	t4::tensor1i _shape = t4::Concat<0>(t4::Concat<0>(count, chunks), component);
	auto _style = t4::Unsqueeze<3>(t4::Unsqueeze<3>(t4::Reshape<3>(style, _shape)));
	t4::tensor4f style_1 = t4::tensor4f::New({_style.shape()[0], _style.shape()[2], 1, 1});
	t4::tensor4f style_2 = t4::tensor4f::New({_style.shape()[0], _style.shape()[2], 1, 1});
	for (int i = 0; i < _style.shape()[0]; ++i)
	{
		style_1.Sub(i).Assign(_style.Sub(i, 0));
		style_2.Sub(i).Assign(_style.Sub(i, 1));
	}
	return style_2 + x * (style_1 + 1.0f);
}

t4::tensor4f blur2d(t4::tensor4f in)
{
	//T4_ScopeProfiler(blur2d)
	const int N = number(in);
	const int C = channels(in);
	const int H = height(in);
	const int W = width(in);

	in = t4::Pad<t4::constant>(in, 0, 0, 1, 1, 0, 0, 1, 1);
	const int _W = W + 2;

	t4::tensor4f out = t4::tensor4f::New({ N, C, H, W });

	for (int n = 0; n < N; ++n)
	{
		parallel_for(int c = 0; c < C; ++c)
		{
			auto inSubtensor = in.Sub(n, c);
			const float* __restrict src = inSubtensor.ptr();
			auto outSubtensor = out.Sub(n, c);
			float* __restrict dst = outSubtensor.ptr();

			for (int i = 0; i < H; i++)
			{
				for (int j = 0; j < W; j++)
				{
					float v = 0;
					i += 1;
					j += 1;
					v += src[(i + 0) * _W + j + 0] * 4;
					v += src[(i + 0) * _W + j - 1] * 2;
					v += src[(i + 0) * _W + j + 1] * 2;
					v += src[(i - 1) * _W + j + 0] * 2;
					v += src[(i + 1) * _W + j + 0] * 2;
					v += src[(i - 1) * _W + j - 1];
					v += src[(i + 1) * _W + j - 1];
					v += src[(i - 1) * _W + j + 1];
					v += src[(i + 1) * _W + j + 1];
					i -= 1;
					j -= 1;
					v /= (4 + 2 * 4 + 4);
					dst[i * W + j] = v;
				}
			}
		}
	}
	return out;
}


t4::tensor4f updcale2d(t4::tensor4f in)
{
	//T4_ScopeProfiler(updcale2d)
	const int N = number(in);
	const int C = channels(in);
	const int Hin = height(in);
	const int Win = width(in);

	const int Hout = 2 * Hin;
	const int Wout = 2 * Win;

	t4::tensor4f out = t4::tensor4f::New({ N, C, Hout, Wout });

	for (int n = 0; n < N; ++n)
	{
		parallel_for(int c = 0; c < C; ++c)
		{
			auto inSubtensor = in.Sub(n, c);
			const float* __restrict src = inSubtensor.ptr();
			auto outSubtensor = out.Sub(n, c);
			float* __restrict dst = outSubtensor.ptr();

			for (int i = 0; i < Hin; i++)
			{
				for (int j = 0; j < Win; j++)
				{
					dst[(2 * i + 0) * Wout + 2 * j + 0] = src[i * Win + j];
					dst[(2 * i + 0) * Wout + 2 * j + 1] = src[i * Win + j];
					dst[(2 * i + 1) * Wout + 2 * j + 0] = src[i * Win + j];
					dst[(2 * i + 1) * Wout + 2 * j + 1] = src[i * Win + j];
				}
			}
		}
	}
	return out;
}

t4::tensor2f GenW(StyleGAN model)
{
	//auto z = t4::tensor2f::RandN({1, 512});
	auto z = model.latents;
	float s = 0;
	const float* __restrict src = z.ptr();
	for (int64_t i = 0, l = z.size(); i < l; ++i)
	{
		float x = src[i] * src[i];
		s += x;
	}
	s /= z.size();

	z = z / sqrt(s + 1e-8f);

	auto w = MappingForward(model, z);

	return w;
}

std::pair<t4::tensor4f, t4::tensor3f> GenImage(StyleGAN model, t4::tensor4f x, t4::tensor2f w, int step)
{
	if (step == 0)
	{
		x = model.block_0_const;
	}
	else
	{
		if (step < 5)
		{
			x = updcale2d(x);
			x = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x, model.block[step].conv_1_weight);
		}
		else
		{
			x = t4::ConvTranspose2d<4, 4, 2, 2, 1, 1, 1, 1>(x, model.block[step].conv_1_weight);
		}
		x = blur2d(x);
	}

	x = x + model.block[step].noise_weight_1 * t4::tensor4f::RandN({x.shape()[0], 1, x.shape()[2], x.shape()[3]});

	x = x + model.block[step].bias_1;

	x = t4::LeakyReluInplace(x, 0.2);

	x = IN(x);

	auto s1 = t4::Linear(w, model.block[step].style_1_weight, model.block[step].style_1_bias);

	x = style_mod(x, s1);
	t4::release(s1);

	x = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x, model.block[step].conv_2_weight);

	x = x + model.block[step].noise_weight_2 * t4::tensor4f::RandN({x.shape()[0], 1, x.shape()[2], x.shape()[3]});

	x = x + model.block[step].bias_2;

	x = t4::LeakyReluInplace(x, 0.2);

	x = IN(x);

	auto s2 = t4::Linear(w, model.block[step].style_2_weight, model.block[step].style_2_bias);

	x = style_mod(x, s2);
	t4::release(s2);

	auto img = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x, model.block[step].to_rgb_weight, model.block[step].to_rgb_bias).Sub(0);
	return std::make_pair(x, img);
}


#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>

using namespace emscripten;
#endif


#ifndef __EMSCRIPTEN__
int main()
{
	auto model = StyleGANLoad("StyleGAN.ct4");

	//for(int i = 0; i < 100; ++i)
	{
		auto w = GenW(model);
		auto w_truncated = (w - t4::Unsqueeze<0>(model.dlatent_avg)) * 0.7f + t4::Unsqueeze<0>(model.dlatent_avg);
		t4::tensor4f x;
		t4::tensor3f img;
		for (int i = 0; i < 9; ++i)
		{
			t4::tensor2f current_w = w;
			if (i < 4)
			{
				current_w = w_truncated;
			}
			auto result = GenImage(model, x, current_w, i);
			x = result.first;
			img = result.second;
		}
		image_io::imwrite(img * 0.5f + 0.5f, "image_12.png");
	}

	return 0;
}
#endif

class Generator
{
public:
	Generator()
	{
		model = StyleGANLoad("StyleGAN.ct4");
		x = t4::tensor4f();
	}
	std::string GenerateImage()
	{
		if (step == 0)
		{
			w = GenW(model);
			w_truncated = (w - t4::Unsqueeze<0>(model.dlatent_avg)) * 0.7f + t4::Unsqueeze<0>(model.dlatent_avg);
		}

		t4::tensor2f current_w = w;
		if (step < 4)
		{
			current_w = w_truncated;
		}

		auto result = GenImage(model, x, current_w, step);
		step++;
		if (step== 9)
		{
			step = 0;
		}

		x = result.first;
		std::string image_png = image_io::imwrite_to_base64(result.second * 0.5f + 0.5f);
		return image_png;
	}

private:
	StyleGAN model;
	t4::tensor4f x;
	t4::tensor2f w;
	t4::tensor2f w_truncated;
	int step = 0;
};

#ifdef __EMSCRIPTEN__
// Binding code
EMSCRIPTEN_BINDINGS(StyleGan) {
  class_<Generator>("Generator")
    .constructor<>()
    .function("GenerateImage", &Generator::GenerateImage)
    ;
}
#endif
