//#define T4_USE_OMP 1
//#define USE_MKLDNN
#include "tensor4.h"
#include "StyleGAN.h"
#include "image_io.h"
#include "numpy-like-randn.h"
#include "zfp.h"



#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>

using namespace emscripten;
#endif


#ifndef __EMSCRIPTEN__
int main()
{
	// if loading compressed
	auto model = StyleGANLoad("StyleGAN.ct4", 9);

	// if loading original
	// auto model = StyleGANLoad("StyleGAN_cat.t4", 7, false);

	{
		auto rs = numpy_like::RandomState(5);
		auto z = GenZ(rs);
		auto w = GenW(model, z);
		auto w_truncated = (w - t4::Unsqueeze<0>(model.dlatent_avg)) * 0.5f + t4::Unsqueeze<0>(model.dlatent_avg);
		t4::tensor4f x;
		t4::tensor3f img;
		for (int i = 0; i < 7; ++i)
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
	Generator():rng(std::random_device()())
	{
		model = StyleGANLoad("StyleGAN.ct4", 9);
		x = t4::tensor4f();
	}

	std::string RandomZ()
	{
		z = GenZ(rng);
		return image_io::base64_encode((uint8_t*)z.ptr(), 512 * sizeof(float));
	}

	std::string RandomZfromASeed(uint32_t seed)
	{
		rng = numpy_like::RandomState(seed);
		return RandomZ();
	}

	void SetZfromString(const std::string& s)
	{
		z = t4::tensor2f::New({1, 512});
		size_t _;
		image_io::base64_decode(s.c_str(), (uint8_t*)z.ptr(), s.size(), _);
	}

	std::string GenerateImage()
	{
		if (step == 0)
		{
			w = GenW(model, z);
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
	numpy_like::RandomState rng;
	StyleGAN model;
	t4::tensor4f x;
	t4::tensor2f z;
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
    .function("RandomZ", &Generator::RandomZ)
    .function("RandomZfromASeed", &Generator::RandomZfromASeed)
    .function("SetZfromString", &Generator::SetZfromString)
    ;
}
#endif
