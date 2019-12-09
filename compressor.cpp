#include "tensor4.h"
#include "zfp.h"
#include <vector>

namespace t4
{
	inline  t4::model_dict compress(const t4::model_dict md, int compression)
	{
		t4::model_dict cd;

		for (auto it = md.m_parameters.cbegin(); it != md.m_parameters.cend(); ++it)
		{
			auto entry = *it;
			model_dict::Entry e;
			printf("%s\n", entry.first.c_str());
			fflush(0);
			e = entry.second;
			zfp_type type = zfp_type_float;
			zfp_field* field = nullptr;
			std::vector<uint64_t> sshape;
			assert(e.compressed_size == 0);

			int nel = 1;
			for (int i  = 0; i < e.ndim; ++i)
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
			// set compression mode and parameters
			zfp_stream_set_precision(zfp, compression);

			// allocate buffer for compressed data
			size_t bufsize = zfp_stream_maximum_size(zfp, field);
			uchar* buffer = new uchar[2 * bufsize];

			// associate bit stream with allocated buffer
			bitstream* stream = stream_open(buffer, bufsize);
			zfp_stream_set_bit_stream(zfp, stream);

			size_t size = zfp_compress(zfp, field);
			printf("Size: original: %d\n", int(e.size));
			printf("Size: %d\n", int(size));
			e.compressed_size = size;

			e.ptr.reset(new uchar[e.compressed_size]);
			memcpy(e.ptr.get(), buffer, e.compressed_size);
			delete[] buffer;
			cd.m_parameters[entry.first] = e;
		}
		return cd;
	}

	inline  t4::model_dict decompress(const t4::model_dict md, int compression)
	{
		t4::model_dict cd;

		for (auto it = md.m_parameters.cbegin(); it != md.m_parameters.cend(); ++it)
		{
			auto entry = *it;
			model_dict::Entry e;
			printf("%s\n", entry.first.c_str());
			fflush(0);
			e = entry.second;
			zfp_type type = zfp_type_float;
			zfp_field* field = nullptr;
			std::vector<uint64_t> sshape;
			assert(e.compressed_size != 0);

			auto compressed_data = e.ptr;
			e.ptr.reset(new uchar[e.size]);

			int nel = 1;
			for (int i  = 0; i < e.ndim; ++i)
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
//
//int main()
//{
//	t4::model_dict dict = t4::load("StyleGAN.t4");
//
//	int compression = 12;
//
//	dict = t4::compress(dict, compression);
//
//	t4::save(dict, "../StyleGAN.ct4");
//
//	t4::model_dict cdict = t4::load("../StyleGAN.ct4");
//
//	dict = t4::decompress(dict, compression);
//
//	t4::save(dict, "../StyleGAN.dct4");
//
//
//	return 0;
//}
