#pragma once
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
