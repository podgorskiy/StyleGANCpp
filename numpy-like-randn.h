#pragma once
#include <math.h>
#include <random>
#include <inttypes.h>


namespace numpy_like
{
	// This RNG draws sequence of doubles from normal distribution that match ones from numpy.
	class RandomState
	{
	public:
		explicit RandomState(uint32_t _seed):gen(_seed)
		{}

		double randn()
		{
			if (m_has_gauss)
			{
		        m_has_gauss = false;
		        return m_gauss;
			}
			else
			{
		        for (;;)
		        {
		            double x1 = next_double();
		            double x2 = next_double() ;
		            double r2 = x1 * x1 + x2 * x2;
		            if (r2 < 1.0 && r2 != 0.0)
		            {
				        double f = sqrt(-2.0 * log(r2) / r2);
				        m_gauss = f * x1;
				        m_has_gauss = true;
				        return f * x2;
		            }
		        }
		    }
		}

	private:
		double next_double()
		{
			uint32_t a = uint32_t(gen()) >> 5U;
			uint32_t b = uint32_t(gen()) >> 6U;
			return (a * 67108864.0 + b) / 4503599627370496.0 - 1.0;
		}

		std::mt19937 gen;
		int m_has_gauss = 0;
		double m_gauss = 0.0;
	};
}
