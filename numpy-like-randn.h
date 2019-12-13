#pragma once
#include <math.h>
#include <random>
#include <inttypes.h>


namespace numpy_like
{
	class RandomState
	{
	public:
		RandomState(uint32_t _seed):gen(_seed)
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
		        double x1, x2, r2;

		        do {
		            x1 = 2.0 * next_double() - 1.0;
		            x2 = 2.0 * next_double() - 1.0;
		            r2 = x1 * x1 + x2 * x2;
		        } while (r2 >= 1.0 || r2 == 0.0);

		        double f = sqrt(-2.0 * log(r2) / r2);
		        m_gauss = f * x1;
		        m_has_gauss = true;
		        return f * x2;
		    }
		}

	private:
		double next_double()
		{
			uint32_t a = gen() >> 5, b = gen() >> 6;
			return (a * 67108864.0 + b) / 9007199254740992.0;
		}

		std::mt19937 gen;
		int m_has_gauss = 0;
		double m_gauss = 0.0;
	};
}
