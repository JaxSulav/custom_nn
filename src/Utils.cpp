#include "../include/Utils.h"


// Generate random number from normal distribution with a mean of 0 and sd of 1
double gen_rand_num_norm_dist()
{
    auto mean = 0;
    auto stdDev = 1;

    std::random_device rd;
    std::mt19937_64 randGenerator(rd());
    std::normal_distribution<> distribution(mean, stdDev);

    return distribution(randGenerator);
}


