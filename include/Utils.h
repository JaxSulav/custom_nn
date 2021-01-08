#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include "Matrix.h"

extern std::string DEBUG_STRING;
extern bool DEBUG;
extern int TEST_INT;
extern std::string TEST_STRING;

int parse_config(std::string cfgFile);

double gen_rand_num_norm_dist();

Matrix *multiply_matrices(Matrix *m1, Matrix *m2);

