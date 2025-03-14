#!/bin/bash

##./compute_chb_univariate_features.py --feature=katz_fractal_dimension
./compute_siena_univariate_features.py --feature=katz_fractal_dimension
./compute_tusz_univariate_features.py --feature=katz_fractal_dimension
./compute_tuep_univariate_features.py --feature=katz_fractal_dimension