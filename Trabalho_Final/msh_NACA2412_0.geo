Point(1) = {-1.000, -1.000, 0, 0.500};
Point(2) = {3.000, -1.000, 0, 0.500};
Point(3) = {3.000, 1.000, 0, 0.500};
Point(4) = {-1.000, 1.000, 0, 0.500};
Point(5) = {1.000000, 0.001300, 0, 0.250000};
Point(6) = {0.950000, 0.011400, 0, 0.250000};
Point(7) = {0.900000, 0.020800, 0, 0.250000};
Point(8) = {0.800000, 0.037500, 0, 0.250000};
Point(9) = {0.700000, 0.051800, 0, 0.250000};
Point(10) = {0.600000, 0.063600, 0, 0.250000};
Point(11) = {0.500000, 0.072400, 0, 0.250000};
Point(12) = {0.400000, 0.078000, 0, 0.250000};
Point(13) = {0.300000, 0.078800, 0, 0.250000};
Point(14) = {0.250000, 0.076700, 0, 0.250000};
Point(15) = {0.200000, 0.072600, 0, 0.250000};
Point(16) = {0.150000, 0.066100, 0, 0.250000};
Point(17) = {0.100000, 0.056300, 0, 0.250000};
Point(18) = {0.075000, 0.049600, 0, 0.250000};
Point(19) = {0.050000, 0.041300, 0, 0.250000};
Point(20) = {0.025000, 0.029900, 0, 0.250000};
Point(21) = {0.012500, 0.021500, 0, 0.250000};
Point(22) = {0.000000, 0.000000, 0, 0.250000};
Point(23) = {0.012500, -0.016500, 0, 0.250000};
Point(24) = {0.025000, -0.022700, 0, 0.250000};
Point(25) = {0.050000, -0.030100, 0, 0.250000};
Point(26) = {0.075000, -0.034600, 0, 0.250000};
Point(27) = {0.100000, -0.037500, 0, 0.250000};
Point(28) = {0.150000, -0.041000, 0, 0.250000};
Point(29) = {0.200000, -0.042300, 0, 0.250000};
Point(30) = {0.250000, -0.042200, 0, 0.250000};
Point(31) = {0.300000, -0.041200, 0, 0.250000};
Point(32) = {0.400000, -0.038000, 0, 0.250000};
Point(33) = {0.500000, -0.033400, 0, 0.250000};
Point(34) = {0.600000, -0.027600, 0, 0.250000};
Point(35) = {0.700000, -0.021400, 0, 0.250000};
Point(36) = {0.800000, -0.015000, 0, 0.250000};
Point(37) = {0.900000, -0.008200, 0, 0.250000};
Point(38) = {0.950000, -0.004800, 0, 0.250000};
Point(39) = {1.000000, -0.001300, 0, 0.250000};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 17};
Line(17) = {17, 18};
Line(18) = {18, 19};
Line(19) = {19, 20};
Line(20) = {20, 21};
Line(21) = {21, 22};
Line(22) = {22, 23};
Line(23) = {23, 24};
Line(24) = {24, 25};
Line(25) = {25, 26};
Line(26) = {26, 27};
Line(27) = {27, 28};
Line(28) = {28, 29};
Line(29) = {29, 30};
Line(30) = {30, 31};
Line(31) = {31, 32};
Line(32) = {32, 33};
Line(33) = {33, 34};
Line(34) = {34, 35};
Line(35) = {35, 36};
Line(36) = {36, 37};
Line(37) = {37, 38};
Line(38) = {38, 39};
Line(39) = {39, 5};

Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
Plane Surface(1) = {1, 2};

Physical Line("inlet") = {1};
Physical Line("outlet") = {3};
Physical Line("lower_wall") = {2};
Physical Line("upper_wall") = {4};
Physical Line("airfoil") = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
Physical Surface("fluid") = {1};
