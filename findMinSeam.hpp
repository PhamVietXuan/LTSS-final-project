#ifndef findminseam_h
#define findminseam_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

struct uchar3{
    uint8_t x, y, z;
};

int* readFileToMatrix(string fileName, int& width, int &height);
int min(int a, int b, int c);
int minIdx(int *arr, int idx, int width, int a, int b, int c);
int min2(int a, int b);
int* findMinSeam(int *matrix, int height, int width);

void removeSeam(uchar3 * matrix, int*seam, int &width, int height);
int * sobel(uint8_t *data, int width, int height, int * filter, int filterWidth);

#endif
