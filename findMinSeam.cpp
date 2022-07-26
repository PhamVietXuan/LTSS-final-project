#include "findMinSeam.hpp"

int* readFileToMatrix(string fileName, int& width, int &height)
{
    int* data;
    int x, y;
    ifstream inFile;
    inFile.open("input.txt");
    inFile>>x>>y;
    width = x;
    height = y;
    data = (int*)malloc(height * width * sizeof(int));
    

    for (int i =0; i < height; i++)
    for(int j =0; j< width; j++)
    {
        inFile >> data[i* width + j];
    }
    inFile.close();
    return data;
    
}
int min(int a, int b, int c)
{
    if (a<=b && a<=c) return a;
    if (b<=a && b<= c) return b;
    return c;
}
int minIdx(int*arr, int idx, int width, int a, int b, int c)
{
    if (arr[idx*width + a]<=arr[idx*width + b] && arr[idx*width + a]<=arr[idx*width + c]) return a;
    if (arr[idx*width + b]<=arr[idx*width + a] && arr[idx*width + b]<= arr[idx*width + c]) return b;
    return c;
}
int min2(int a, int b)
{
    if (a<= b) return a;
    return b;
}
int* findMinSeam(int *matrix, int height, int width)
{
    int *sum;
    sum =(int*)malloc(height * width * sizeof(int));
    for (int i =0; i < height * width; i++)
        sum[i] = 0;

    //first value row in sum eqal to matrix
    for (int i =1; i < width; i++)
        sum[i] = matrix[i];

    // index in [i][j] in 2D matrix = i*width+j
    for (int i =1; i < height; i++)
    for(int j =0; j< width; j++)
        {
            sum[i*width + j] = matrix[i*width + j] + min(sum[(i-1)*width + max(0,j-1)], sum[(i-1)*width + j], sum[(i-1)*width + min2(width-1,j+1)]);
        }
    
    int step = 0;
    int *seam = (int*)malloc(height * sizeof(int));

    for (int i=0; i< width; i++)
        if(sum[(height - 1)*width + i] < sum[(height - 1)*width +step])
            step = i;
    seam[height-1] = step;
    for (int i = height - 1; i>0; i--)
    {
        step = minIdx(sum,i-1, width, max(0, step-1), step, min(step + 1, width - 1));
        seam[i-1] = step;
    }
    
    free(sum);

    return seam;
}

void removeSeam(uchar3 * matrix, int*seam, int &width, int height)
{
   for (int i = height - 1; i>=0; i--)
   {
       for (int j = seam[i]; j<width * height-1; j++)
            matrix[i*width + j] = matrix[i*width + j + 1];
   }

    width--;
    return;
}

int * sobel(uint8_t *data, int width, int height, int * filter, int filterWidth)
{
    int*res = (int*)malloc(width*height*sizeof(int));

    for(int i=0; i< height; i++)
    for(int j =0; j< width; j++)
    {
        int tmp = 0;
        for(int fi = 0; fi<filterWidth; fi++)
        for(int fj = 0; fj<filterWidth; fj++)
        {
            int fVal;
            fVal = filter[fi*filterWidth + fj];
            int row = (i - filterWidth/2) + fi;
            int column = (j - filterWidth/2) + fj;

            row = min(height - 1, max(0, row));
            column = min(width-1, max(0, column));

            tmp += fVal*data[row*width + column];
        }
        res[i*width+j] = tmp;

    }

    return res;
}
