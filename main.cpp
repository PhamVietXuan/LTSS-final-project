#include "findMinSeam.hpp"
#include "Bitmap.hpp"

int main()
{
    int numC, width, height;
    uint8_t *pixels = NULL;
    readPnm("15-In.pnm", numC, width, height, pixels);
    uint8_t *grayPixels = new uint8_t[width*height];
    
    
    int filter1[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int filter2[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    
    for (int itr = 0; itr < 20; ++itr){
        convertRgb2Gray(pixels, width, height, grayPixels);
        
        int *sob1 = sobel(grayPixels, width, height, filter1, 3);
        int *sob2 = sobel(grayPixels, width, height, filter2, 3);
        for (int i = 0; i < width*height; i++)
            sob1[i] = abs(sob1[i]) + abs(sob2[i]);
        
        int *minSeam = findMinSeam(sob1, height, width);
        removeSeam((uchar3*)pixels, minSeam, width, height);
        
        delete [] sob1;
        delete [] sob2;
        delete [] minSeam;
    }
    
    writeBitmap(pixels, 3, width, height, "out.bmp");
    
    return 0;
}
