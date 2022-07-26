#include <stdio.h>
#include <stdint.h>
#include <string.h>

const int BYTES_PER_PIXEL = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);
unsigned char* createBitmapFileHeader(int height, int stride);
unsigned char* createBitmapInfoHeader(int height, int width);

void writeBitmap(uint8_t * pixels, int numChannels, int width, int height, char* imageFileName){
    unsigned char image[height][width][BYTES_PER_PIXEL];

    if (numChannels == 1){
        int i, j;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                int k = (height - i - 1) * width + j;
                image[i][j][2] = (unsigned char) ( pixels[k] ); ///red
                image[i][j][1] = (unsigned char) ( pixels[k] ); ///green
                image[i][j][0] = (unsigned char) ( pixels[k] ); ///blue
            }
        }
    }
    else{
        int i, j;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                int k = (height - i - 1) * width + j;
                image[i][j][2] = (unsigned char) ( pixels[3 * k + 0] ); ///red
                image[i][j][1] = (unsigned char) ( pixels[3 * k + 1] ); ///green
                image[i][j][0] = (unsigned char) ( pixels[3 * k + 2] ); ///blue
            }
        }
    }
    generateBitmapImage((unsigned char*) image, height, width, imageFileName);
}

void convertRgb2Gray(uint8_t * inPixels, int width, int height, uint8_t * outPixels){
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            uint8_t red = inPixels[3 * i];
            uint8_t green = inPixels[3 * i + 1];
            uint8_t blue = inPixels[3 * i + 2];
            outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
        }
    }

}

void readPnm(char * fileName, int &numChannels, int &width, int &height, uint8_t * &pixels)
{
    FILE * f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
//        exit(EXIT_FAILURE);
        return;
    }

    char type[3];
    fscanf(f, "%s", type);
    if (strcmp(type, "P2") == 0)
        numChannels = 1;
    else if (strcmp(type, "P3") == 0)
        numChannels = 3;
    else // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
//        exit(EXIT_FAILURE);
        return;
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
//        exit(EXIT_FAILURE);
        return;
    }

//    pixels = (uint8_t *)malloc(width * height * numChannels);
    pixels = (uint8_t *)(new char[width * height * numChannels]);
    for (int i = 0; i < width * height * numChannels; i++)
        fscanf(f, "%hhu", &pixels[i]);

    fclose(f);
}


void generateBitmapImage (unsigned char* image, int height, int width, char* imageFileName)
{
    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = {0, 0, 0};
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes) + paddingSize;

    FILE* imageFile = fopen(imageFileName, "wb");

    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

    int i;
    for (i = 0; i < height; i++) {
        fwrite(image + (i*widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
    }

    fclose(imageFile);
}

unsigned char* createBitmapFileHeader (int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[ 0] = (unsigned char)('B');
    fileHeader[ 1] = (unsigned char)('M');
    fileHeader[ 2] = (unsigned char)(fileSize      );
    fileHeader[ 3] = (unsigned char)(fileSize >>  8);
    fileHeader[ 4] = (unsigned char)(fileSize >> 16);
    fileHeader[ 5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char* createBitmapInfoHeader (int height, int width)
{
    static unsigned char infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    infoHeader[ 0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[ 4] = (unsigned char)(width      );
    infoHeader[ 5] = (unsigned char)(width >>  8);
    infoHeader[ 6] = (unsigned char)(width >> 16);
    infoHeader[ 7] = (unsigned char)(width >> 24);
    infoHeader[ 8] = (unsigned char)(height      );
    infoHeader[ 9] = (unsigned char)(height >>  8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL*8);

    return infoHeader;
}
