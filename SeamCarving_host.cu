#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, 
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
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
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
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

void conv(uint8_t * data, int width, int height, int * filter, int filterWidth, int * outPixels)
{
    for(int i=0; i< height; i++)
        for(int j =0; j< width; j++)
        {
            int tmp = 0;
            for(int fi = 0; fi<filterWidth; fi++)
                for(int fj = 0; fj<filterWidth; fj++)
                {
                    int fVal = filter[fi*filterWidth + fj];
                    int row = (i - filterWidth/2) + fi;
                    int column = (j - filterWidth/2) + fj;

                    row = min(height - 1, max(0, row));
                    column = min(width-1, max(0, column));

                    tmp += fVal*data[row*width + column];
                }
            outPixels[i*width+j] = tmp;
        }
}

int minIdx(int*arr, int idx, int width, int a, int b, int c)
{
    if (arr[idx*width + a]<=arr[idx*width + b] && arr[idx*width + a]<=arr[idx*width + c]) return a;
    if (arr[idx*width + b]<=arr[idx*width + a] && arr[idx*width + b]<= arr[idx*width + c]) return b;
    return c;
}

void findMinSeam(int * matrix, int width, int height, int * seam)
{
    int * sum = (int*)malloc(height * width * sizeof(int));
    memset(sum, 0, height * width * sizeof(int)); 

    //first value row in sum eqal to matrix
    for (int i =1; i < width; i++)
        sum[i] = matrix[i];

    // index in [i][j] in 2D matrix = i*width+j
    for (int i =1; i < height; i++)
        for(int j =0; j< width; j++)
        {
            sum[i*width + j] = matrix[i*width + j] + min(sum[(i-1)*width + max(0,j-1)], min(sum[(i-1)*width + j], sum[(i-1)*width + min(width-1,j+1)]));
        }
    
    int step = 0;
    for (int i=0; i< width; i++)
        if(sum[(height - 1)*width + i] < sum[(height - 1)*width +step])
            step = i;

    seam[height-1] = step;
    for (int i = height - 1; i>0; i--)
    {
        step = minIdx(sum, i-1, width, max(0, step-1), step, min(step + 1, width - 1));
        seam[i-1] = step;
    }
    
    free(sum);
}

void removeSeam(uchar3 * matrix, int &width, int height, int * seam)
{
    for (int i = height - 1; i>=0; i--)
    {
        for (int j = seam[i]; j<width * height-1; j++){
            matrix[i*width + j] = matrix[i*width + j + 1];
        }
    }
    width--;
}

void seamCarving(uint8_t * inPixels, int &width, int height, int numstep, uint8_t * &outPixels)
{
    GpuTimer timer;
    timer.Start();

    uint8_t * inPixels_ = (uint8_t *)malloc(width * height * 3);
    memcpy(inPixels_, inPixels, width * height * 3);

    uint8_t * grayscale = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    int * sobel_h = (int *)malloc(width * height * sizeof(int));
    int * sobel_v = (int *)malloc(width * height * sizeof(int));
    int * sobel = (int *)malloc(width * height * sizeof(int));
    int * seam = (int *)malloc(height * sizeof(int));

    int filter_h[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int filter_v[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    for (int itr = 0; itr < numstep; ++itr)
    {
        // convert to grayscale
        GpuTimer timer1;
        timer1.Start();

        convertRgb2Gray(inPixels_, width, height, grayscale);

        timer1.Stop();
        printf("Processing time (use host - convert image to grayscale): %f ms\n", timer1.Elapsed());


        // edge detection
        GpuTimer timer2;
        timer2.Start();

        conv(grayscale, width, height, filter_h, 3, sobel_h);
        conv(grayscale, width, height, filter_v, 3, sobel_v);
        for (int i = 0; i < width*height; i++)
            sobel[i] = abs(sobel_h[i]) + abs(sobel_v[i]);

        timer2.Stop();
        printf("Processing time (use host - detect edge): %f ms\n", timer2.Elapsed());
        
        // find seam
        GpuTimer timer3;
        timer3.Start();

        findMinSeam(sobel, width, height, seam);

        timer3.Stop();
        printf("Processing time (use host - find optimum seam): %f ms\n", timer3.Elapsed());

        // carve image
        GpuTimer timer4;
        timer4.Start();

        removeSeam((uchar3 *)inPixels_, width, height, seam);
        
        timer4.Stop();
        printf("Processing time (use host - remove seam): %f ms\n", timer4.Elapsed());
    }

    free(grayscale);
    free(sobel_h);
    free(sobel_v);
    free(sobel);
    free(seam);

    outPixels = (uint8_t *)malloc(width * height * 3);
    memcpy(outPixels, inPixels_, width * height * 3);

    free(inPixels_);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use host - summary): %f ms\n", time);
}

int main(int argc, char ** argv)
{	
	if (argc != 4)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Convert RGB to grayscale not using device
	uint8_t * outPixels = NULL;
	seamCarving(inPixels, width, height, atoi(argv[2]), outPixels);

	// Write results to files
	writePnm(outPixels, 3, width, height, argv[3]);

	// Free memories
	free(inPixels);
	free(outPixels);
}
