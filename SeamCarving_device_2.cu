#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define FILTER_WIDTH 3
__constant__ int dc_filter[FILTER_WIDTH * FILTER_WIDTH];
__device__ int bCount = 0;

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
  }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

void readPnm(char *fileName, int &numChannels, int &width, int &height,
             uint8_t *&pixels) {
  FILE *f = fopen(fileName, "r");
  if (f == NULL) {
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

void writePnm(uint8_t *pixels, int numChannels, int width, int height,
              char *fileName) {
  FILE *f = fopen(fileName, "w");
  if (f == NULL) {
    printf("Cannot write %s\n", fileName);
    exit(EXIT_FAILURE);
  }

  if (numChannels == 1)
    fprintf(f, "P2\n");
  else if (numChannels == 3)
    fprintf(f, "P3\n");
  else {
    fclose(f);
    printf("Cannot write %s\n", fileName);
    exit(EXIT_FAILURE);
  }

  fprintf(f, "%i\n%i\n255\n", width, height);

  for (int i = 0; i < width * height * numChannels; i++)
    fprintf(f, "%hhu\n", pixels[i]);

  fclose(f);
}

void convertRgb2Gray(uint8_t *inPixels, int width, int height,
                     uint8_t *outPixels) {
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      int i = r * width + c;
      uint8_t red = inPixels[3 * i];
      uint8_t green = inPixels[3 * i + 1];
      uint8_t blue = inPixels[3 * i + 2];
      outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
  }
}

void conv(uint8_t *data, int width, int height, int *filter, int filterWidth,
          int *outPixels) {
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      int tmp = 0;
      for (int fi = 0; fi < filterWidth; fi++)
        for (int fj = 0; fj < filterWidth; fj++) {
          int fVal = filter[fi * filterWidth + fj];
          int row = (i - filterWidth / 2) + fi;
          int column = (j - filterWidth / 2) + fj;

          row = min(height - 1, max(0, row));
          column = min(width - 1, max(0, column));

          tmp += fVal * data[row * width + column];
        }
      outPixels[i * width + j] = tmp;
    }
}

int minIdx(int *arr, int idx, int width, int a, int b, int c) {
  if (arr[idx * width + a] <= arr[idx * width + b] &&
      arr[idx * width + a] <= arr[idx * width + c])
    return a;
  if (arr[idx * width + b] <= arr[idx * width + a] &&
      arr[idx * width + b] <= arr[idx * width + c])
    return b;
  return c;
}

void findMinSeam(int *matrix, int width, int height, int *seam) {
  int *sum = (int *)malloc(height * width * sizeof(int));

  // first value row in sum eqal to matrix
  for (int i = 0; i < width; i++)
    sum[i] = matrix[i];

  // index in [i][j] in 2D matrix = i*width+j
  for (int i = 1; i < height; i++)
    for (int j = 0; j < width; j++) {
      sum[i * width + j] =
          matrix[i * width + j] +
          min(sum[(i - 1) * width + max(0, j - 1)],
              min(sum[(i - 1) * width + j],
                  sum[(i - 1) * width + min(width - 1, j + 1)]));
    }

  int step = 0;
  for (int i = 0; i < width; i++)
    if (sum[(height - 1) * width + i] < sum[(height - 1) * width + step])
      step = i;

  seam[height - 1] = step;
  for (int i = height - 1; i > 0; i--) {
    step = minIdx(sum, i - 1, width, max(0, step - 1), step,
                  min(step + 1, width - 1));
    seam[i - 1] = step;
  }

  free(sum);
}

void removeSeam(uchar3 *matrix, int &width, int height, int *seam) {
  for (int i = height - 1; i >= 0; i--) {
    for (int j = seam[i]; j < width * height - 1; j++) {
      matrix[i * width + j] = matrix[i * width + j + 1];
    }
  }
  width--;
}

void seamCarvingByHost(uint8_t *inPixels, int &width, int height, int numstep,
                       uint8_t *&outPixels) {
  GpuTimer timer;
  timer.Start();

  uint8_t *inPixels_ = (uint8_t *)malloc(width * height * 3);
  memcpy(inPixels_, inPixels, width * height * 3);

  uint8_t *grayscale = (uint8_t *)malloc(width * height * sizeof(uint8_t));
  int *sobel_h = (int *)malloc(width * height * sizeof(int));
  int *sobel_v = (int *)malloc(width * height * sizeof(int));
  int *sobel = (int *)malloc(width * height * sizeof(int));
  int *seam = (int *)malloc(height * sizeof(int));

  int filter_h[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  int filter_v[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

  for (int itr = 0; itr < numstep; ++itr) {
    // convert to grayscale
    GpuTimer timer1;
    timer1.Start();

    convertRgb2Gray(inPixels_, width, height, grayscale);

    timer1.Stop();
    printf("Processing time (use host - convert image to grayscale): %f ms\n",
           timer1.Elapsed());

    // edge detection
    GpuTimer timer2;
    timer2.Start();

    conv(grayscale, width, height, filter_h, 3, sobel_h);
    conv(grayscale, width, height, filter_v, 3, sobel_v);
    for (int i = 0; i < width * height; i++)
      sobel[i] = abs(sobel_h[i]) + abs(sobel_v[i]);

    timer2.Stop();
    printf("Processing time (use host - detect edge): %f ms\n",
           timer2.Elapsed());

    // find seam
    GpuTimer timer3;
    timer3.Start();

    findMinSeam(sobel, width, height, seam);

    timer3.Stop();
    printf("Processing time (use host - find optimum seam): %f ms\n",
           timer3.Elapsed());

    // carve image
    GpuTimer timer4;
    timer4.Start();

    removeSeam((uchar3 *)inPixels_, width, height, seam);

    timer4.Stop();
    printf("Processing time (use host - remove seam): %f ms\n",
           timer4.Elapsed());
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

__global__ void removeSeamKernel(uchar3 *inPixels, int width, int height,
                                 int *seam, uchar3 *outPixels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    int curIdx = i % width;
    int row = i / width;
    int offset = row;
    if (seam[row] < curIdx)
      ++offset;
    if (seam[row] != curIdx)
      outPixels[i - offset] = inPixels[i];
  }
}

__global__ void convKernel(uint8_t *inPixels, int width, int height,
                           int filterWidth, int *outPixels) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;

  // creat and copy data to SMEM
  extern __shared__ uint8_t s_inPixels[];
  int s_width = blockDim.x + filterWidth - 1;
  int s_height = blockDim.y + filterWidth - 1;
  int offsetX = blockIdx.x * blockDim.x;
  int offsetY = blockIdx.y * blockDim.y;

  for (int fi = threadIdx.x; fi < s_width; fi += blockDim.x)
    for (int fj = threadIdx.y; fj < s_height; fj += blockDim.y) {
      int inPixelRow = min(max(fi + offsetX - filterWidth / 2, 0), width - 1);
      int inPixelCol = min(max(fj + offsetY - filterWidth / 2, 0), height - 1);
      s_inPixels[fj * s_width + fi] = inPixels[inPixelRow + width * inPixelCol];
    }

  __syncthreads();

  int out = 0;
  for (int fi = 0; fi < filterWidth; fi++)
    for (int fj = 0; fj < filterWidth; fj++) {
      int fVal = dc_filter[fj * filterWidth + fi];

      out +=
          fVal * s_inPixels[(threadIdx.x + fi) + s_width * (threadIdx.y + fj)];
    }
  int i = py * width + px;
  outPixels[i] = out;
}

__global__ void convertRgb2GrayKernel(uint8_t *inPixels, int width, int height,
                                      uint8_t *outPixels) {
  // TODO
  // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < height && c < width) {
    int i = r * width + c;
    uint8_t red = inPixels[3 * i];
    uint8_t green = inPixels[3 * i + 1];
    uint8_t blue = inPixels[3 * i + 2];
    outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
  }
}

__global__ void addVecKernel(int *vec1, int *vec2, int length, int *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length)
    out[i] = abs(vec1[i]) + abs(vec2[i]);
}

__global__ void cumEnergyMapKernel2(int *matrix, int width, int *ceMap,
                                    int numBlks, volatile int *blockArr) {
  // Lay chi so bi cua block thay cho blockIdx
  __shared__ int bi;
  if (threadIdx.x == 0) {
    bi = atomicAdd(&bCount, 1);
    // printf("%d\n", bi);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (bi >= numBlks) {
      int r = bi / numBlks;
      int c = bi % numBlks;
      int iMid = bi - numBlks;
      int iLeft = (r - 1) * numBlks + max(c - 1, 0);
      int iRight = (r - 1) * numBlks + min(c + 1, numBlks - 1);
      while (blockArr[iMid] == 0 || blockArr[iLeft] == 0 ||
             blockArr[iRight] == 0)
        ;
    }
  }
  __syncthreads();

  int col = (bi % numBlks) * blockDim.x + threadIdx.x;
  int row = bi / numBlks;
  if (col < width) {
    if (row == 0) {
      ceMap[col] = matrix[col];
    } else {
      int idx = row * width + col;
      int middle = ceMap[(row - 1) * width + col];
      int left = ceMap[(row - 1) * width + max(col - 1, 0)];
      int right = ceMap[(row - 1) * width + min(col + 1, width - 1)];
      ceMap[idx] = matrix[idx] + min(left, min(right, middle));
    }
  }

  __syncthreads();
  if (threadIdx.x == 0)
    blockArr[bi] = 1;
}
__global__ void minKernel(int *in, int n, int *out) {

  // creat smem
  __shared__ int blkData[];
  blkData[threadIdx.x] = in[numElemsBeforeBlk + threadIdx.x];
  blkData[blockDim.x + threadIdx.x] =
      in[numElemsBeforeBlk + blockDim.x + threadIdx.x];
  __syncthreads();

  int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
  for (int stride = blockDim.x; stride > 0; stride /= 2) {
    int i = numElemsBeforeBlk + threadIdx.x;
    if (threadIdx.x < stride)
      if ((i - numElemsBeforeBlk) == threadIdx.x)
        in[i] = min(in[i], in[i + stride]);
    __syncthreads();
  }

  if (threadIdx.x == 0)
    out[blockIdx.x] = in[numElemsBeforeBlk];
}

void seamCarvingByDevice(uint8_t *inPixels, int &width, int height, int numstep,
                         uint8_t *&outPixels) {
  GpuTimer timer;
  timer.Start();

  int *seam = (int *)malloc(height * sizeof(int));
  int *ceMap = (int *)malloc(width * height * sizeof(int));

  int filter_h[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  int filter_v[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

  // device malloc
  uint8_t *d_inPixels;
  CHECK(cudaMalloc(&d_inPixels, width * height * 3));
  CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * 3,
                   cudaMemcpyHostToDevice));
  uint8_t *d_outPixels;
  CHECK(cudaMalloc(&d_outPixels, width * height * 3));
  int *d_seam;
  CHECK(cudaMalloc(&d_seam, height * sizeof(int)));
  int *d_sobel;
  CHECK(cudaMalloc(&d_sobel, width * height * sizeof(int)));
  int *d_ceMap;
  CHECK(cudaMalloc(&d_ceMap, width * height * sizeof(int)));
  int filterSize = FILTER_WIDTH * FILTER_WIDTH * sizeof(int);
  int *d_sobel_h;
  CHECK(cudaMalloc(&d_sobel_h, width * height * sizeof(int)));
  int *d_sobel_v;
  CHECK(cudaMalloc(&d_sobel_v, width * height * sizeof(int)));
  uint8_t *d_grayscale;
  CHECK(cudaMalloc(&d_grayscale, width * height * sizeof(uint8_t)));

  int blockSize = 128;
  dim3 blockSize2d(32, 32);

  int *blockArr;
  CHECK(cudaMalloc(&blockArr,
                   height * ((width - 1) / blockSize + 1) * sizeof(int)));

  for (int itr = 0; itr < numstep; ++itr) {
    int gridSize = (width * height - 1) / blockSize + 1;
    dim3 gridSize2d((width - 1) / blockSize2d.x + 1,
                    (height - 1) / blockSize2d.y + 1);

    // convert to grayscale
    GpuTimer timer1;
    timer1.Start();

    convertRgb2GrayKernel<<<gridSize2d, blockSize2d>>>(d_inPixels, width,
                                                       height, d_grayscale);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    timer1.Stop();
    printf("Processing time (use host - convert image to grayscale): %f ms\n",
           timer1.Elapsed());

    // edge detection
    GpuTimer timer2;
    timer2.Start();

    //--
    int n = width * height;
    size_t nBytes = n * sizeof(uint8_t);
    uint8_t *in = d_grayscale;
    uint8_t *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    int nStreams = 3;
    cudaStream_t *streams =
        (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));
    for (int i = 0; i < nStreams; i++)
      CHECK(cudaStreamCreate(&streams[i]));

    int streamSize = n / nStreams; // size of each stream

    int streamBytes = streamSize * sizeof(uint8_t);
    for (int i = 0; i < nStreams; i++) {
      int offset = streamSize * i;
      if ((i == nStreams - 1) && (nStreams != 1)) {
        streamBytes += (n % nStreams) * sizeof(uint8_t);
        streamSize += n % nStreams;
      }
      CHECK(cudaMemcpyToSymbol(dc_filter, filter_h, filterSize));
      convKernel<<<gridSize2d, blockSize2d,
                   (blockSize2d.x + FILTER_WIDTH - 1) *
                       (blockSize2d.y + FILTER_WIDTH - 1),
                   streams[i]>>>(&d_grayscale[offset], width, height,
                                 FILTER_WIDTH, &d_sobel_h[offset]);
      cudaDeviceSynchronize();
      CHECK(cudaGetLastError());
      CHECK(cudaMemcpyToSymbol(dc_filter, filter_v, filterSize));
      convKernel<<<gridSize2d, blockSize2d,
                   (blockSize2d.x + FILTER_WIDTH - 1) *
                       (blockSize2d.y + FILTER_WIDTH - 1),
                   streams[i]>>>(&d_grayscale[offset], width, height,
                                 FILTER_WIDTH, &d_sobel_v[offset]);
      cudaDeviceSynchronize();
      CHECK(cudaGetLastError());
    }
    //--

    /*
CHECK(cudaMemcpyToSymbol(dc_filter, filter_h, filterSize));
convKernel<<<gridSize2d, blockSize2d,
             (blockSize2d.x + FILTER_WIDTH - 1) *
                 (blockSize2d.y + FILTER_WIDTH - 1)>>>(
    d_grayscale, width, height, FILTER_WIDTH, d_sobel_h);
cudaDeviceSynchronize();
CHECK(cudaGetLastError());

CHECK(cudaMemcpyToSymbol(dc_filter, filter_v, filterSize));
convKernel<<<gridSize2d, blockSize2d,
             (blockSize2d.x + FILTER_WIDTH - 1) *
                 (blockSize2d.y + FILTER_WIDTH - 1)>>>(
    d_grayscale, width, height, FILTER_WIDTH, d_sobel_v);
cudaDeviceSynchronize();
CHECK(cudaGetLastError());
*/
    addVecKernel<<<gridSize, blockSize>>>(d_sobel_h, d_sobel_v, width * height,
                                          d_sobel);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    timer2.Stop();
    printf("Processing time (use device - detect edge): %f ms\n",
           timer2.Elapsed());

    // find seam
    GpuTimer timer3;
    timer3.Start();

    int gridSizeSeam = height * ((width - 1) / blockSize + 1);
    int numBlks = (width - 1) / blockSize + 1;
    int zero = 0;
    CHECK(cudaMemcpyToSymbol(bCount, &zero, sizeof(int)));
    // CHECK(cudaMemset(&bCount, 0, sizeof(int)));
    CHECK(cudaMemset(blockArr, 0, gridSizeSeam * sizeof(int)));

    cumEnergyMapKernel2<<<gridSizeSeam, blockSize>>>(d_sobel, width, d_ceMap,
                                                     numBlks, blockArr);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(ceMap, d_ceMap, width * height * sizeof(int),
                     cudaMemcpyDeviceToHost));

    int minPos = (height - 1) * width;
    for (int i = (height - 1) * width; i < width * height; ++i)
      if (ceMap[i] < ceMap[minPos])
        minPos = i;
    minPos -= (height - 1) * width;

    seam[height - 1] = minPos;
    for (int i = height - 1; i > 0; --i) {
      minPos = minIdx(ceMap, i - 1, width, max(0, minPos - 1), minPos,
                      min(minPos + 1, width - 1));
      seam[i - 1] = minPos;
    }

    timer3.Stop();
    printf("Processing time (use device - find optimum seam): %f ms\n",
           timer3.Elapsed());

    // carve image
    GpuTimer timer4;
    timer4.Start();

    CHECK(
        cudaMemcpy(d_seam, seam, height * sizeof(int), cudaMemcpyHostToDevice));

    removeSeamKernel<<<gridSize, blockSize>>>(
        (uchar3 *)d_inPixels, width, height, d_seam, (uchar3 *)d_outPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    uint8_t *tmp = d_inPixels;
    d_inPixels = d_outPixels;
    d_outPixels = tmp;

    --width;

    timer4.Stop();
    printf("Processing time (use device - remove seam): %f ms\n",
           timer4.Elapsed());
  }

  outPixels = (uint8_t *)malloc(width * height * 3);
  CHECK(cudaMemcpy(outPixels, d_inPixels, width * height * 3,
                   cudaMemcpyDeviceToHost));

  // device
  cudaFree(d_inPixels);
  cudaFree(d_outPixels);
  cudaFree(d_seam);
  cudaFree(d_sobel);
  cudaFree(d_ceMap);
  cudaFree(d_sobel_h);
  cudaFree(d_sobel_v);
  cudaFree(d_grayscale);
  cudaFree(blockArr);

  free(seam);
  free(ceMap);

  timer.Stop();
  float time = timer.Elapsed();
  printf("Processing time (use host - summary): %f ms\n", time);
}

void seamCarving(uint8_t *inPixels, int &width, int height, int numstep,
                 uint8_t *&outPixels, bool useDevice = false) {
  if (useDevice == false) {
    printf("\nSeam Carving by host\n");
    seamCarvingByHost(inPixels, width, height, numstep, outPixels);
  } else // use device
  {
    printf("\nSeam Carving by device\n");
    seamCarvingByDevice(inPixels, width, height, numstep, outPixels);
  }
}

char *concatStr(const char *s1, const char *s2) {
  char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
  strcpy(result, s1);
  strcat(result, s2);
  return result;
}

void printDeviceInfo() {
  cudaDeviceProp devProv;
  CHECK(cudaGetDeviceProperties(&devProv, 0));
  printf("**********GPU info**********\n");
  printf("Name: %s\n", devProv.name);
  printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
  printf("Num SMs: %d\n", devProv.multiProcessorCount);
  printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
  printf("Max num warps per SM: %d\n",
         devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
  printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
  printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
  printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
  printf("****************************\n");
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("The number of arguments is invalid\n");
    return EXIT_FAILURE;
  }

  // PRINT OUT DEVICE INFO
  printDeviceInfo();

  // Read input RGB image file
  int numChannels, width, height;
  uint8_t *inPixels;
  readPnm(argv[1], numChannels, width, height, inPixels);
  if (numChannels != 3)
    return EXIT_FAILURE; // Input image must be RGB
  printf("Image size (width x height): %i x %i\n\n", width, height);

  char *outFileNameBase = strtok(argv[3], "."); // Get rid of extension

  // Seam Carving using host
  int w = width;
  uint8_t *outPixelsHost = NULL;
  // seamCarving(inPixels, width, height, atoi(argv[2]), outPixelsHost);
  // writePnm(outPixelsHost, 3, width, height, concatStr(outFileNameBase,
  // "_host.pnm"));

  // Seam Carving using device
  width = w;
  uint8_t *outPixelsDevice = NULL;
  seamCarving(inPixels, width, height, 100, outPixelsDevice, true);
  writePnm(outPixelsDevice, 3, width, height,
           concatStr(outFileNameBase, "_device.pnm"));

  // Free memories
  free(inPixels);
  free(outPixelsHost);
  free(outPixelsDevice);

  return EXIT_SUCCESS;
}
