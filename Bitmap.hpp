//
//  Bitmap.hpp
//  SeamCarving
//
//  Created by Nguyễn Trần Trung on 17/01/2022.
//

#ifndef Bitmap_hpp
#define Bitmap_hpp

void writeBitmap(uint8_t * pixels, int numChannels, int width, int height, char* imageFileName);
void convertRgb2Gray(uint8_t * inPixels, int width, int height, uint8_t * outPixels);
void readPnm(char * fileName, int &numChannels, int &width, int &height, uint8_t * &pixels);

#endif /* Bitmap_hpp */
