#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int width;
    int height;
    unsigned char* data;
} PPMImage;

PPMImage* readPPM(const char* filename)
{
    unsigned char* image = NULL;

    char buff[16];
    FILE* fp;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file");
        return NULL;
    }

    // read image format
    if (!fgets(buff, sizeof(buff), fp)) {
        perror("Can't read image format");
        exit(1);
    }

    // read image size
    int x = 0;
    int y = 0;
    if (fscanf(fp, "%d %d", &x, &y) != 2) {
        fprintf(stderr, "Invalid image size");
    }
    printf("%d, %d\n", x,y);

    // read rbg component
    int rgb_comp = 0;
    if (fscanf(fp, "%d", &rgb_comp) != 1) {
        fprintf(stderr, "Invalid RGB component");
    }

    // allocate memory
    int size = x*y*3*sizeof(unsigned char);
    image = (unsigned char*)malloc(size);

    // read 
    int offset = 0;
    while (fscanf(fp, "%hhu %hhu %hhu", image+offset, image+offset+1, image+offset+2) == 3)
    {
        offset+= 3; 
    }

    fclose(fp);
    PPMImage* imageStruct = (PPMImage*)malloc(sizeof(PPMImage));
    imageStruct->width = x;
    imageStruct->height = y;
    imageStruct->data = image;

    return imageStruct;
}

void writePPM(const char* filename, PPMImage* image)
{
    FILE* fp = NULL;
    fp = fopen(filename, "w");   

    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", image->width, image->height);
    fprintf(fp, "255\n");

    int size = image->width * image->height;

    for (int i = 0; i < size; i++)
    {
        fprintf(fp, "%d %d %d\n", image->data[i*3], image->data[i*3+1], image->data[i*3+2]);
    }

    fclose(fp);
}

#endif /* UTILS_H */
