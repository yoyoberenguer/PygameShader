/* C implementation

                  GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

 Copyright Yoann Berenguer

*/

/*
gcc -O2 -fomit-frame-pointer -o ShaderLib ShaderLib.c
gcc -ffast-math -O3 -fomit-frame-pointer -o ShaderLib ShaderLib.c

WITH OPENMP
gcc -ffast-math -O3 -fopenmp -o ShaderLib ShaderLib.c

This will generate an object file (.o), now you take it and create the .so file:

gcc hello.o -shared -o libhello.so
EDIT: Suggestions from the comments:

gcc -shared -o libhello.so -fPIC hello.c

*/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

inline void swap(int* a, int* b);
inline int partition (int arr[], int low, int high);
inline int * quickSort(int arr[], int low, int high);
inline float Q_inv_sqrt( float number );
inline float hue_to_rgb(float m1, float m2, float hue);

inline struct hsl struct_rgb_to_hsl(float r, float g, float b);
inline struct rgb struct_hsl_to_rgb(float h, float s, float l);

inline struct rgb struct_hsv_to_rgb(float h, float s, float v);
inline struct hsv struct_rgb_to_hsv(float r, float g, float b);


inline float fmax_rgb_value(float red, float green, float blue);
inline float fmin_rgb_value(float red, float green, float blue);

inline struct rgb_color_int wavelength_to_rgb(int wavelength, float gamma);
inline struct rgb_color_int wavelength_to_rgb_custom(int wavelength, int arr[], float gamma);

inline float perlin(float x, float y);
inline float randRangeFloat(float lower, float upper);
inline int randRange(int lower, int upper);
inline int get_largest(int arr[], int n);
inline int get_lowest(int arr[], int n);

#define ONE_SIX 1.0/6.0
#define ONE_THIRD 1.0 / 3.0
#define TWO_THIRD 2.0 / 3.0
#define ONE_255 1.0/255.0
#define ONE_360 1.0/360.0

struct hsv{
    float h;    // hue
    float s;    // saturation
    float v;    // value
};

struct hsl{
    float h;    // hue
    float s;    // saturation
    float l;    // value
};

struct rgb{
    float r;
    float g;
    float b;
};

struct rgb_color_int{
    int r;
    int g;
    int b;
};

struct extremum{
    int min;
    int max;
};

void init_clock(){
    // clock_t t;
    srand(clock());
}

float randRangeFloat(float lower, float upper){
    return lower + ((float)rand()/(float)(RAND_MAX)) * (upper - lower);
}

int randRange(int lower, int upper)
{
    return (rand() % (upper - lower  + 1)) + lower;
}


// All inputs have to be float precision (python float) in range [0.0 ... 255.0]
// Output: return the maximum value from given RGB values (float precision).
inline float fmax_rgb_value(float red, float green, float blue)
{
    if (red>green){
        if (red>blue) {
		    return red;
	}
		else {
		    return blue;
	    }
    }
    else if (green>blue){
	    return green;
	}
    else {
	    return blue;
	}
}

// All inputs have to be float precision (python float) in range [0.0 ... 255.0]
// Output: return the minimum value from given RGB values (float precision).
inline float fmin_rgb_value(float red, float green, float blue)
{
    if (red<green){
        if (red<blue){
            return red;
        }
    else{
	    return blue;}
    }
    else if (green<blue){
	    return green;
	}
    else{
	    return blue;
	}
}



// A utility function to swap two elements
inline void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
	array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
inline int partition (int arr[], int low, int high)
{
	int pivot = arr[high]; // pivot
	int i = (low - 1); // Index of smaller element
    int j = 0;
//    #pragma omp parallel shared(j, arr, pivot, i, low, high) //shared(total_Sum)
//    {
//    #pragma omp parallel
//    {
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
//    }
//    }
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}

/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index */
inline int * quickSort(int arr[], int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		at right place */
		int pi = partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
return arr;
}



inline float Q_inv_sqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration

	return y;
}



inline float hue_to_rgb(float m1, float m2, float h)
{
    if ((fabs(h) > 1.0f) && (h > 0.0f)) {
      h = (float)fmod(h, 1.0f);
    }
    else if (h < 0.0f){
    h = 1.0f - (float)fabs(h);
    }

    if (h < ONE_SIX){
        return m1 + (m2 - m1) * h * 6.0f;
    }
    if (h < 0.5f){
        return m2;
    }
    if (h < TWO_THIRD){
        return m1 + ( m2 - m1 ) * (float)((float)TWO_THIRD - h) * 6.0f;
    }
    return m1;
}



// HSL: Hue, Saturation, Luminance
// H: position in the spectrum
// L: color lightness
// S: color saturation
// all inputs have to be float precision, (python float) in range [0.0 ... 1.0]
// outputs is a C array containing HSL values (float precision) normalized.
// h (°) = h * 360
// s (%) = s * 100
// l (%) = l * 100
inline struct hsl struct_rgb_to_hsl(float r, float g, float b)
{
    // check if all inputs are normalized
    assert ((0.0<= r) <= 1.0);
    assert ((0.0<= g) <= 1.0);
    assert ((0.0<= b) <= 1.0);

    struct hsl hsl_={.h=0.0f, .s=0.0f, .l=0.0f};

    float cmax=0.0f, cmin=0.0f, delta=0.0f, t;

    cmax = fmax_rgb_value(r, g, b);
    cmin = fmin_rgb_value(r, g, b);
    delta = (cmax - cmin);


    float h, l, s;
    l = (cmax + cmin) / 2.0f;

    if (delta == 0) {
    h = 0.0f;
    s = 0.0f;
    }
    else {
    	  if (cmax == r){
    	        t = (g - b) / delta;
    	        if ((fabs(t) > 6.0f) && (t > 0.0f)) {
                  t = (float)fmod(t, 6.0f);
                }
                else if (t < 0.0f){
                t = 6.0f - (float)fabs(t);
                }

	            h = 60.0f * t;
          }
    	  else if (cmax == g){
                h = 60.0f * (((b - r) / delta) + 2.0f);
          }

    	  else if (cmax == b){
    	        h = 60.0f * (((r - g) / delta) + 4.0f);
          }

    	  if (l <=0.5f) {
	            s=(delta/(cmax + cmin));
	      }
	  else {
	        s=(delta/(2.0f - cmax - cmin));
	  }
    }

    hsl_.h = (float)(h * (float)ONE_360);
    hsl_.s = s;
    hsl_.l = l;
    return hsl_;
}


// Convert HSL color model into RGB (red, green, blue)
// all inputs have to be float precision, (python float) in range [0.0 ... 1.0]
// outputs is a C array containing RGB values (float precision) normalized.
inline struct rgb struct_hsl_to_rgb(float h, float s, float l)
{

    struct rgb rgb_={.r=0.0f, .g=0.0f, .b=0.0f};

    float m2=0.0f, m1=0.0f;

    if (s == 0.0){
        rgb_.r = l;
        rgb_.g = l;
        rgb_.b = l;
        return rgb_;
    }
    if (l <= 0.5f){
        m2 = l * (1.0f + s);
    }
    else{
        m2 = l + s - (l * s);
    }
    m1 = 2.0f * l - m2;

    rgb_.r = hue_to_rgb(m1, m2, (float)(h + ONE_THIRD));
    rgb_.g = hue_to_rgb(m1, m2, h);
    rgb_.b = hue_to_rgb(m1, m2, (float)(h - ONE_THIRD));
    return rgb_;
}

/*
Return a structure instead of pointers
// outputs is a C structure containing 3 values, HSV (double precision)
// to convert in % do the following:
// h = h * 360.0
// s = s * 100.0
// v = v * 100.0
*/
inline struct hsv struct_rgb_to_hsv(float r, float g, float b)
{
//    // check if all inputs are normalized
//    assert ((0.0<=r) <= 1.0);
//    assert ((0.0<=g) <= 1.0);
//    assert ((0.0<=b) <= 1.0);

    float mx, mn;
    float h, df, s, v, df_;
    struct hsv hsv_;

    mx = fmax_rgb_value(r, g, b);
    mn = fmin_rgb_value(r, g, b);

    df = mx-mn;
    df_ = 1.0f/df;
    if (mx == mn)
    {
        h = 0.0f;}
    // The conversion to (int) approximate the final result
    else if (mx == r){
	    h = (float)fmod(60.0f * ((g-b) * df_) + 360.0f, 360);
	}
    else if (mx == g){
	    h = (float)fmod(60.0f * ((b-r) * df_) + 120.0f, 360);
	}
    else if (mx == b){
	    h = (float)fmod(60.0f * ((r-g) * df_) + 240.0f, 360);
    }
    if (mx == 0){
        s = 0.0f;
    }
    else{
        s = df/mx;
    }
    v = mx;
    hsv_.h = (float)(h * (float)ONE_360);
    hsv_.s = s;
    hsv_.v = v;
    return hsv_;
}

// Convert HSV color model into RGB (red, green, blue)
// all inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// outputs is a C structure containing RGB values (double precision) normalized.
// to convert for a pixel colors
// r = r * 255.0
// g = g * 255.0
// b = b * 255.0

inline struct rgb struct_hsv_to_rgb(float h, float s, float v)
{
//    // check if all inputs are normalized
//    assert ((0.0<= h) <= 1.0);
//    assert ((0.0<= s) <= 1.0);
//    assert ((0.0<= v) <= 1.0);

    int i;
    float f, p, q, t;
    struct rgb rgb_={.r=0.0, .g=0.0, .b=0.0};

    if (s == 0.0f){
        rgb_.r = v;
        rgb_.g = v;
        rgb_.b = v;
        return rgb_;
    }

    i = (int)(h*6.0f);

    f = (h*6.0f) - i;
    p = v*(1.0f - s);
    q = v*(1.0f - s*f);
    t = v*(1.0f - s*(1.0f-f));
    i = i%6;

    if (i == 0){
        rgb_.r = v;
        rgb_.g = t;
        rgb_.b = p;
        return rgb_;
    }
    else if (i == 1){
        rgb_.r = q;
        rgb_.g = v;
        rgb_.b = p;
        return rgb_;
    }
    else if (i == 2){
        rgb_.r = p;
        rgb_.g = v;
        rgb_.b = t;
        return rgb_;
    }
    else if (i == 3){
        rgb_.r = p;
        rgb_.g = q;
        rgb_.b = v;
        return rgb_;
    }
    else if (i == 4){
        rgb_.r = t;
        rgb_.g = p;
        rgb_.b = v;
        return rgb_;
    }
    else if (i == 5){
        rgb_.r = v;
        rgb_.g = p;
        rgb_.b = q;
        return rgb_;
    }
    return rgb_;
}




inline struct rgb_color_int wavelength_to_rgb(int wavelength, float gamma){
    /*

    == A few notes about color ==

    Color   Wavelength(nm) Frequency(THz)
    Red     620-750        484-400
    Orange  590-620        508-484
    Yellow  570-590        526-508
    Green   495-570        606-526
    Blue    450-495        668-606
    Violet  380-450        789-668

    f is frequency (cycles per second)
    l (lambda) is wavelength (meters per cycle)
    e is energy (Joules)
    h (Plank's constant) = 6.6260695729 x 10^-34 Joule*seconds
                         = 6.6260695729 x 10^-34 m^2*kg/seconds
    c = 299792458 meters per second
    f = c/l
    l = c/f
    e = h*f
    e = c*h/l

    List of peak frequency responses for each type of
    photoreceptor cell in the human eye:
        S cone: 437 nm
        M cone: 533 nm
        L cone: 564 nm
        rod:    550 nm in bright daylight, 498 nm when dark adapted.
                Rods adapt to low light conditions by becoming more sensitive.
                Peak frequency response shifts to 498 nm.

    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    */

    struct rgb_color_int color = {.r=0, .g=0, .b=0};
    float attenuation=0;

    // VIOLET
    if ((wavelength >= 380) & (wavelength <= 440))
    {
      attenuation = 0.3f + 0.7f * (wavelength - 380.0f) / 60.0f;
      color.r = (int)fmax((pow((((380 - wavelength) / 60.0f) * attenuation), gamma) * 255.0f), 0);

      color.b = (int)(pow(attenuation, gamma + 3.0f) * 255.0f);
    }

    // BLUE
    else if((wavelength >=440) && (wavelength <= 490))
    {
      color.g = (int)((float)pow((wavelength - 440) / 50.0f, gamma) * 255.0f);
      color.b = 255;
    }

    // GREEN
    else if ((wavelength>=490) && (wavelength <= 510)){
      color.g = 255;
      color.b = (int)((float)pow((510 - wavelength) / 20.0f, gamma) * 255.0f);
    }

    // YELLOW
    else if ((wavelength>=510) && (wavelength <= 580)){
      color.r = (int)((float)pow((wavelength - 510) / 70.0f, gamma) * 255.0f);
      color.g = 255;

    }
    // ORANGE
    else if ((wavelength>=580) && (wavelength <= 645)){
      color.r = 255;
      color.g = (int)((float)pow((645 - wavelength) / 65.0f, gamma) * 255.0f);
    }
    // RED
    else if ((wavelength>=645) && (wavelength <= 750)){
      attenuation = 0.3f + 0.7f * (750 - wavelength) / 105.0f;
      color.r = (int)((float)pow(attenuation, gamma) * 255.0f);

    }

    else
    {
    color.r = 0;
    color.g = 0;
    color.b = 0;
    }
    return color;
}




inline struct rgb_color_int wavelength_to_rgb_custom(int wavelength, int arr[], float gamma)
{

    struct rgb_color_int color = {.r=0, .g=0, .b=0};
    float attenuation=0;

    // VIOLET
    if ((wavelength >= arr[0]) & (wavelength <= arr[1]))
    {
      attenuation = 0.3f + 0.7f * (wavelength - (float)arr[0]) / (float)(arr[1] - arr[0]);
      color.r = (int)fmax((pow(((((float)arr[0] - wavelength) /
      (float)(arr[1] - arr[0])) * attenuation), gamma) * 255.0f), 0);
      color.b = (int)(pow(attenuation, gamma + 3.0f) * 255.0f);
    }

    // BLUE
    else if((wavelength >=arr[2]) && (wavelength <= arr[3]))
    {
      color.g = (int)((float)pow((wavelength - (float)arr[2]) /
      (float)(arr[3] - arr[2]), gamma) * 255.0f);
      color.b = 255;
    }


    // GREEN
    else if ((wavelength>=arr[4]) && (wavelength <= arr[5])){
      color.g = 255;
      color.b = (int)(pow(((float)arr[5] - wavelength) /(float)(arr[5] - arr[4]), gamma) * 255.0f);
    }



    // YELLOW
    else if ((wavelength>=arr[6]) && (wavelength <= arr[7])){
      color.r = (int)(pow((wavelength - (float)arr[6]) / (float)(arr[7] - arr[6]), gamma) * 255.0f);
      color.g = 255;

    }


    // ORANGE
    else if ((wavelength>=arr[8]) && (wavelength <= arr[9])){
      color.r = 255;
      color.g = (int)(pow(((float)arr[9] - wavelength) / (float)(arr[9] - arr[8]), gamma) * 255.0f);
    }


    // RED
    else if ((wavelength>=arr[10]) && (wavelength <= arr[11])){
      attenuation = 0.3f + 0.7f * ((float)arr[11] - wavelength) / (float)(arr[11] - arr[10]);
      color.r = (int)(pow(attenuation, gamma) * 255.0f);

    }
    else
    {
    wavelength = max(wavelength, 1000);
    attenuation = (float)(0.99f * (float)(1000 - wavelength) / (float)(1000 - arr[11]));
    color.r = (int)attenuation;
    color.r = (int)(pow(attenuation, gamma) * 255.0f);
    color.g = 0;
    color.b = 0;
    }
    return color;
}





/* Function to linearly interpolate between a0 and a1
 * Weight w should be in the range [0.0, 1.0]
 */
float interpolate(float a0, float a1, float w) {
    /* // You may want clamping by inserting:
     * if (0.0 > w) return a0;
     * if (1.0 < w) return a1;
     */
    return (a1 - a0) * w + a0;
    /* // Use this cubic interpolation [[Smoothstep]] instead, for a smooth appearance:
     * return (a1 - a0) * (3.0 - w * 2.0) * w * w + a0;
     *
     * // Use [[Smootherstep]] for an even smoother result with a second derivative equal to zero on boundaries:
     * return (a1 - a0) * ((w * (w * 6.0 - 15.0) + 10.0) * w * w * w) + a0;
     */
}

typedef struct {
    float x, y;
} vector2;

/* Create pseudorandom direction vector
 */
vector2 randomGradient(int ix, int iy) {
    // No precomputed gradients mean this works for any number of grid coordinates
    const unsigned w = 8 * sizeof(unsigned);
    const unsigned s = w / 2; // rotation width
    unsigned a = ix, b = iy;
    a *= 3284157443; b ^= (a << s) | (a >> (w-s));
    b *= 1911520717; a ^= (b << s) | (b >> (w-s));
    a *= 2048419325;
    float random = a * (3.14159265f / ~(~0u >> 1)); // in [0, 2*Pi]
    vector2 v;
    v.x = (float)sin(random); v.y = (float)cos(random);
    return v;
}

// Computes the dot product of the distance and gradient vectors.
float dotGridGradient(int ix, int iy, float x, float y) {
    // Get gradient from integer coordinates
    vector2 gradient = randomGradient(ix, iy);

    // Compute the distance vector
    float dx = x - (float)ix;
    float dy = y - (float)iy;

    // Compute the dot-product
    return (dx*gradient.x + dy*gradient.y);
}

// Compute Perlin noise at coordinates x, y
inline float perlin(float x, float y) {
    // Determine grid cell coordinates
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    // Determine interpolation weights
    // Could also use higher order polynomial/s-curve here
    float sx = x - (float)x0;
    float sy = y - (float)y0;

    // Interpolate between grid point gradients
    float n0, n1, ix0, ix1, value;

    n0 = dotGridGradient(x0, y0, x, y);
    n1 = dotGridGradient(x1, y0, x, y);
    ix0 = interpolate(n0, n1, sx);

    n0 = dotGridGradient(x0, y1, x, y);
    n1 = dotGridGradient(x1, y1, x, y);
    ix1 = interpolate(n0, n1, sx);

    value = interpolate(ix0, ix1, sy);
    return value;
}

// C function to find maximum in arr[] of size n
inline int get_largest(int arr[], int n)
{
    int i;

    // Initialize maximum element
    int max = arr[0];

    // Traverse array elements from second and
    // compare every element with current max
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];

    return max;
}


// C function to find minimum in arr[] of size n
inline int get_lowest(int arr[], int n)
{
    int i;

    // Initialize minimum element
    int min = arr[0];

    // Traverse array elements from second and
    // compare every element with current min
    for (i = 1; i < n; i++)
        if (arr[i] < min)
            min = arr[i];

    return min;
}




int main(){
return 0;
}