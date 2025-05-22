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

inline float * my_sort(float buffer[], int filter_size);

inline void swap(int* a, int* b);
inline int partition (int arr[], int low, int high);
inline int * quickSort(int arr[], int low, int high);

inline void new_swap(unsigned char* a, unsigned char* b);
inline int new_partition (unsigned char arr[], int low, int high);
inline unsigned char * new_quickSort(unsigned char arr[], int low, int high);



inline float Q_inv_sqrt(float number );
inline float hue_to_rgb(float m1, float m2, float hue);

inline struct hsl struct_rgb_to_hsl(const float r, const float g, const float b);
inline struct rgb struct_hsl_to_rgb(const float h, const float s, const float l);

inline struct rgb struct_hsv_to_rgb(const float h, const float s, const float v);
inline struct hsv struct_rgb_to_hsv(const float r, const float g, const float b);

inline struct yiq rgb_to_yiq(const float r, const float g, const float b);
inline struct rgb yiq_to_rgb(const float y, const float i, const float q);

inline float fmax_rgb_value(float red, float green, float blue);
inline float fmin_rgb_value(float red, float green, float blue);

inline struct rgb_color_int wavelength_to_rgb(int wavelength, float gamma);
inline struct rgb_color_int wavelength_to_rgb_custom(int wavelength, int arr[], float gamma);

inline float perlin(float x, float y);
inline float randRangeFloat(float lower, float upper);
inline int randRange(int lower, int upper);
inline int get_largest(int arr[], int n);
inline float min_f(float arr[], unsigned int n);
inline struct s_min minf_struct(float arr[], unsigned int n);
inline int min_c(int arr[], int n);
inline unsigned int min_index(float arr[], unsigned int n);

#define ONE_SIX 1.0/6.0
#define ONE_THIRD 1.0 / 3.0
#define TWO_THIRD 2.0 / 3.0
#define ONE_255 1.0/255.0
#define ONE_360 1.0/360.0

struct im_stats{
    float red_mean;
    float red_std_dev;
    float green_mean;
    float green_std_dev;
    float blue_mean;
    float blue_std_dev;
};

struct im_stats_with_alpha{
    float red_mean;
    float red_std_dev;
    float green_mean;
    float green_std_dev;
    float blue_mean;
    float blue_std_dev;
    float alpha_mean;
    float alpha_std_dev;
};

struct lab{
    float l;
    float a;
    float b;
};

struct xyz{
    float x;
    float y;
    float z;
};

struct yiq{
    float y;
    float i;
    float q;
};

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

struct s_min{
    float value;
    unsigned int index;
};

void init_clock(){
    // clock_t t;
    srand(clock());
}


/*
randRangeFloat - Generate a Random Floating-Point Number Within a Specified Range
Description:

The randRangeFloat function generates a random floating-point number within the specified range [lower, upper].
It uses the standard rand() function to generate a pseudo-random integer and then normalizes and scales
the result to fall within the desired range.
Parameters:

    lower (float): The lower bound of the range. The generated number will be greater than or equal to this value.
    upper (float): The upper bound of the range. The generated number will be less than or equal to this value.

Returns:

A random floating-point number within the range [lower, upper].
How It Works:
    Generate a Random Integer: The rand() function generates a random integer between 0 and RAND_MAX.
    Normalize to Range [0, 1]: By dividing the result of rand() by RAND_MAX, the number is normalized to a
    floating-point value in the range [0.0, 1.0].
    Scale to Desired Range: The normalized value is then scaled by multiplying it by (upper - lower) to
    span the range between lower and upper.
    Shift to Start from lower: Finally, lower is added to the result, ensuring the generated number falls
    within the correct range [lower, upper].

Formula:

The formula for generating a random floating-point number within the range [lower, upper] is:
random_float=lower+(rand()RAND_MAX)×(upper−lower)
random_float=lower+(RAND_MAXrand()​)×(upper−lower)
*/

// Generates a random floating-point number within the specified range [lower, upper].
float randRangeFloat(float lower, float upper) {
    // rand() generates a random integer between 0 and RAND_MAX.
    // Dividing by RAND_MAX normalizes it to a range [0, 1].
    // Multiplying by (upper - lower) scales it to the desired range.
    // Adding 'lower' shifts it to start from the lower bound.
    return lower + ((float)rand() / (float)(RAND_MAX)) * (upper - lower);
}


/*
randRange - Generate a Random Integer Within a Specified Range
Description:

The randRange function generates a random integer within a specified range [lower, upper],
inclusive of both bounds. It utilizes the rand() function to produce a random number and then
scales and shifts the result to fit within the given range.
Parameters:

    lower (int): The lower bound of the range. The generated number will be greater than or equal to this value.
    upper (int): The upper bound of the range. The generated number will be less than or equal to this value.

Returns:

A random integer within the range [lower, upper], inclusive.
How It Works:

    Generate a Random Integer: The rand() function generates a pseudo-random integer between 0 and RAND_MAX.
    Scale to Range Size: The result of rand() is taken modulo (upper - lower + 1).
    This ensures that the random number fits within the desired range size and includes the upper bound.
    Shift to Desired Range: Adding the lower value shifts the generated number to start from the lower bound,
    ensuring the final result lies within [lower, upper].

Formula:

The formula for generating a random integer within the range [lower, upper] is:
random_int=(rand()%(upper−lower+1))+lower

*/
// Function to generate a random integer within a specified range
int randRange(int lower, int upper)
{
    // rand() generates a random number, then we take the remainder of dividing it by the range size
    // (upper - lower + 1) to ensure the range is inclusive of both bounds.
    // Adding 'lower' shifts the result into the desired range.
    return (rand() % (upper - lower + 1)) + lower;
}


/*
fmax_rgb_value - Return the Maximum Value from RGB Components
Description:

The fmax_rgb_value function takes three floating-point values representing the RGB components of
a color and returns the maximum value among them. The inputs are expected to be in the range [0.0, 255.0],
which corresponds to the typical range for RGB components in many color models. This function is useful
for tasks like determining the brightness or intensity of a color by finding the maximum component.
Parameters:

    red (float): The red component of the color, in the range [0.0, 255.0].
    green (float): The green component of the color, in the range [0.0, 255.0].
    blue (float): The blue component of the color, in the range [0.0, 255.0].

Returns:

The function returns the maximum value among the three RGB components (red, green, and blue).
This value is a float representing the highest intensity among the RGB components.
How It Works:

    The function compares the values of red, green, and blue using a series of if and else statements.
    It checks if red is greater than green and blue. If true, red is returned as the maximum.
    If green is greater than both red and blue, green is returned.
    If neither of the above conditions are met, then blue is returned as the maximum value.

Formula:

The maximum value max_value is computed as:
max_value=max⁡(red,green,blue)
max_value=max(red,green,blue)
*/

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


inline float * my_sort(float buffer[], int filter_size){
float temp=0;
int i, j;
for (i = 0; i < (filter_size - 1); ++i)
    {
        for (j = 0; j < filter_size - 1 - i; ++j )
        {
            if (buffer[j] > buffer[j+1])
            {
                temp = buffer[j+1];
                buffer[j+1] = buffer[j];
                buffer[j] = temp;
            }
        }
    }
return buffer;
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
    for (j = low; j <= high- 1; j++)
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




// A utility function to swap two elements
inline void new_swap(unsigned char* a, unsigned char* b)
{
	unsigned char t = *a;
	*a = *b;
	*b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
	array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
inline int new_partition (unsigned char arr[], int low, int high)
{
	unsigned char pivot = arr[high]; // pivot
	int i = (low - 1); // Index of smaller element
    int j = 0;
//    #pragma omp parallel shared(j, arr, pivot, i, low, high) //shared(total_Sum)
//    {
//    #pragma omp parallel
//    {
    for (j = low; j <= high- 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            new_swap(&arr[i], &arr[j]);
        }
    }
//    }
//    }
	new_swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}

/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index */
inline unsigned char * new_quickSort(unsigned char arr[], int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		at right place */
		int pi = new_partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
		new_quickSort(arr, low, pi - 1);
		new_quickSort(arr, pi + 1, high);
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


// Function to convert hue to RGB value based on the given parameters m1, m2, and h
inline float hue_to_rgb(float m1, float m2, float h)
{
    // Ensure that hue 'h' is within the range [0.0, 1.0] (wraps or adjusts if out of bounds)
    if ((fabsf(h) > 1.0f) && (h > 0.0f)) {
        // Wrap 'h' into the [0, 1] range by using modulo operation
        h = (float)fmodf(h, 1.0f);
    }
    else if (h < 0.0f) {
        // If 'h' is negative, adjust it to be within the [0, 1] range
        h = 1.0f - (float)fabsf(h);
    }

    // The following conditions compute the RGB value based on the adjusted hue
    // 1. If hue is in the range [0, 1/6), compute the value using a linear interpolation
    if (h < ONE_SIX) {
        return m1 + (m2 - m1) * h * 6.0f;
    }

    // 2. If hue is in the range [1/6, 1/2), simply return 'm2' (a constant value)
    if (h < 0.5f) {
        return m2;
    }

    // 3. If hue is in the range [1/2, 2/3), compute the value using a linear interpolation
    if (h < TWO_THIRD) {
        return m1 + (m2 - m1) * (float)((float)TWO_THIRD - h) * 6.0f;
    }

    // 4. If hue is in the range [2/3, 1), return the base value 'm1'
    return m1;
}


/*
struct_rgb_to_hsl - Convert RGB to HSL Color Model
Description:

The struct_rgb_to_hsl function converts RGB (Red, Green, Blue) color values to the HSL
(Hue, Saturation, Lightness) color model. The input RGB values should be in the normalized range [0.0, 1.0].
The function returns a structure of type hsl containing the corresponding HSL values.
Parameters:

    r (float): The red component of the color, in the range [0.0, 1.0].
    g (float): The green component of the color, in the range [0.0, 1.0].
    b (float): The blue component of the color, in the range [0.0, 1.0].

Returns:

A struct hsl containing the converted HSL values:

    h (float): The hue component of the color, in the range [0.0, 1.0] (normalized for full hue spectrum).
    s (float): The saturation component of the color, in the range [0.0, 1.0].
    l (float): The lightness component of the color, in the range [0.0, 1.0].

How It Works:

    Normalize Inputs: The function expects that the input RGB values (r, g, b) are in the normalized
    range [0.0, 1.0]. If they are outside this range, the behavior is undefined.
    Calculate Maximum and Minimum Values:
        The maximum (cmax) and minimum (cmin) values of the RGB components are determined
        using fmax_rgb_value and fmin_rgb_value.
        The delta is the difference between cmax and cmin, representing the color contrast or saturation.
    Calculate Lightness (L): The lightness is computed as the average of cmax and cmin.
    Determine Saturation (S): If the delta is 0 (meaning the color is grayscale), saturation is set to 0.
    Otherwise, it is calculated based on the relationship between the delta, cmax, and cmin.
    Determine Hue (H):
        The hue is calculated based on the dominant color component (r, g, or b)
        and the relationship between the other components.
        If delta is non-zero, hue is computed by determining the angle in the color wheel,
        and the result is scaled to the [0.0, 360.0] range.

Formula:

    Lightness (L):

L=cmax+cmin2
L=2cmax+cmin​

    Saturation (S):

    If L <= 0.5, then:
    S=δcmax+cmin
    S=cmax+cminδ​
    If L > 0.5, then:
    S=δ2.0−cmax−cmin
    S=2.0−cmax−cminδ​

    Hue (H):
    If cmax == r, then:

H=60×(g−bδ)
H=60×(δg−b​)

If cmax == g, then:
H=60×(b−rδ+2)
H=60×(δb−r​+2)

If cmax == b, then:
H=60×(r−gδ+4)
H=60×(δr−g​+4)

    Hue normalization: The resulting hue H is normalized to the range [0.0, 1.0] by dividing by 360.0.

*/
// HSL: Hue, Saturation, Luminance
// H: position in the spectrum
// S: color saturation
// L: color lightness
// all inputs have to be float precision, (python float) in range [0.0 ... 1.0]
// Outputs is a struct containing HSL values (float precision) normalized
// h (°) = h * 360
// s (%) = s * 100
// l (%) = l * 100
inline struct hsl struct_rgb_to_hsl(const float r, const float g, const float b)
{
    // Assert that RGB values are within the normalized range [0.0, 1.0]
    assert(r >= 0.0f && r <= 1.0f);
    assert(g >= 0.0f && g <= 1.0f);
    assert(b >= 0.0f && b <= 1.0f);

    struct hsl hsl_;
    hsl_.h=0.0f;
    hsl_.s=0.0f;
    hsl_.l=0.0f;

    float cmax=0.0f, cmin=0.0f, delta=0.0f, t;

    cmax = fmax_rgb_value(r, g, b);
    cmin = fmin_rgb_value(r, g, b);
    delta = (cmax - cmin);


    float h=0.0f, l, s=0.0f;
    l = (cmax + cmin) * 0.5f;

    if (delta == 0) {
    h = 0.0f;
    s = 0.0f;
    }
    else {
    	  if (cmax == r){
    	        t = (g - b) / delta;
    	        if ((fabsf(t) > 6.0f) && (t > 0.0f)) {
                  t = (float)fmodf(t, 6.0f);
                }
                else if (t < 0.0f){
                t = 6.0f - (float)fabsf(t);
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



/*

struct_hsl_to_rgb - Convert HSL to RGB Color Model
Description:

The struct_hsl_to_rgb function converts HSL (Hue, Saturation, Lightness) values to the RGB (Red, Green, Blue)
color model. The function takes in HSL values and computes the corresponding RGB values in
the normalized range [0.0, 1.0]. This function uses the HSL-to-RGB conversion formulas to
generate the appropriate RGB values.
Parameters:

    h (float): The hue component of the color, in the range [0.0, 1.0]. Hue is normalized for
    a full hue spectrum (360 degrees mapped to the range [0.0, 1.0]).
    s (float): The saturation component of the color, in the range [0.0, 1.0]. A saturation of 0
    results in a grayscale color, while 1 represents the full saturation (vibrant color).
    l (float): The lightness component of the color, in the range [0.0, 1.0]. A value of 0.0
    represents black, 1.0 represents white, and 0.5 represents a fully saturated color.

Returns:

A struct rgb containing the converted RGB values:

    r (float): The red component of the color, in the range [0.0, 1.0].
    g (float): The green component of the color, in the range [0.0, 1.0].
    b (float): The blue component of the color, in the range [0.0, 1.0].

How It Works:

    Gray Color Handling (Saturation = 0):
        If the saturation (s) is 0, the color is a shade of gray, and the RGB values are all
        set to the lightness value (l), creating a neutral color (gray).
    Lightness Adjustments:
        If lightness (l) is less than or equal to 0.5, the m2 value is calculated using
        m2 = l * (1.0 + s). This represents the adjusted maximum RGB component.
        If lightness (l) is greater than 0.5, m2 is calculated as m2 = l + s - (l * s) to prevent oversaturation.
    Calculate m1:
        The m1 value is calculated as m1 = 2.0f * l - m2, which helps determine the lower bound of the RGB components.
    Hue-to-RGB Conversion:
        The hue is adjusted for the red, green, and blue components by adding or subtracting 1/3 to
        determine the appropriate color shifts. These adjustments are passed to the helper function
        hue_to_rgb, which calculates the RGB values for each channel.
    RGB Final Values:
        The final RGB values are calculated and returned as a structure containing the
        normalized red, green, and blue components.
*/
// Convert HSL color model into RGB (red, green, blue)
// all inputs have to be float precision, (python float) in range [0.0 ... 1.0]
// outputs is a struct containing RGB values (float precision) normalized.
// Convert HSL values to RGB
inline struct rgb struct_hsl_to_rgb(const float h, const float s, const float l)
{
    struct rgb rgb_;  // Initialize the RGB structure to store the result
    rgb_.r = 0.0f;    // Default value for Red channel
    rgb_.g = 0.0f;    // Default value for Green channel
    rgb_.b = 0.0f;    // Default value for Blue channel

    float m2 = 0.0f, m1 = 0.0f;

    // If saturation is 0, the color is a shade of gray (the RGB values are equal)
    if (s == 0.0f) {
        rgb_.r = l;  // Set all RGB components to the lightness value
        rgb_.g = l;
        rgb_.b = l;
        return rgb_;  // Return the gray color (no hue or saturation)
    }

    // Otherwise, calculate m2 and m1 based on the lightness (l) and saturation (s)
    if (l <= 0.5f) {
        // If lightness is <= 0.5, use this formula for m2
        m2 = l * (1.0f + s);
    }
    else {
        // If lightness is > 0.5, use this formula for m2
        m2 = l + s - (l * s);
    }

    // Calculate m1 using the formula based on m2 and l
    m1 = 2.0f * l - m2;

    // Calculate RGB components using the hue and the m1, m2 values
    // The hue is adjusted by adding/subtracting 1/3 to get the red, green, and blue components
    rgb_.r = hue_to_rgb(m1, m2, (float)(h + ONE_THIRD));  // Red channel hue adjustment (h + 1/3)
    rgb_.g = hue_to_rgb(m1, m2, h);                       // Green channel hue (no adjustment)
    rgb_.b = hue_to_rgb(m1, m2, (float)(h - ONE_THIRD));  // Blue channel hue adjustment (h - 1/3)

    return rgb_;  // Return the final RGB structure
}

/*
variables r, g, b are normalised color components
Return a structure instead of pointers
// outputs is a C structure containing 3 values, HSV (double precision)
// to convert in % do the following:
// h = h * 360.0
// s = s * 100.0
// v = v * 100.0
*/
// Convert RGB values to HSV
inline struct hsv struct_rgb_to_hsv(const float r, const float g, const float b)
{
     // Assert that RGB values are within the normalized range [0.0, 1.0]
    assert(r >= 0.0f && r <= 1.0f);
    assert(g >= 0.0f && g <= 1.0f);
    assert(b >= 0.0f && b <= 1.0f);

    // Initialize variables
    float mx, mn;       // Max and min RGB values
    float h = 0.0f;     // Hue, initialized to 0
    float df, s, v;     // Delta, saturation, and value (brightness)
    float df_;          // Inverse of delta for efficiency
    struct hsv hsv_;    // Structure to store the final HSV values

    // Find the maximum and minimum RGB values
    mx = fmax_rgb_value(r, g, b);  // Max value
    mn = fmin_rgb_value(r, g, b);  // Min value

    // Calculate the difference between max and min RGB values
    df = mx - mn;
    df_ = 1.0f / df;  // Inverse of the delta for later use in hue calculations

    // Hue calculation based on the max RGB value
    if (mx == mn) {
        // If the RGB values are the same (gray), there is no hue (undefined)
        h = 0.0f;  // Gray does not have a hue
    }
    else if (mx == r) {
        // If red is the max, compute the hue based on green and blue
        h = (float)fmodf(60.0f * ((g - b) * df_) + 360.0f, 360);
    }
    else if (mx == g) {
        // If green is the max, compute the hue based on red and blue
        h = (float)fmodf(60.0f * ((b - r) * df_) + 120.0f, 360);
    }
    else if (mx == b) {
        // If blue is the max, compute the hue based on red and green
        h = (float)fmodf(60.0f * ((r - g) * df_) + 240.0f, 360);
    }

    // Saturation calculation
    if (mx == 0) {
        // If the max value is 0 (black), there is no saturation
        s = 0.0f;
    }
    else {
        // Calculate saturation based on the value (mx)
        s = df / mx;
    }

    // Value (brightness) is just the max RGB value
    v = mx;

    // Normalize hue to the range [0, 1] (i.e., divide by 360 if needed)
    hsv_.h = (float)(h * (float)ONE_360);  // Assuming ONE_360 is 1/360
    hsv_.s = s;  // Saturation
    hsv_.v = v;  // Value (Brightness)

    // Return the HSV structure
    return hsv_;
}

// Convert HSV color model into RGB (red, green, blue)
// all inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// outputs is a C structure containing RGB values (double precision) normalized.
// to convert for a pixel colors
// r = r * 255.0
// g = g * 255.0
// b = b * 255.0
// Convert HSV values to RGB
inline struct rgb struct_hsv_to_rgb(const float h, const float s, const float v)
{
    int i;          // Index for hue sector (0 to 5)
    float f, p, q, t;  // Intermediate variables for calculations
    struct rgb rgb_;  // Structure to store the resulting RGB values
    rgb_.r = 0.0f;    // Initialize red to 0
    rgb_.g = 0.0f;    // Initialize green to 0
    rgb_.b = 0.0f;    // Initialize blue to 0

    // Case when saturation is 0, meaning the color is a shade of gray
    if (s == 0.0f) {
        // When saturation is 0, all RGB channels are equal to the value (v)
        rgb_.r = v;
        rgb_.g = v;
        rgb_.b = v;
        return rgb_;  // Return the grayscale RGB value
    }

    // Calculate the sector of the hue (h) in the color wheel (from 0 to 5)
    i = (int)(h * 6.0f);  // Convert hue to one of 6 sectors (each 60 degrees)

    f = (h * 6.0f) - i;  // Fractional part of hue to help calculate the RGB interpolation
    p = v * (1.0f - s);  // Calculate p: the value when the color is desaturated
    q = v * (1.0f - s * f);  // Calculate q: the value at a specific hue point in the sector
    t = v * (1.0f - s * (1.0f - f));  // Calculate t: the value at the other side of the hue sector

    // Ensure i is within the range [0, 5] by taking the modulus 6
    i = i % 6;

    // Determine the RGB values based on the sector in the hue wheel
    if (i == 0) {
        // Red is the maximum, Green is t, Blue is p
        rgb_.r = v;
        rgb_.g = t;
        rgb_.b = p;
    }
    else if (i == 1) {
        // Green is the maximum, Red is q, Blue is p
        rgb_.r = q;
        rgb_.g = v;
        rgb_.b = p;
    }
    else if (i == 2) {
        // Green is the maximum, Blue is t, Red is p
        rgb_.r = p;
        rgb_.g = v;
        rgb_.b = t;
    }
    else if (i == 3) {
        // Blue is the maximum, Green is q, Red is p
        rgb_.r = p;
        rgb_.g = q;
        rgb_.b = v;
    }
    else if (i == 4) {
        // Blue is the maximum, Red is t, Green is p
        rgb_.r = t;
        rgb_.g = p;
        rgb_.b = v;
    }
    else if (i == 5) {
        // Red is the maximum, Blue is q, Green is p
        rgb_.r = v;
        rgb_.g = p;
        rgb_.b = q;
    }

    return rgb_;  // Return the RGB structure with the final values
}

/*
rgb_to_yiq - Convert RGB to YIQ Color Model
Description:

The rgb_to_yiq function converts RGB (Red, Green, Blue) color values to the YIQ color model.
The YIQ color model is used primarily in video encoding systems (such as NTSC) and separates the
luminance (Y) from chrominance (I and Q) to optimize the encoding of color information.

    Y (Luminance): Represents the brightness of the color (monochrome information).
    I (In-phase chrominance): Encodes the color difference along the red-blue axis.
    Q (Quadrature chrominance): Encodes the color difference along the green-blue axis.

This function computes the Y, I, and Q components based on the input RGB values and returns
a structure containing these values.

Parameters:

    r (float): The red component of the color, in the range [0.0, 1.0].
    g (float): The green component of the color, in the range [0.0, 1.0].
    b (float): The blue component of the color, in the range [0.0, 1.0].

Returns:

The function returns a struct yiq containing the following components:

    y (float): Luminance (Y) component, in the range [0.0, 1.0].
    i (float): In-phase chrominance (I) component, in the range [-1.0, 1.0].
    q (float): Quadrature chrominance (Q) component, in the range [-1.0, 1.0].

Formula:

The conversion from RGB to YIQ is done using the following formulas:

    Y (Luminance):
    Y=0.299⋅R+0.587⋅G+0.114⋅B
    Y=0.299⋅R+0.587⋅G+0.114⋅B

    I (In-phase Chrominance):
    I=0.5959⋅R−0.2746⋅G−0.3213⋅B
    I=0.5959⋅R−0.2746⋅G−0.3213⋅B

    Q (Quadrature Chrominance):
    Q=0.2115⋅R−0.5227⋅G+0.3112⋅B
    Q=0.2115⋅R−0.5227⋅G+0.3112⋅B
*/
// Convert RGB values to YIQ
inline struct yiq rgb_to_yiq(const float r, const float g, const float b)
{
    // Initialize the YIQ structure to hold the resulting Y, I, Q values
    struct yiq yiq_;
    yiq_.y = 0.0f;  // Luminance (Y)
    yiq_.i = 0.0f;  // In-phase chrominance (I)
    yiq_.q = 0.0f;  // Quadrature chrominance (Q)

    // Calculate the Y (luminance) component using the standard formula
    // Y = 0.299 * R + 0.587 * G + 0.114 * B
    yiq_.y = (float)0.299 * r + (float)0.587 * g + (float)0.114 * b;

    // Calculate the I (in-phase chrominance) component
    // I = 0.5959 * R - 0.2746 * G - 0.3213 * B
    yiq_.i = (float)0.5959 * r - (float)0.2746 * g - (float)0.3213 * b;

    // Calculate the Q (quadrature chrominance) component
    // Q = 0.2115 * R - 0.5227 * G + 0.3112 * B
    yiq_.q = (float)0.2115 * r - (float)0.5227 * g + (float)0.3112 * b;

    // Return the resulting YIQ structure
    return yiq_;
}


/*

yiq_to_rgb - Convert YIQ to RGB Color Model
Description:

The yiq_to_rgb function converts YIQ color model values to RGB (Red, Green, Blue) color space.
The YIQ model is commonly used in video encoding (e.g., NTSC) and separates luminance (Y) from
chrominance (I and Q). This function performs the reverse transformation, converting YIQ values back to RGB.

    Y (Luminance): Represents the brightness of the color.
    I (In-phase chrominance): Encodes the color difference along the red-blue axis.
    Q (Quadrature chrominance): Encodes the color difference along the green-blue axis.

This function uses a mathematical formula to compute the RGB components from the provided
YIQ components and ensures the resulting RGB values are within the valid range [0.0, 1.0].
Parameters:

    y (float): The luminance component of the color, in the range [0.0, 1.0].
    i (float): The in-phase chrominance component, in the range [-1.0, 1.0].
    q (float): The quadrature chrominance component, in the range [-1.0, 1.0].

Returns:

The function returns a struct rgb containing the RGB components of the color:

    r (float): The red component, in the range [0.0, 1.0].
    g (float): The green component, in the range [0.0, 1.0].
    b (float): The blue component, in the range [0.0, 1.0].

Formula:

The conversion from YIQ to RGB is done using the following formulas:

    R (Red):
    R=Y+0.956⋅I+0.619⋅Q
    R=Y+0.956⋅I+0.619⋅Q

    G (Green):
    G=Y−0.272⋅I−0.647⋅Q
    G=Y−0.272⋅I−0.647⋅Q

    B (Blue):
    B=Y−1.106⋅I+1.703⋅Q
    B=Y−1.106⋅I+1.703⋅Q

After the RGB components are calculated, they are clamped to ensure that they stay within the valid range [0.0, 1.0]:

    If any component is less than 0.0, it is clamped to 0.0.
    If any component is greater than 1.0, it is clamped to 1.0.

*/


// Convert YIQ to RGB color model
inline struct rgb yiq_to_rgb(const float y, const float i, const float q)
{
    // Initialize the rgb structure to hold the result
    struct rgb rgb_;

    rgb_.r = 0.0f;  // Red component (initialize to 0.0)
    rgb_.g = 0.0f;  // Green component (initialize to 0.0)
    rgb_.b = 0.0f;  // Blue component (initialize to 0.0)

    // Calculate the RGB values using the YIQ-to-RGB conversion formulas:
    // R = Y + 0.956 * I + 0.619 * Q
    float r = y + (float)0.956 * i + (float)0.619 * q;

    // G = Y - 0.272 * I - 0.647 * Q
    float g = y - (float)0.272 * i - (float)0.647 * q;

    // B = Y - 1.106 * I + 1.703 * Q
    float b = y - (float)1.106 * i + (float)1.703 * q;

    // Assign the calculated RGB values to the rgb structure
    rgb_.r = r;
    rgb_.g = g;
    rgb_.b = b;

    // Ensure the RGB components are within the valid range [0.0, 1.0]
    if (r < 0) {
        rgb_.r = (float)0.0;  // Clamp red to 0.0 if it is less than 0
    } else if (r > 1.0) {
        rgb_.r = (float)1.0;  // Clamp red to 1.0 if it exceeds 1.0
    }

    if (g < 0) {
        rgb_.g = (float)0.0;  // Clamp green to 0.0 if it is less than 0
    } else if (g > 1.0) {
        rgb_.g = (float)1.0;  // Clamp green to 1.0 if it exceeds 1.0
    }

    if (b < 0) {
        rgb_.b = (float)0.0;  // Clamp blue to 0.0 if it is less than 0
    } else if (b > 1.0) {
        rgb_.b = (float)1.0;  // Clamp blue to 1.0 if it exceeds 1.0
    }

    // Return the RGB structure with the final values
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

    struct rgb_color_int color;
    color.r=(int)0;
    color.g=(int)0;
    color.b=(int)0;

    float attenuation=0;

    // VIOLET
    if ((wavelength >= 380) & (wavelength <= 440))
    {
      attenuation = 0.3f + 0.7f * (wavelength - 380.0f) / 60.0f;
      color.r = (int)fmaxf((powf((((380 - wavelength) / 60.0f) * attenuation), gamma) * 255.0f), 0);

      color.b = (int)(powf(attenuation, gamma + 3.0f) * 255.0f);
    }

    // BLUE
    else if((wavelength >=440) && (wavelength <= 490))
    {
      color.g = (int)((float)powf((wavelength - 440) / 50.0f, gamma) * 255.0f);
      color.b = 255;
    }

    // GREEN
    else if ((wavelength>=490) && (wavelength <= 510)){
      color.g = 255;
      color.b = (int)((float)powf((510 - wavelength) / 20.0f, gamma) * 255.0f);
    }

    // YELLOW
    else if ((wavelength>=510) && (wavelength <= 580)){
      color.r = (int)((float)powf((wavelength - 510) / 70.0f, gamma) * 255.0f);
      color.g = 255;

    }
    // ORANGE
    else if ((wavelength>=580) && (wavelength <= 645)){
      color.r = 255;
      color.g = (int)((float)powf((645 - wavelength) / 65.0f, gamma) * 255.0f);
    }
    // RED
    else if ((wavelength>=645) && (wavelength <= 750)){
      attenuation = 0.3f + 0.7f * (750 - wavelength) / 105.0f;
      color.r = (int)((float)powf(attenuation, gamma) * 255.0f);

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

    struct rgb_color_int color;
    color.r = (int)0;
    color.g = (int)0;
    color.b = (int)0;

    float attenuation=0;

    // VIOLET
    if ((wavelength >= arr[0]) & (wavelength <= arr[1]))
    {
      attenuation = 0.3f + 0.7f * (wavelength - (float)arr[0]) / (float)(arr[1] - arr[0]);
      color.r = (int)fmaxf(((float)pow(((((float)arr[0] - wavelength) /
      (float)(arr[1] - arr[0])) * attenuation), gamma) * 255.0f), 0);
      color.b = (int)(powf(attenuation, gamma + 3.0f) * 255.0f);
    }

    // BLUE
    else if((wavelength >=arr[2]) && (wavelength <= arr[3]))
    {
      color.g = (int)((float)powf((wavelength - (float)arr[2]) /
      (float)(arr[3] - arr[2]), gamma) * 255.0f);
      color.b = 255;
    }


    // GREEN
    else if ((wavelength>=arr[4]) && (wavelength <= arr[5])){
      color.g = 255;
      color.b = (int)(powf(((float)arr[5] - wavelength) /(float)(arr[5] - arr[4]), gamma) * 255.0f);
    }



    // YELLOW
    else if ((wavelength>=arr[6]) && (wavelength <= arr[7])){
      color.r = (int)(powf((wavelength - (float)arr[6]) / (float)(arr[7] - arr[6]), gamma) * 255.0f);
      color.g = 255;

    }


    // ORANGE
    else if ((wavelength>=arr[8]) && (wavelength <= arr[9])){
      color.r = 255;
      color.g = (int)(powf(((float)arr[9] - wavelength) / (float)(arr[9] - arr[8]), gamma) * 255.0f);
    }


    // RED
    else if ((wavelength>=arr[10]) && (wavelength <= arr[11])){
      attenuation = 0.3f + 0.7f * ((float)arr[11] - wavelength) / (float)(arr[11] - arr[10]);
      color.r = (int)(powf(attenuation, gamma) * 255.0f);

    }
    else
    {
    wavelength = (int)fmaxf((float)wavelength, 1000.0f);
    attenuation = (float)(0.99f * (float)(1000.0f - (float)wavelength) / (float)(1000.0f - arr[11]));
    color.r = (int)attenuation;
    color.r = (int)(powf(attenuation, gamma) * 255.0f);
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
    v.x = (float)sinf(random); v.y = (float)cosf(random);
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
// Integer array values
inline int min_c(int arr[], int n)
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



// C function to find minimum in arr[] of size n
// float array values
inline float min_f(float arr[], unsigned int n)
{
    unsigned int i=0;

    // Initialize minimum element
    float min = arr[0];

    // Find min value from array
    for (i = 1; i < n; i++)
        if (arr[i] < min)
            min = arr[i];
    return min;
}


// C function to find minimum in arr[] of size n
// float array values
inline struct s_min minf_struct(float arr[], unsigned int n)
{
    unsigned int i=0;

    struct s_min s_min_f;
    s_min_f.value = (float)0.0;
    s_min_f.index = (unsigned int)0;

    // Initialize minimum element
    float min = arr[0];

    // Find min value from array
    for (i = 1; i < n; i++)
        if (arr[i] < min)
            min = arr[i];

    s_min_f.value = min;
    s_min_f.index = i;
    return s_min_f;
}



inline unsigned int min_index(float arr[], unsigned int n)
{
    register unsigned int i=0;

    // Initialize minimum element
    register unsigned int index = 0;
    float min = arr[0];

    // Find min value from array
    for (i = 1; i < n; i++)
        if (arr[i] < min){
            min = arr[i];
            index = i;
            }
    return index;
}


int main(){
return 0;
}

//
//int main(){
//
//struct hsl hsl_;
//struct rgb rgb_;
//double h, l, s;
//double r, g, b;
//int i = 0, j = 0, k = 0;
//
//int n = 1000000;
//double *ptr;
//clock_t begin = clock();
//
///* here, do your time-consuming job */
//for (i=0; i<=n; ++i){
//hsl_ = struct_rgb_to_hsl(25.0/255.0, 60.0/255.0, 128.0/255.0);
//}
//clock_t end = clock();
//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//printf("\ntotal time %f :", time_spent);
//
//printf("\nTesting algorithm(s).");
//n = 0;
//
//for (i=0; i<256; i++){
//    for (j=0; j<256; j++){
//        for (k=0; k<256; k++){
//
//            hsl_ = struct_rgb_to_hsl(i/255.0, j/255.0, k/255.0);
//            h = hsl_.h;
//            s = hsl_.s;
//            l = hsl_.l;
//
//            rgb_ = struct_hsl_to_rgb(h, s, l);
//            r = round(rgb_.r * 255.0);
//            g = round(rgb_.g * 255.0);
//            b = round(rgb_.b * 255.0);
//
////            printf("\n\nRGB VALUES:R:%i G:%i B:%i H:%f", i, j, k, h);
////            printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
////            if (s > 1.0f) {
////            printf("\n s>1 %f", s);
////            }
//            if (h > 1.0f) {
//            printf("\n h>1 %f", h);
//            }
//            if (h < 0.0f) {
//            printf("\n h<0.0 %f", h);
//            }
////            if (l > 1.0f) {
////            printf("\n l>1 %f", l);
////            }
//
//
//
//            if (abs(i - r) > 0.1) {
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, l, s);
//                        n+=1;
//                return -1;
//            }
//            if (abs(j - g) > 0.1){
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, l, s);
//                        n+=1;
//                return -1;
//            }
//
//            if (abs(k - b) > 0.1){
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, l, s);
//                n+=1;
//		        return -1;
//
//            }
//
//            }
//
//        }
//    }
//
//
//printf("\nError(s) found n=%i", n);
//return 0;
//}

//

//int main(){
//
//struct hsv hsv_;
//struct rgb rgb_;
//double h, s, v;
//double r, g, b;
//int i = 0, j = 0, k = 0;
//
//int n = 1000000;
//double *ptr;
//clock_t begin = clock();
//
///* here, do your time-consuming job */
//for (i=0; i<=n; ++i){
//hsv_ = struct_rgb_to_hsv(25.0/255.0, 60.0/255.0, 128.0/255.0);
//}
//clock_t end = clock();
//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//printf("\ntotal time %f :", time_spent);
//
//printf("\nTesting algorithm(s).");
//n = 0;
//
//for (i=0; i<256; i++){
//    for (j=0; j<256; j++){
//        for (k=0; k<256; k++){
//
//            hsv_ = struct_rgb_to_hsv(i/255.0, j/255.0, k/255.0);
//            h = hsv_.h;
//            s = hsv_.s;
//            v = hsv_.v;
//            printf("\n\nHSV VALUES:H:%f S:%f V:%f", h, s, v);
//            rgb_ = struct_hsv_to_rgb(h, s, v);
//            r = round(rgb_.r * 255.0);
//            g = round(rgb_.g * 255.0);
//            b = round(rgb_.b * 255.0);
//
////            printf("\n\nRGB VALUES:R:%i G:%i B:%i H:%f", i, j, k, h);
////            printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
////            if (s > 1.0f) {
////            printf("\n s>1 %f", s);
////            }
//            if (h > 1.0f) {
//            printf("\n h>1 %f", h);
//            }
//            if (h < 0.0f) {
//            printf("\n h<0.0 %f", h);
//            }
////            if (l > 1.0f) {
////            printf("\n l>1 %f", l);
////            }
//
//
//
//            if (abs(i - r) > 0.1) {
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, s, v);
//                        n+=1;
//                return -1;
//            }
//            if (abs(j - g) > 0.1){
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, s, v);
//                        n+=1;
//                return -1;
//            }
//
//            if (abs(k - b) > 0.1){
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, s, v);
//                n+=1;
//		        return -1;
//
//            }
//
//            }
//
//        }
//    }
//
//
//printf("\nError(s) found n=%i", n);
//float hh;
//hh = 1.2f;
//hh = (float)fmodf(hh, 1.0f);
//printf("\n%f", hh);
//
//hh = -0.5f;
//hh = (float)fmodf(hh, 1.0f);
//printf("\n%f", hh);
//
//
//hh = 1.2f;
//hh =hh % 1.0f;
//printf("\n%f", hh);
//
//hh = -0.5f;
//hh = hh %1.0f;
//printf("\n%f", hh);
//
//return 0;
//}
