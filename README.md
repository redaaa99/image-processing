# image-processing


* #### int seuillage_otsu(vector<int> histo) ;
  Function that takes as parameter the histogram of an image and
  returns an integer which is the threshold obtained with the method
  Otsu for this picture.
* #### double** calculNoyauGaussien(int s) ;
  The function takes the following arguments:
  s : the radius of the kernel. The kernel will therefore be of size: (s*2+1)* (s*2+1).
  This function returns a matrix that represents the kernel
  Gaussian and whose different boxes have values
  calculated using the standard Gaussian distribution.
* #### Mat lissageGaussien(Mat &img ,int s=1) ;
  The function takes the following arguments:
  img: the image we want to smooth (reduce noise) with a
  Gaussian smoothing.
  s: the radius of the Gaussian kernel, by default equal to 1.
  This function allows for each pixel centered the area on
  which it applies the Gaussian kernel, to modify its
  value according to the values ​​of the neighboring pixels,
  assigned to the kernel coefficients.
  Output: Resulting image of the same size as the input.
  The pixel values ​​of this image are those of the image
  original after applying the Gaussian kernel.
* #### Mat lissageMedian(Mat &img,int r=1);
  The function takes the following arguments:
  img: the image we want to smooth (reduce noise) with a
  Median smoothing.
  r: the radius of the neighborhood of a pixel, by default equal to 1.
  Output: Resulting image of the same size as the input.
  The value of each pixel in this image receives the median value 
  of the corresponding pixel neighborhood in the original image.
* #### Mat FiltreMedian(Mat& img, int r=1) ;
  The same arguments as the previous function except that in
  this function is the difference between two cases.
  If the image is in gray, we return the former function.
  Otherwise, the image is separated into 3 images (one containing the
  values ​​of red, one containing the values ​​of blue and one
  containing the green values). After, we apply the previous function
  for each of these images and finally merge them 
  and return the smoothed image.
* #### Mat FiltreMoyenneur(Mat& src, int r=1 ) ;
  The function takes the following arguments:
  img: the image we want to smooth (reduce noise) with mean smoothing.
  r: the radius of the neighborhood of a pixel, by default equal to 1.
   Output: Resulting image of the same size as the input. 
   The value of each pixel in this image receives the
  average value of pixel values ​​of the pixel neighborhood
  corresponding in the original image.
* #### Mat contourGradient(Mat img , int MasqueX[3][3] , int MasqueY[3][3]) ;
  The function takes the following arguments:
  img: the image on which we want to apply the outline of the
  gradient.
  MasqueX: the suitable mask that measures the variation of the
  value of the pixel in relation to the horizontal direction.
  MasqueY: the mask that measures the variation of the value of the
  pixel relative to the vertical direction.
  If the input image is in color, it is converted to gray.
  For all areas of the image:
  We apply the X mask.
  We apply the YMask.
  The gradient is calculated for the pixel that has the center of
  the area using the relation | G | = sqrt (MaskX2 + MaskY2).
  Output: resulting image of the same size as the original one
  contains the edge points of the input image.
  Masks are applied using the function:
* #### int Calcul(Mat& img, int x , int y , int Masque[3][3]) ;
  The function takes the following arguments:
  Img: the image on which we want to apply the mask.
  x: the horizontal position of the pixel that presents the center
  of the area on which you want to apply the Mask.
  y: the vertical position of the pixel that presents the center of
  the area on which we want to apply the mask.
  Mask [3] [3]: The mask that we want to apply.
* #### Mat contourCanny(Mat img) ;
  The function takes the following arguments:
  img: the image on which we want to apply the outline of the
  canny.
  After calculating the variation of the value of each pixel
  compared to the horizontal and vertical direction, one
  determines the orientation of the edge for each pixel and consequently
  we determine the two neighbors belonging to the edge passing through
  this pixel.
  Output: resulting image of the same size as the original one
  contains edge points that have a local maximum
  in his neighbor.
* #### Mat contourLaplacien(Mat img , int S) ;
  The function takes the following arguments:
  img: the image on which we want to apply the outline of
  The place.
  S: a threshold which has by default the value of the threshold obtained with
  Otsu's method on the image passed in parameter.
  The Laplace mask is first applied to the image (it
  this is the 2nd derivative), after considering all the zones
  of size 3 * 3 of the image. If the center of the area is a
  zero crossing one compares its local variance to the threshold S if the
  threshold S exceeds the local variance, we declare an edge.
  Output: resulting image of the same size as the original one
  contains the defined edge points using the contour of
  The place.
* #### Mat Seuillage(Mat img , int seuil) ;
  The function takes the following arguments:
  img: the image on which we want to apply the outline of the
  canny.
  threshold: a threshold which has by default the value of the threshold obtained
  with the Otsu method on the image passed in parameter.
  if the color image is converted to gray.
  Output: Resulting image of the same size as the original one.
  If in the original image the value of a pixel is greater
  at the threshold where the pixel is bound to a pixel whose value is
  greater than the threshold then the value of this pixel is preserved
  in the resulting image, otherwise it is set to zero in
  the resulting image.
* #### vector<int> CalculHist(Mat image) ;
  The function takes the following arguments:
  img: image in gray for which we want to calculate the
  histogram.
  Output: a vector of length 256 and which contains the number
  occurrence of each value belonging to the interval [0-
  255] in the image.
* #### vector<int> CalculHistNeg(Mat& image) ;
  The function takes the following arguments:
  img: image in gray for which we want to calculate the
  negative histogram.
  Output: a vector of length 256 symmetric to the vector
  presents the histogram of the image in relation to the right
  of equation x = 127.
* #### vector<int> CalculHist(Mat& image, int x) ;
  The function takes the following arguments:
  img: image in gray for which we want to calculate the
  histogram.
  x: integer equal to 0 or 1 or 2, ie it is the image
  which represents the values ​​of blue or red or the
  green in the original image.
  Output: a vector of length 256 and which contains the number
  occurrence of each value belonging to the interval [0-
  255] in the image.
* #### vector<int> CalculHistNeg(Mat& image, int x) ;
  The function takes the following arguments:
  img: image in gray for which we want to calculate the
  negative histogram.
  x: integer equal to 0 or 1 or 2, ie it is the image
  which represents the values ​​of blue or red or the
  green in the original image.
  Output: a vector of length 256 symmetric to the vector
  presents the histogram of the image in relation to the right
  of equation x = 127.
* #### Mat Histogramme(Mat image) ;
  The function takes the following arguments:
  img: the image for which we want to draw the histogram.
  If the image is on gray then:
  Output an image containing a curve that represents the
  vector histogram obtained using the function
* #### vector<int> CalculHistNeg(Mat& image) ;
  If not then:
  Output an image containing three curves. Each of them
  represents the histogram vector obtained using the
  vector function <int> CalculHist (Mat & image, int x); with the
  associated color.
* #### Mat HistogrammeNegatif(Mat image) ;
  The function takes the following arguments:
  img: the image for which we want to draw the histogram.
  If the image is on gray then:
  Output an image containing a curve that represents the
  vector histogram obtained using the functions
* #### vector<int> CalculHistNeg(Mat& image) ;
  If not then:
  Output: an image containing three curves. Each of them
  represents the histogram vector obtained using the
  vector function <int> CalculHistNeg (Mat & image, int x); with
  the associated color.
* #### inline uchar reduire(int n) ;
  Function used in additions of noises to always have
  pixels whose values ​​do not exceed 0 and 255.
* #### Mat Ajouter_bruit_gaussien(cv::Mat src,float moyenne,float ecart_type);
  This function takes as input 3 parameters:
  The first recess will be our source image.
  The second will be the first parameter of the distribution
  Gaussian (the average).
  The third will be the second parameter of the distribution
  Gaussian (the standard deviation)
  We first test if the image is in grayscale or in
  RGB (the treatments will be almost the same for both
  case)
  We fill a matrix with the noise that will be generated
  randomly by a Gaussian distribution and then adds
  the latter has our source image while taking into account that
  our pixels do not have the values ​​0 and 255.
* #### Mat Ajouter_bruit_uniforme(cv::Mat src,float mini,float maxi) ;
  This function takes as input 3 parameters:
  The first will be our source image.
  The 2nd and the 3rd will be the parameters of the distribution
  uniform (min and max).
  It will be on the same treatment as that of the
  Gaussian distribution, the only difference will be the generation
  noise by a uniform distribution.
* #### Mat Ajouter_bruit_sel_poivre(cv::Mat src,float poivre,float sel) ;
  This function takes as input 3 parameters:
  The first will be our source image.
  The second will be the amount of pepper between 0 and 255.
  The third will be the amount of salt between 0 and 255.
  We generate a copy of the source image and we put pixels 0
  (respectively 255) randomly by a distribution
  uniform.
* #### Mat dessiner_histogramme(cv::Mat src) ;
  Only one parameter will be passed to the call of this function which
  will be on the source image.
  The process first checks whether the image is in levels of
  gray or RGB.
  We first separate the RGB channels from the image and then for each
  channel we calculate the histogram that will be put in the vector
  which will be drawn by linking each 2 points
  successive lines.
  Before drawing on the histogram a normalization of the
  vector will be needed since the values ​​of the histogram
  can have a very large variance, in this
  implementation we use the norm min max to normalize
  our histogram.
* #### Mat dessiner_negatif_histogramme(cv::Mat src) ;
  The only parameter passed will be the source image.
  We call the histogram drawing function but in it
  passing as the source the negative of the image passed in parameter.
* #### Mat niveaux_de_gris(cv::Mat src) ;
  This function turns an RGB image into gray.
* #### Mat equalization_histogramme(cv::Mat src) ;
  Equalization of the histogram:
  Only one parameter: source image.
  We go from RGB to YCrCb and we divide the Y and Cr and Cb channels
  since the equalization must be done on the Y channel of the
  luminance.
  We mix the 3 channels and we reconvert our image to the base
  RGB.
* #### Mat binarisation(int seuil,cv::Mat src) ;
  Function that takes as argument a threshold and the source image then
  returns a binarized image.
* #### Mat binarisation_otsu(Mat src) ;
  Binarization based on a threshold found by the method
  Otsu.
* #### Mat segmenter_image(Mat src,int k) ;
  This function takes as parameter the source image and an integer
  K.
  The treatment is as follows:
  We look for the K most dominant pixels in the image in
  using the K-means method and we are redesigning our
  image using only these K colors
  N.B: This function and after many optimizations takes a
  considerable time to run depending on the number of
  K colors of the segmentation and the size of the image.
* #### vector<int> encode_RLE(vector<int> str) ;
  Takes an input vector and returns its RLE encoding
  vector.
* #### vector<int> decode_RLE(vector<int> str) ;
  RLE decoding.
* #### Mat calculDCT(cv::Mat oo) ;
  Calculation of the discrete cosine transform of a picture block
  of size 8x8.
* #### Mat calculiDCT(cv::Mat oo) ;
  Calculation of the discrete inverse cosine transform of a block
  8x8 size image.
* #### Mat compression_dct(cv::Mat src) ;
  Takes a source image as parameter and returns an image with
  reduced quality simulating compression then
  DCT decompression of a jpeg image taking into account if
  the image is in RGB or gray level.
  HUFFMAN COMPRESSION
  Define the tree used in compression
  Huffman:
struct Arbre
{
 int val;
 int occ;
 Arbre *fd, *fg;
 Arbre(int valeur, int occurence)
 {
 fg = fd = NULL;
 this->val = valeur;
 this->occ = occurence;
 }
};
Définition de la structure utilisée pour la comparaison
dans : priority_queue<Arbre*, vector<Arbre*>, comparer>

struct comparer
{
 bool operator()(Arbre* l, Arbre* r)
 {
 return (l->occ > r->occ);
 }
};
* #### void HuffmanAr(vector<int> vect,priority_queue<Arbre*,vector<Arbre*>, comparer>& mafile) ;
The function takes the following arguments:
vect: a vector of integers.
mafile: priority queue to store the heap, compared to the
value of the root node.
This function allows you to build the Huffman tree and the
store in mafile.
* #### void HuffmanArbre(Mat & image,priority_queue<Arbre*,vector<Arbre*>, comparer>& mafile) ;
The function takes the following arguments:
image: image in gray.
mafile: priority queue to store the heap, compared to the
value of the root node.
First, a histogram vector of the named image is calculated
result. Then we call the function
HuffmanAr (matches, mafile).
This function is used to determine the Huffman tree for
a gray image.
* #### void codage(struct Arbre* monarbre, string str,vector<string> & vect) ;
The function takes the following arguments:
tree: Huffman tree.
str: an empty string for easy reading.
vect: a length vector 256 contains strings of
characters .
This function allows reading of the Huffman tree and
for each index of the vector vect it will be stored the code
associated.
* #### string CodageHuffmanseulniveau(Mat image,priority_queue<Arbre*, vector<Arbre*>, comparer> &file) ;
  The function takes the following arguments:
  image: image in gray.
  file: priority queue to store the heap, relative to the
  value of the root node.
  In this function we use the functions
  HuffmanArbre (image, file);
  and encoding (file.top (), "", result);
  and from the result vector and flowing the image
  original pixel by pixel we determine the Huffman code of
  this image .
  Output: a string containing the Huffman code and
  the length and width of the original image.
* #### vector<string> CodageHuffman(Mat image,priority_queue<Arbre*, vector<Arbre*>, comparer> &file,priority_queue<Arbre*, vector<Arbre*>, comparer> &file1=mafile1,priority_queue<Arbre*, vector<Arbre*>,comparer> & file2= mafile2) ;
  The function takes the following arguments:
  image: original image.
  file: priority queue to store the heap, relative to the
  value of the root node.
  file1: priority queue to store the heap, relative to the
  value of the root node.
  file2: priority queue to store the heap, relative to the
  value of the root node.
  file2 and file3 are used for the case of an image in
  color.
  If the image in gray then:
  Output: a string vector contains the Huffman code of
  the image.
  Otherwise: We divide the image into 3 images (one containing them
  values ​​of red, one containing the values ​​of blue and one
  containing the green values). After, we apply the function
  Huffman coding for each of these images
  Output: a string vector containing three Huffman codes
  for each of the three color levels of the image.
* #### Mat DecodageHuffmanunniveau(string Huffmancode,struct Arbre* Ar) :
  The function takes the following arguments:
  Huffmancode: The Huffman code.
  Ar: The Huffman tree.
  Output: resulting image obtained after reading the code
  Huffman using the Huffman tree.
* #### Mat DecodageHuffman(vector<string> Huffmancode,struct Arbre* Arbre1 , struct Arbre* Arbre2 = Ar1, struct Arbre* Arbre3 = Ar1) ;
  The function takes the following arguments:
  Huffmancode: Huffman code.
  Tree1: Huffman tree
  Tree2: Huffman tree
  Tree3: Huffman tree
  Tree2 and Tree3 are used for the case of an image in
  color.
  If the image in gray then:
  Output: resulting image obtained after reading the code
  Huffman using the Huffman tree by calling the function
  DecodageHuffmanunniveau (Huffmancode [0], Arbre1);
  Otherwise: We apply the DecodingHuffmannumber function for
  each of these code and these trees. We get the image
  resultant of each color level (blue, red and green)
  then they are merged into the resulting image.
  Output: resulting color image obtained after reading
  of the Huffman code using the Huffman tree.
