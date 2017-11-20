#include <iostream>
#include "TraitementImage.hpp"
using namespace std;
using namespace cv;

int main(){

    priority_queue<Arbre*, vector<Arbre*>, comparer>  mafile;
    priority_queue<Arbre*, vector<Arbre*>, comparer>  mafile1;
    priority_queue<Arbre*, vector<Arbre*>, comparer>  mafile2;
    
    Mat src = imread("/Users/mac/Desktop/lena.jpg");
    imshow("Original",src);
    //imshow("Lissage Gaussien",lissageGaussien(src));
    //imshow("Contour de Canny", contourCanny(src));
    //imshow("Contour de Sobel", contourSobel(src));
    //imshow("Contour de Prewit", contourPrewit(src));
    //imshow("Contour de Robert", contourRobert(src));
    //imshow("Contour de Laplace", contourLaplacien(src));
    //imshow("Seuillage", Seuillage(contourCanny(src)));
    //imshow("Filtre Mediane", FiltreMedian(src,1));
   // imshow("Filtre Moyenneur", FiltreMoyenneur(src,1));
    //imshow("Compression Huffman", DecodageHuffman(CodageHuffman(src, mafile,mafile1,mafile2),mafile.top(),mafile1.top(),mafile2.top()));
   // imshow("Histogramme version1",Histogramme(src));
   // imshow("Histogramme Nergative version1",HistogrammeNegatif(src));
    //imshow("Bruit Gaussien",Ajouter_bruit_gaussien(src,10,30));
   // imshow("Bruit uniforme",Ajouter_bruit_uniforme(src,20,50));
   // imshow("Bruit sel et poivre",Ajouter_bruit_uniforme(src,10,250));
    //imshow("histogramme version opencv",dessiner_histogramme(src));
    //imshow("Equalization histogramme",equalization_histogramme(src));
    //imshow("Negatif histogramme",dessiner_negatif_histogramme(src));
   // imshow("Binarisation Otsu",binarisation_otsu(src));
   // imshow("Niveaux de Gris",niveaux_de_gris(src));
    //imshow("Segmenter image",segmenter_image(src,5));
    //imshow("Compression DCT",compression_dct(src));
    
    waitKey(0);
    return 0;
}

















