//
//  TraitementImage.hpp
//  Opencv-Projet
//
//  Created by Youness Lagrini on 02/08/2017.
//  Copyright © 2017 Youness Lagrini. All rights reserved.
//

#ifndef TraitementImage_hpp
#define TraitementImage_hpp
#include <iostream>
#include <string>
#include<cmath>
#include <vector>
#include <algorithm>
#include <queue>
#include <fstream>
#include <sstream>
#include <numeric>
#include <limits.h>
#include <errno.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

double** calculNoyauGaussien(int s=1);
Mat lissageGaussien(Mat &img ,int  s=1);
int Calcul(Mat& img, int x , int y , int Masque[3][3]);
Mat contourGradient(Mat img , int MasqueX[3][3] , int MasqueY[3][3]);
Mat contourSobel(Mat image);
Mat contourRobert(Mat image);
Mat contourPrewit(Mat image);
Mat contourCanny(Mat img);
Mat contourLaplacien(Mat img , int S=-1);
Mat Seuillage(Mat img , int seuil=-1);
Mat lissageMedian(Mat &img,int r=1);
Mat FiltreMedian(Mat& img, int r=1);
Mat FiltreMoyenneur(Mat& src, int r=1);
//Calcul Histogramme
vector<int> CalculHist(Mat& image, int x);
vector<int> CalculHist(Mat image);
vector<int> CalculHistNeg(Mat& image, int x);
vector<int> CalculHistNeg(Mat& image);
//Histogramme Version1
Mat Histogramme(Mat image);
Mat HistogrammeNegatif(Mat image);
//Histogramme Version Opencv
cv::Mat dessiner_histogramme(cv::Mat src);
cv::Mat dessiner_negatif_histogramme(cv::Mat src);
cv::Mat niveaux_de_gris(cv::Mat src);
//Equalisation Histogramme
cv::Mat equalization_histogramme(cv::Mat src);
//Bruit
inline uchar reduire(int n);
cv::Mat Ajouter_bruit_gaussien(cv::Mat src,float moyenne,float ecart_type);
cv::Mat Ajouter_bruit_uniforme(cv::Mat src,float mini,float maxi);
cv::Mat Ajouter_bruit_sel_poivre(cv::Mat src,float mini,float maxi);
//Calcul seuil avec methode d'otsu
int seuillage_otsu(vector<int> histo);
//binarisation
cv::Mat binarisation(int seuil,cv::Mat src);
cv::Mat binarisation_otsu(Mat src);
cv::Mat segmenter_image(Mat src,int k);
//RLE
vector<int> encode_RLE(vector<int> str);
vector<int> decode_RLE(vector<int>  str);
//DCT
cv::Mat calculDCT(cv::Mat oo);
cv::Mat calculiDCT(cv::Mat oo);
cv::Mat dct_idct(cv::Mat src);
cv::Mat compression_dct(cv::Mat src);
//Huffman
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
struct comparer
{
    bool operator()(Arbre* l, Arbre* r)
    {
        return (l->occ > r->occ);
    }
};
struct comparer;
void codage(struct Arbre* monarbre, string str,vector<string> & vect);
void HuffmanAr(vector<int> vect,priority_queue<Arbre*, vector<Arbre*>, comparer>& mafile);
void HuffmanArbre(Mat & image,priority_queue<Arbre*, vector<Arbre*>, comparer>& mafile);
string CodageHuffmanseulniveau(Mat image,priority_queue<Arbre*, vector<Arbre*>, comparer> & file);
vector<string> CodageHuffman(Mat image,priority_queue<Arbre*, vector<Arbre*>, comparer> & file,priority_queue<Arbre*, vector<Arbre*>, comparer> & file1,priority_queue<Arbre*, vector<Arbre*>, comparer> & file2);
Mat DecodageHuffmanunniveau(string Huffmancode,struct Arbre* Ar);
Mat DecodageHuffman(vector<string> Huffmancode,struct Arbre* Arbre1 , struct Arbre* Arbre2, struct Arbre* Arbre3);




















#endif /* TraitementImage_hpp */
