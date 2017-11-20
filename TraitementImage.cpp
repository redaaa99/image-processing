#include "TraitementImage.hpp"


using namespace cv;
using namespace std;


Mat matImageSource;
Mat monImageGris,monImageContour,monImageSeuil;

string window_name = "final result";

//les Masques

int SobelX[3][3] ={{-1,0,1},{-2,0,2},{-1,0,1}} ;
int SobelY[3][3] ={{1,2,1},{0,0,0},{-1,-2,-1}} ;
int PrewitX[3][3] ={{-1,0,1},{-1,0,1},{-1,0,1}} ;
int PrewitY[3][3] ={{1,1,1},{0,0,0},{-1,-1,-1}} ;
int RobertX[3][3] ={{1,0,0},{0,-1,0},{0,0,0}} ;
int RobertY[3][3] ={{0,1,0},{-1,0,0},{0,0,0}} ;
int laplace[3][3] ={{0,1,0},{1,-4,1},{0,1,0}} ;

//Calcul Masque * Zone

int Calcul(Mat& img, int x , int y , int Masque[3][3])
{
    return  img.at<uchar>(x-1,y-1)*Masque[0][0] +  img.at<uchar>(x-1,y)*Masque[0][1] +  img.at<uchar>(x-1,y+1)*Masque[0][2] + img.at<uchar>(x,y-1)*Masque[1][0] +  img.at<uchar>(x,y)*Masque[1][1] +  img.at<uchar>(x,y+1)*Masque[1][2] + img.at<uchar>(x+1,y-1)*Masque[2][0] +  img.at<uchar>(x+1,y)*Masque[2][1] +  img.at<uchar>(x+1,y+1)*Masque[2][2];
}

//Gradient
Mat contourGradient(Mat img , int MasqueX[3][3] , int MasqueY[3][3])
{
    if(img.channels()!=1) cvtColor(img,img,CV_RGB2GRAY);
    Mat contourImage = img.clone();
    for(int i=1; i < img.rows-1; i++)
    {
        for(int j=1; j < img.cols-1; j++)
        {
            int GradientX = Calcul(img, i , j ,MasqueX);
            int GradientY = Calcul(img, i , j ,MasqueY);
            
            contourImage.at<uchar>(i,j) = sqrt(abs(GradientX)*abs(GradientX) + abs(GradientY)*abs(GradientY));
        }
    }
    return contourImage;
}
Mat contourSobel(Mat image)
{
    return contourGradient( image , SobelX , SobelY);
}
Mat contourRobert(Mat image)
{
    return contourGradient( image , RobertX , RobertY);
}
Mat contourPrewit(Mat image)
{
    return contourGradient( image , PrewitX , PrewitY);
}

//Canny

Mat contourCanny(Mat img)
{
    if(img.channels()!=1) cvtColor(img,img,CV_RGB2GRAY);
    
    Mat grad_x, grad_y ;
    Mat Grad_x, Grad_y;
    
    Sobel(img,grad_x,CV_16S,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x,Grad_x);
    Sobel(img,grad_y,CV_16S,0,1,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_y,Grad_y);
    
    addWeighted( Grad_x,0.5,Grad_y,0.5,0,img);
    
    int ligne   = img.rows;
    int colonne = img.cols;
    
    for(int i=1 ; i<ligne-1 ;i++)
    {
        for(int j =1; j<colonne-1 ; j++)
        {
            double theta;
            theta = atan2(Grad_y.at<uchar>(i,j), Grad_x.at<uchar>(i,j))*(180/3.14);
            
            if(((-22.5 < theta) && (theta <= 22.5)) || ((157.5 < theta) && (theta <= -157.5)))
            {
                if ((img.at<uchar>(i,j) < img.at<uchar>(i,j+1)) || (img.at<uchar>(i,j) < img.at<uchar>(i,j-1)))
                    img.at<uchar>(i, j) = 0;
            }
            
            if (((-112.5 < theta) && (theta <= -67.5)) || ((67.5 < theta) && (theta <= 112.5)))
            {
                if ((img.at<uchar>(i,j) < img.at<uchar>(i+1,j)) || (img.at<uchar>(i,j) < img.at<uchar>(i-1,j)))
                    img.at<uchar>(i, j) = 0;
            }
            
            if (((-67.5 < theta) && (theta <= -22.5)) || ((112.5 < theta) && (theta <= 157.5)))
            {
                if ((img.at<uchar>(i,j) < img.at<uchar>(i+1,j+1)) || (img.at<uchar>(i,j) < img.at<uchar>(i-1,j-1)))
                    img.at<uchar>(i, j) = 0;
            }
            
            if (((-157.5 < theta) && (theta <= -112.5)) || ((22.5 < theta) && (theta <= 67.5)))
            {
                if ((img.at<uchar>(i,j) < img.at<uchar>(i-1,j+1)) || (img.at<uchar>(i,j) < img.at<uchar>(i+1,j-1)))
                    img.at<uchar>(i, j) = 0;
            }
        }
    }
    return img;
}

//Laplacien
Mat contourLaplacien(Mat img , int S)
{
    if(S<0 || S>255) S = seuillage_otsu(CalculHist(img));
    if(img.channels()!=1) cvtColor(img,img,CV_RGB2GRAY);
    Mat contourImage = img.clone();
    
    for(int i = 1; i < img.rows - 1; i++)
    {
        for(int j = 1; j< img.cols - 1; j++)
        {
            contourImage.at<uchar>(i,j) = Calcul(img, i, j, laplace);
        }
    }
    
    for(int i = 1; i < img.rows - 1; i++)
    {
        for(int j = 1; j< img.cols - 1; j++)
        {
            
            if(contourImage.at<uchar>(i,j)==0)
            {
                int maxi = max(contourImage.at<uchar>(i,j)-contourImage.at<uchar>(i-1,j-1),max(contourImage.at<uchar>(i,j)-contourImage.at<uchar>(i-1,j),max(contourImage.at<uchar>(i,j)-contourImage.at<uchar>(i-1,j+1),max(contourImage.at<uchar>(i,j)-contourImage.at<uchar>(i,j+1),max(contourImage.at<uchar>(i,j)-contourImage.at<uchar>(i,j-1),max(contourImage.at<uchar>(i,j)-contourImage.at<uchar>(i+1,j-1),max(contourImage.at<uchar>(i,j)-contourImage.at<uchar>(i+1,j),contourImage.at<uchar>(i,j)-contourImage.at<uchar>(i+1,j+1))))))));
                if(maxi<S)  contourImage.at<uchar>(i,j)=0;
            }
        }
    }
    
    return contourImage;
}
//Seuillage
Mat Seuillage(Mat img , int seuil)
{
    if(seuil<0 || seuil>255) seuil = seuillage_otsu(CalculHist(img));
    int ligne   = img.rows;
    int colonne = img.cols;
    if(img.channels()!=1) cvtColor(img, img, CV_BGR2GRAY);
    Mat imageSeuil(ligne,colonne,CV_8UC1,Scalar(0,0,0));
    
    for(int i=0 ; i<ligne ; i++)
    {
        for(int j=0 ; j<colonne ; j++)
            imageSeuil.at<uchar>(i,j)=img.at<uchar>(i,j);
    }
    
    for(int i=0 ; i<ligne ; i++)
    {
        for(int j=0 ; j<colonne ; j++)
        {
            if(imageSeuil.at<uchar>(i,j)<seuil)
            {
                if( (imageSeuil.at<uchar>(i-1,j-1)<=seuil) && (imageSeuil.at<uchar>(i-1,j)<=seuil)
                   && (imageSeuil.at<uchar>(i-1,j+1)<=seuil) && (imageSeuil.at<uchar>(i,j-1)<=seuil)
                   && (imageSeuil.at<uchar>(i,j+1)<=seuil) && (imageSeuil.at<uchar>(i+1,j-1)<=seuil)
                   && (imageSeuil.at<uchar>(i+1,j)<=seuil) && (imageSeuil.at<uchar>(i+1,j+1)<=seuil))
                    imageSeuil.at<uchar>(i,j) = 0;
            }
        }
    }
    return imageSeuil;
}
//Calcul du noyau gaussien
double** calculNoyauGaussien(int s)
{
    if(s<=0) s=1;
    double** noyeau = new double* [2*s+1];
    for(int i=0 ; i<2*s+1 ; i++) noyeau[i] = new double [2*s+1];
    double sum(0.0);
    for(int x=-s ; x<=s ; x++)
    {
        for(int y=-s ; y<=s ; y++)
        {
            double radius = sqrt(x*x+y*y);
            noyeau[x+s][y+s] = (1/sqrt(2*M_PI))*exp(-(radius*radius)/2);
            sum +=noyeau[x+s][y+s];
        }
    }
    for(int i=0 ; i<s*2+1 ; i++)
    {
        for(int j=0 ; j<s*2+1 ; j++)
        {
            noyeau[i][j] /= sum;
        }
    }
    return noyeau;
}

//Gaussien
Mat lissageGaussien(Mat &img ,int  s)
{
    int ligne = img.rows;
    int colonne = img.cols;
    if(s<=0) s=1;
    double** Filtre = calculNoyauGaussien(s);
    if(img.channels()==1)
    {
        Mat imageLissee(ligne,colonne,CV_8UC1,Scalar(0,0,0));
        for(int i=0 ; i<ligne ; i++)
        {
            for(int j=0 ; j<colonne ; j++)
            {
                if( (j-s<0) || (j+s>=colonne) || (i-s<0) || (i+s>=ligne) ) imageLissee.at<uchar>(i,j) = img.at<uchar>(i,j);
                else
                {
                    imageLissee.at<uchar>(i,j)=0;
                    for(int x=-s ; x<=s ; x++)
                    {
                        for(int y=-s ; y<=s ; y++)
                            imageLissee.at<uchar>(i,j) += img.at<uchar>(i+x,j+y) * Filtre[x+s][y+s];
                    }
                }
            }
        }
        return imageLissee;
    }
    else
    {
        Mat imageLissee(ligne,colonne,CV_8UC3,Scalar(0,0,0));
        for(int i=0 ; i<ligne ; i++)
        {
            for(int j=0 ; j<colonne ; j++)
            {
                if( (j-s<0) || (j+s>=colonne) || (i-s<0) || (i+s>=ligne) ) imageLissee.at<Vec3b>(i,j) = img.at<Vec3b>(i,j);
                else
                {
                    imageLissee.at<Vec3b>(i,j)=0;
                    for(int x=-s ; x<=s ; x++)
                    {
                        for(int y=-s ; y<=s ; y++)
                            imageLissee.at<Vec3b>(i,j) += img.at<Vec3b>(i+x,j+y) * Filtre[x+s][y+s];
                    }
                }
            }
        }
        return imageLissee;
    }
}
/// Mediane
Mat lissageMedian(Mat &img,int r)
{
    int ligne = img.rows;
    int colonne = img.cols;
    if(r<1 || r>ligne || r>colonne) r=1;
    Mat imageLissee(ligne,colonne,CV_8UC1,Scalar(0,0,0));
    for(int i=0 ; i<ligne ; i++)
    {
        for(int j=0 ; j<colonne ; j++)
        {
            if( (j-r<0) || (j+r>=colonne) || (i-r<0) || (i+r>=ligne) )
                imageLissee.at<uchar>(i,j) = img.at<uchar>(i,j);
            else
            {
                vector<int> tab;
                for(int x=-r ; x<=r ; x++)
                {
                    for(int y=-r ; y<=r ; y++)
                        tab.push_back(img.at<uchar>(i+x,j+y));
                }
                sort(tab.begin(),tab.end());
                imageLissee.at<uchar>(i,j) = tab[(r*2+1)/2];
            }
        }
    }
    
    return imageLissee;
}

Mat FiltreMedian(Mat& img, int r)
{
    if(img.channels()==1) return lissageMedian(img,r);
    else
    {
        vector<Mat> rgb(3);
        Mat Result;
        split(img, rgb);
        rgb[0]= lissageMedian(rgb[0],r);
        rgb[1]= lissageMedian(rgb[1],r);
        rgb[2]= lissageMedian(rgb[2],r);
        merge(rgb, Result);
        return Result;
    }
}
//// Moyenneur
Mat FiltreMoyenneur(Mat& src, int r)
{
    int ligne = src.rows;
    int col = src.cols;
    if(r<1 || r>ligne || r>col) r=1;
    int taille = (2*r+1)*(2*r+1);
    if(src.channels()==1)
    {
        Mat Resultat(ligne,col,CV_8UC1,Scalar(0,0,0));
        
        for(int i=0 ; i<ligne ; i++)
        {
            for(int j=0 ; j<col ; j++)
            {
                for(int x=-r ; x<=r ; x++)
                {
                    for(int y=-r ; y<=r ; y++)
                    {
                        if((i+x<0) || (i+x>=ligne) || (j+y<0) || (j+y>=col))
                        {
                            if((i+x<0) || (i+x>=ligne)) x++;
                            else y++;
                        }
                        else Resultat.at<uchar>(i,j) += src.at<uchar>(i+x,j+y)/taille;
                    }
                }
            }
        }
        return Resultat;
    }
    else
    {
        Mat Resultat(ligne,col,CV_8UC3,Scalar(0,0,0));
        
        for(int i=0 ; i<ligne ; i++)
        {
            for(int j=0 ; j<col ; j++)
            {
                for(int x=-r ; x<=r ; x++)
                {
                    for(int y=-r ; y<=r ; y++)
                    {
                        if((i+x<0) || (i+x>=ligne) || (j+y<0) || (j+y>=col))
                        {
                            if((i+x<0) || (i+x>=ligne)) x++;
                            else y++;
                        }
                        else Resultat.at<Vec3b>(i,j) += src.at<Vec3b>(i+x,j+y)/taille;
                    }
                }
            }
        }
        return Resultat;
    }
}


/// COMPRESSION HUFFMAN

//***********************

priority_queue<Arbre*, vector<Arbre*>, comparer>  mafile;
priority_queue<Arbre*, vector<Arbre*>, comparer>  mafile1;
priority_queue<Arbre*, vector<Arbre*>, comparer>  mafile2;

Arbre* Ar1 = new Arbre(0, 0);
//**********************
void codage(struct Arbre* monarbre, string str,vector<string> & vect)
{
    if (!monarbre)
        return;
    
    if (monarbre->val != 300)
    {
        vect[monarbre->val] = new char (str.length());
        vect[monarbre->val]=str;
    }
    codage(monarbre->fg, str + "0" ,vect);
    codage(monarbre->fd, str + "1" ,vect);
}


void HuffmanAr(vector<int> vect,priority_queue<Arbre*, vector<Arbre*>, comparer>& mafile)
{
    struct Arbre *fd, *fg, *top;
    
    for (int i = 0; i < vect.size(); ++i)
        mafile.push(new Arbre(i, vect[i]));
    
    while (mafile.size() != 1)
    {
        fg = mafile.top();
        mafile.pop();
        fd = mafile.top();
        mafile.pop();
        top = new Arbre(300, fg->occ + fd->occ);
        top->fg = fg;
        top->fd = fd;
        mafile.push(top);
    }
}

void HuffmanArbre(Mat & image,priority_queue<Arbre*, vector<Arbre*>, comparer>& mafile) //RLE AUSSI
{
    vector<int> result(256);
    fill(result.begin(),result.end(),0);
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0 ;j<image.cols ; j++)
            result[image.at<uchar>(i,j)]++;
    }
    
    HuffmanAr(result,mafile);
}

string CodageHuffmanseulniveau(Mat image,priority_queue<Arbre*, vector<Arbre*>, comparer> & file)
{
    string Result="";
    vector<string> resultat(256);
    HuffmanArbre(image,file);
    codage(file.top(), "" , resultat);
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0 ;j<image.cols ; j++)
        {
            Result += resultat[image.at<uchar>(i,j)];
        }
    }
    Result += ':' + to_string(image.rows) + ':' + to_string(image.cols);
    return Result;
}

vector<string> CodageHuffman(Mat image,priority_queue<Arbre*, vector<Arbre*>, comparer> & file,priority_queue<Arbre*, vector<Arbre*>, comparer> & file1=mafile1,priority_queue<Arbre*, vector<Arbre*>, comparer> & file2= mafile2)
{
    vector<string> Resultat;
    if(image.channels()==1) Resultat.push_back(CodageHuffmanseulniveau(image,file));
    else
    {
        vector<Mat> bgr(3);
        split(image, bgr);
        Resultat.push_back(CodageHuffmanseulniveau(bgr[0],file));
        Resultat.push_back(CodageHuffmanseulniveau(bgr[1],file1));
        Resultat.push_back(CodageHuffmanseulniveau(bgr[2],file2));
    }
    return Resultat;
}

/////DECOMPRESSION

Mat DecodageHuffmanunniveau(string Huffmancode,struct Arbre* Ar)
{
    int i = Huffmancode.length()-1;
    string colonne = "";
    string ligne  = "";
    while(Huffmancode[i]!=':')
    {
        colonne = Huffmancode[i] + colonne;
        Huffmancode.pop_back();
        i--;
    }
    Huffmancode.pop_back();
    i--;
    while(Huffmancode[i]!=':')
    {
        ligne = Huffmancode[i] + ligne;
        Huffmancode.pop_back();
        i--;
    }
    Huffmancode.pop_back();
    Mat resultat(stoi(ligne),stoi(colonne),CV_8UC1);
    int l(0) , c(0);
    
    Arbre* node = Ar;
    
    for (int i = 0; i != Huffmancode.size(); i++)
    {
        if (Huffmancode[i] == '0')
        {
            node = node->fg;
        }
        else
        {
            node = node->fd;
        }
        if (node->val != 300)
        {
            resultat.at<uchar>(l,c) =node->val;
            c++;
            node = Ar;
        }
        if(c>=stoi(colonne))
        {
            c=0; l++;
        }
        if(l>stoi(ligne) || c>stoi(colonne)) break;
    }
    return resultat;
}
Mat DecodageHuffman(vector<string> Huffmancode,struct Arbre* Arbre1 , struct Arbre* Arbre2 = Ar1, struct Arbre* Arbre3 = Ar1)
{
    if(Huffmancode.size()==1) return DecodageHuffmanunniveau(Huffmancode[0],Arbre1);
    else
    {
        Mat Resultat;
        vector<Mat> bgr(3);
        bgr[0] = DecodageHuffmanunniveau(Huffmancode[0],Arbre1);
        bgr[1] = DecodageHuffmanunniveau(Huffmancode[1],Arbre2);
        bgr[2] = DecodageHuffmanunniveau(Huffmancode[2],Arbre3);
        merge(bgr, Resultat);
        return Resultat;
    }
}
//Histogramme V1
vector<int> CalculHist(Mat& image, int x) // 0 1 ou 2 pour x
{
    vector<int> result(256);
    fill(result.begin(),result.end(),0);
    
    for(int i=0 ; i<image.rows ; i++)
    {
        for(int j=0; j<image.cols ;j++)
        {
            result[image.at<Vec3b>(i,j).val[x]]++ ;
        }
    }
    return result;
}

vector<int> CalculHist(Mat image)
{
    if(image.channels()!=1) cvtColor(image,image,CV_RGB2GRAY);;
    vector<int> result(256);
    fill(result.begin(),result.end(),0);
    
    for(int i=0 ; i<image.rows ; i++)
    {
        for(int j=0; j<image.cols ;j++)
        {
            result[image.at<uchar>(i,j)]++ ;
        }
    }
    return result;
}
vector<int> CalculHistNeg(Mat& image, int x) // 0 1 ou 2 pour x
{
    vector<int> result(256);
    fill(result.begin(),result.end(),0);
    
    for(int i=0 ; i<image.rows ; i++)
    {
        for(int j=0; j<image.cols ;j++)
        {
            result[255 - image.at<Vec3b>(i,j).val[x]]++ ;
        }
    }
    return result;
}

vector<int> CalculHistNeg(Mat& image)
{
    
    vector<int> result(256);
    fill(result.begin(),result.end(),0);
    
    for(int i=0 ; i<image.rows ; i++)
    {
        for(int j=0; j<image.cols ;j++)
        {
            result[255 - image.at<uchar>(i,j)]++ ;
        }
    }
    return result;
}


Mat Histogramme(Mat image)
{
    
    int colonne = 512;
    int ligne = 400;
    int largeur = cvRound((double) colonne/256);
    if(image.channels()==1)
    {
        vector<int> histogram = CalculHist(image);
        
        int maxi = histogram[0];
        for(int i = 1; i < 256; i++)  maxi = max(maxi , histogram[i]);
        
        for(int i = 0; i < 256; i++){
            histogram[i] = ((double)histogram[i]/maxi)*ligne;
        }
        
        Mat resultat(ligne, colonne, CV_8UC1,Scalar(0,0,0));
        for(int i = 1; i < 256; i++)
        {
            line(resultat, Point(largeur*(i-1),ligne - histogram[i-1]),Point(largeur*(i), ligne - histogram[i]),Scalar(255,255,255));
        }
        
        return resultat;
    }
    else
    {
        Mat resultat(ligne, colonne, CV_8UC3,Scalar(0,0,0));
        int v[3][3]={{255,0,0},{0,255,0},{0,0,255}};
        for(int i=0 ; i<3 ; i++)
        {
            vector<int> histogram = CalculHist(image,i);
            
            //l'intensité maximale
            int maxi = histogram[0];
            for(int i = 1; i < 256; i++)  maxi = max(maxi , histogram[i]);
            //normalisation
            
            for(int i = 0; i < 256; i++){
                histogram[i] = ((double)histogram[i]/maxi)*ligne;
            }
            
            for(int j = 1; j < 256; j++)
            {
                line(resultat, Point(largeur*(j-1),ligne - histogram[j-1]),Point(largeur*(j), ligne - histogram[j]),Scalar(v[i][0],v[i][1],v[i][2]));
            }
            
        }
        return resultat;
    }
}
Mat HistogrammeNegatif(Mat image)
{
    
    int colonne = 512;
    int ligne = 400;
    int largeur = cvRound((double) colonne/256);
    if(image.channels()==1)
    {
        vector<int> histogram = CalculHistNeg(image);
        
        int maxi = histogram[0];
        for(int i = 1; i < 256; i++)  maxi = max(maxi , histogram[i]);
        
        for(int i = 0; i < 256; i++){
            histogram[i] = ((double)histogram[i]/maxi)*ligne;
        }
        
        Mat resultat(ligne, colonne, CV_8UC1,Scalar( 0,0,0));
        for(int i = 1; i < 256; i++)
        {
            line(resultat, Point(largeur*(i-1),ligne - histogram[i-1]),Point(largeur*(i), ligne - histogram[i]),Scalar(255,255,255));
        }
        
        return resultat;
    }
    
    else
    {
        Mat resultat(ligne, colonne, CV_8UC3,Scalar(0,0,0));
        int v[3][3]={{255,0,0},{0,255,0},{0,0,255}};
        for(int i=0 ; i<3 ; i++)
        {
            vector<int> histogram = CalculHistNeg(image,i);
            
            //l'intensité maximale
            int maxi = histogram[0];
            for(int i = 1; i < 256; i++)  maxi = max(maxi , histogram[i]);
            //normalisation
            
            for(int i = 0; i < 256; i++){
                histogram[i] = ((double)histogram[i]/maxi)*ligne;
            }
            
            for(int j = 1; j < 256; j++)
            {
                line(resultat, Point(largeur*(j-1),ligne - histogram[j-1]),Point(largeur*(j), ligne - histogram[j]),Scalar(v[i][0],v[i][1],v[i][2]));
            }
        }
        return resultat;
    }
}

// Bruiit
inline uchar reduire(int n)
{
    n = n>255 ? 255 : n;
    return n<0 ? 0 : n;
}

cv::Mat Ajouter_bruit_gaussien(cv::Mat src,float moyenne,float ecart_type)
{
    RNG rng;
    if(src.channels()==1)
    {
        Mat dst(src.rows,src.cols,CV_8UC1);
        Mat bruitmat = src.clone();
        rng.fill(bruitmat,RNG::NORMAL,moyenne,ecart_type);
        //addWeighted(src, 1.0, bruitmat, 1.0, 0.0, src); peut etre pour un version plus optimisÈe
        for (int Rows = 0; Rows < src.rows; Rows++)
        {
            for (int Cols = 0; Cols < src.cols; Cols++)
            {
                dst.at<uchar>(Rows,Cols)= reduire(src.at<uchar>(Rows,Cols)+bruitmat.at<uchar>(Rows,Cols));
            }
        }
        return dst;
    }
    Mat dst(src.rows,src.cols,CV_8UC3);
    Mat bruitmat = src.clone();
    rng.fill(bruitmat,RNG::NORMAL,moyenne,ecart_type);
    //addWeighted(src, 1.0, bruitmat, 1.0, 0.0, src); peut etre pour un version plus optimisÈe
    for (int Rows = 0; Rows < src.rows; Rows++)
    {
        for (int Cols = 0; Cols < src.cols; Cols++)
        {
            for (int i = 0; i < 3; i++)
            {
                dst.at<Vec3b>(Rows,Cols).val[i]= reduire(src.at<Vec3b>(Rows,Cols).val[i] + bruitmat.at<Vec3b>(Rows,Cols).val[i]);
            }
        }
    }
    return dst;
}

cv::Mat Ajouter_bruit_uniforme(cv::Mat src,float mini,float maxi)
{
    RNG rng;
    if(src.channels()==1)
    {
        Mat dst(src.rows,src.cols,CV_8UC1);
        Mat bruitmat = src.clone();
        rng.fill(bruitmat,RNG::UNIFORM,mini,maxi);
        //addWeighted(src, 1.0, bruitmat, 1.0, 0.0, src); peut etre pour un version plus optimisÈe
        for (int Rows = 0; Rows < src.rows; Rows++)
        {
            for (int Cols = 0; Cols < src.cols; Cols++)
            {
                dst.at<uchar>(Rows,Cols)= reduire(src.at<uchar>(Rows,Cols)+bruitmat.at<uchar>(Rows,Cols));
            }
        }
        return dst;
    }
    Mat dst(src.rows,src.cols,CV_8UC3);
    Mat bruitmat = src.clone();
    rng.fill(bruitmat,RNG::UNIFORM,mini,maxi);
    //addWeighted(src, 1.0, bruitmat, 1.0, 0.0, src); peut etre pour un version plus optimisÈe
    for (int Rows = 0; Rows < src.rows; Rows++)
    {
        for (int Cols = 0; Cols < src.cols; Cols++)
        {
            for (int i = 0; i < 3; i++)
            {
                dst.at<Vec3b>(Rows,Cols).val[i]= reduire(src.at<Vec3b>(Rows,Cols).val[i] + bruitmat.at<Vec3b>(Rows,Cols).val[i]);
            }
        }
    }
    return dst;
}

cv::Mat Ajouter_bruit_sel_poivre(cv::Mat src,float mini,float maxi)
{
    if(mini<0)
    {
        mini = 0;
    }
    if(maxi>255)
    {
        maxi = 255;
    }
    if(mini>maxi)
    {
        mini = 0;
        maxi = 255;
    }
    Mat saltpepper_noise = Mat::zeros(src.rows, src.cols,CV_8U);
    randu(saltpepper_noise,0,255);
    
    Mat blanc = saltpepper_noise < mini;
    Mat noir = saltpepper_noise > maxi;
    
    Mat saltpepper_img = src.clone();
    saltpepper_img.setTo(255,blanc);
    saltpepper_img.setTo(0,noir);
    return saltpepper_img;
}

//Histogramme V Openv
cv::Mat dessiner_histogramme(cv::Mat src)
{
    if(src.channels()==1)
    {
        int histSize = 256;
        float range[2]={0,256};
        const float* histRange = { range };
        
        Mat gray_hist;
        
        
        
        calcHist( &src, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange);
        
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );
        Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 40,40,40) );
        normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX);
        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(gray_hist.at<float>(i-1)) ) ,
                 Point( bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i)) ),Scalar( 255, 255, 255));
        }
        
        return histImage;
    }
    // separation des channels
    vector<Mat> bgr_planes;
    split( src, bgr_planes );
    
    int histSize = 256;
    float range[2]={0,256};
    const float* histRange = { range };
    
    Mat b_hist, g_hist, r_hist;
    
    /// Calcul d'histogrammes:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);
    
    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 40,40,40) );
    Mat negativeImage( hist_h, hist_w, CV_8UC3, Scalar( 40,40,40) );
    
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX);
    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
             Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
             Scalar( 0, 0, 255), 2, 8, 0  );
        
    }
    
    return histImage;
}

cv::Mat dessiner_negatif_histogramme(cv::Mat src)
{
    Mat histImage(src.rows,src.cols,CV_8UC3);
    bitwise_not(src,histImage);
    return dessiner_histogramme(histImage);
}
// Convertion en gris
cv::Mat niveaux_de_gris(cv::Mat src)
{
    if(src.channels()==1)
    {
        return src;
    }
    Mat Gris(src.rows,src.cols,CV_8UC1,Scalar( 0,0,0));
    for(int i=0;i<src.rows;i++)
    {
        for(int j=0;j<src.cols;j++)
        {
            Gris.at<uchar>(i,j) = (int)(src.at<Vec3b>(i, j)[0]*0.1140 + src.at<Vec3b>(i, j)[1]*0.5870 + src.at<Vec3b>(i, j)[2]*0.2989);
        }
    }
    return Gris;
}
// Equalisation
cv::Mat equalization_histogramme(cv::Mat src)
{
    Mat dst;
    if(src.channels()!=3)
    {
        /// Convert to grayscale
        cvtColor( src, src, CV_BGR2GRAY );
        /// Apply Histogram Equalization
        equalizeHist( src, dst );
    }
    else
    {
        vector<Mat> channels;
        cvtColor(src, dst, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
        split(dst,channels); //split the image into channels
        equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
        merge(channels,dst); //merge 3 channels including the modified 1st channel into one image
        cvtColor(dst, dst, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)
    }
    return dst;
}


int seuillage_otsu(vector<int> histo)
{
    int N = std::accumulate(histo.begin(), histo.end(), 0);;
    int sum =0;
    int sumB =0;
    int wB=0;
    int wF =0;
    int mB,mF;
    int maxi=0;
    int middle;
    int seuil = 0;
    
    for(int i=0;i<256;i++)
    {
        sum += i*histo[i];
    }
    
    for(int i=0;i<256;i++)
    {
        wB += histo[i];
        if(wB == 0)
        {
            continue;
        }
        
        wF = N-wB;
        if(wF==0)
        {
            continue;
        }
        sumB += i*histo[i];
        mB = sumB/wB;
        mF = (sum-sumB)/wF;
        middle = wB*wF*((mB-mF)*(mB-mF));
        if(middle>maxi)
        {
            seuil =i ;
            maxi = middle;
        }
    }
    return seuil;
}

cv::Mat binarisation(int seuil,cv::Mat src)
{
    src = niveaux_de_gris(src);
    for(int i=0;i<src.rows;i++)
    {
        for(int j=0;j<src.cols;j++)
        {
            src.at<uchar>(i,j) = ((src.at<uchar>(i,j) >= seuil) ? 255:0);
        }
    }
    return src;
}

cv::Mat binarisation_otsu(Mat src)
{
    return binarisation(seuillage_otsu(CalculHist(src)),src);
}

cv::Mat segmenter_image(Mat src,int k)
{
    //int h=src.rows;
    //int w=src.cols;
    //cv::resize(src,src,Size((h/2),(w/2)));
    Mat samples(src.rows * src.cols, 3, CV_32F);
    for( int y = 0; y < src.rows; y++ )
        for( int x = 0; x < src.cols; x++ )
            for( int z = 0; z < 3; z++)
                samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];
    
    
    int clusterCount = k;
    Mat labels;
    int attempts = 2;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 5, 0.1), attempts, KMEANS_PP_CENTERS, centers );
    
    
    Mat new_image( src.size(), src.type() );
    for( int y = 0; y < src.rows; y++ )
        for( int x = 0; x < src.cols; x++ )
        {
            int cluster_idx = labels.at<int>(y + x*src.rows,0);
            new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
            new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
            new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        }
    //cv::resize(new_image,new_image,Size(w,h));
    return new_image;
}

vector<int> encode_RLE(vector<int> str)
{
    
    vector<int> encoding;
    
    int count;
    
    for (int i = 0; i<str.size(); i++)
    {
        // count occurrences of character at index i
        count = 1;
        while (str[i] == str[i + 1])
            count++, i++;
        encoding.push_back(str[i]);
        encoding.push_back(count);
        
        
        
    }
    return encoding;
}

vector<int> decode_RLE(vector<int>  str)
{
    
    vector<int> decode;
    
    for (int i = 1; i<str.size(); i=i+2)
    {
        for(int j=0;j<str[i];j++)
        {
            decode.push_back(str[i-1]);
        }
    }
    return decode;
}

cv::Mat calculDCT(cv::Mat oo){
    int i,j;
    cv::Mat Cosinus(8,8,CV_32F);
    for(i=0;i<8;i++){
        for(j=0;j<8;j++){
            if(i==0){Cosinus.at<float>(i,j)=(1.0/sqrt(8.0));
            }
            else {Cosinus.at<float>(i,j)=float((sqrt(2.0/8.0))*cos(((2*(float)(j)+1)*(float)(i)*3.14)/16));}
            
        }
        
        
    }
    
    
    cv::Mat TranspoCosinus(8,8,CV_32F);
    for(i=0;i<8;i++){
        for(j=0;j<8;j++){
            TranspoCosinus.at<float>(i,j)=Cosinus.at<float>(j,i);
            
        }
        
    }
    
    cv::Mat DCT(8,8,CV_32F);
    DCT=Cosinus*oo*TranspoCosinus;
    for(i=0;i<8;i++){
        for(j=0;j<8;j++){
            DCT.at<float>(i,j)=round((double)(DCT.at<float>(i,j)));
        }
    }
    
    return DCT;
}

cv::Mat calculiDCT(cv::Mat oo){
    int i,j;
    cv::Mat Cosinus(8,8,CV_32F);
    for(i=0;i<8;i++){
        for(j=0;j<8;j++){
            if(i==0){Cosinus.at<float>(i,j)=(1.0/sqrt(8.0));
            }
            else {Cosinus.at<float>(i,j)=float((sqrt(2.0/8.0))*cos(((2*(float)(j)+1)*(float)(i)*3.14)/16));}
            
        }
        
        
    }
    
    
    cv::Mat TranspoCosinus(8,8,CV_32F);
    for(i=0;i<8;i++){
        for(j=0;j<8;j++){
            TranspoCosinus.at<float>(i,j)=Cosinus.at<float>(j,i);
            
        }
        
    }
    
    cv::Mat iDCT(8,8,CV_32F);
    iDCT=TranspoCosinus*oo*Cosinus;
    for(i=0;i<8;i++){
        for(j=0;j<8;j++){
            iDCT.at<float>(i,j)=round((double)(iDCT.at<float>(i,j)));
        }
    }
    
    
    return iDCT;
}

cv::Mat dct_idct(cv::Mat src){ // IMAGE CARREE SEULEMENT
    
    /*   int zz[64]={ 1,2,6,7,15,16,28,29,
     3,5,8,14,17,27,30,43,
     4,9,13,18,26,31,42,44,
     10,12,19,25,32,41,45,54,
     11,20,24,33,40,46,53,55,
     21,23,34,39,47,52,56,61,
     22,35,38,48,51,57,60,62,
     36,37,49,50,58,59,63,64}; */
    
    
    
    
    float tabb[64]={53,37,33,53,80,133,170,203,
        40,40,47,63,87,193,200,183,
        47,43,53,80,133,190,230,187,
        47,57,73,97,170,290,267,207,
        60,73,123,187,227,363,343,257,
        80,117,183,213,270,347,377,307,
        163,213,260,290,343,403,400,337,
        240,307,317,327,373,333,343,330};
    
    
    //   long taillereelle=(src.rows)*(src.cols)*8;
    cv::resize(src,src,cv::Size(512,512));
    
    
    cv::Mat imagecomp = cv::Mat(src.rows, src.cols, CV_8UC1);
    
    
    int largeur=src.size().width;
    int longueur=src.size().height;
    float block[64]={};
    int i,j,v,k,l,u;
    int accu=0;
    
    
    for(u=0;u<largeur;u+=8){
        for(v=0;v<longueur;v+=8){
            
            for(k=0;k<8;k++){
                for(l=0;l<8;l++){
                    
                    block[k*8+l]=src.at<uchar>((u+k),(v+l))-128;
                }
            }
            
            
            cv::Mat img = cv::Mat(8, 8, CV_32F, block);
            cv::Mat DCT=cv::Mat(8,8,CV_64FC1);
            //cv::dct(img,DCT);
            DCT=calculDCT(img);
            Mat Qantifiee=cv::Mat(8,8,CV_8SC1);
            for(i=0;i<8;i++){
                for(j=0;j<8;j++){
                    Qantifiee.at<char>(i,j)=round((DCT.at<float>(i,j)/tabb[i*8+j]));
                }
                
            }
            
            Mat Qantifiee1=cv::Mat(8,8,CV_32F);
            for(i=0;i<8;i++){
                for(j=0;j<8;j++){
                    Qantifiee1.at<float>(i,j)=Qantifiee.at<char>(i,j)*tabb[i*8+j];
                }
            }
            cv::Mat Decomp=cv::Mat(8,8,CV_32F);
            //idct(Qantifiee1,Decomp);
            Decomp=calculiDCT(Qantifiee1);
            Decomp=Decomp+128;
            
            int decomp[64]={};
            
            
            
            for(i=0;i<8;i++){
                for(j=0;j<8;j++){
                    decomp[i*8+j]=(round(Decomp.at<float>(i,j)));
                    if(decomp[i*8+j]>255)
                    {
                        decomp[i*8+j]=255;
                    }
                    if(decomp[i*8+j]<0){
                        decomp[i*8+j]=0;
                    }
                    
                }
            }
            
            for(k=0;k<8;k++){
                for(l=0;l<8;l++){
                    
                    imagecomp.at<uchar>((u+k),(v+l))= decomp[k*8+l];
                }
            }
            
            
            
        }
    }
    return imagecomp;
}

cv::Mat compression_dct(cv::Mat src)
{
    if(src.channels()==3)
    {
        vector<Mat> planes;
        split(src,planes);
        planes[0] = dct_idct(planes[0]);
        planes[1] = dct_idct(planes[1]);
        planes[2] = dct_idct(planes[2]);
        Mat result(src.rows,src.cols,CV_8UC3);
        merge(planes,result);
        return result;
    }
    else
    {
        return dct_idct(src);
    }
}
