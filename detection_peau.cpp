/*
   Etudiant:      ELIODOR Ednalson Guy Mirlin
                  P21 - IFI 2016
   Cours   :  Vision par Ordinateur
              TP1

*/


//#include <Detection.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

#define NB_IMAGE 25
#define PATH_TO_PEAU_IMAGES "base/peau/"
#define PATH_TO_NON_PEAU_IMAGES "base/non-peau/"

using namespace std;
using namespace cv;



// 1@Segmetation...................Debut 
	
/*Mat img0 = imread(argv[1]); // Cette fonction nous permet de faire la lecture de l'img couleur en entrée
	if (!img0.data) {
		cout << "Attention, il faut donner le nom de l'image\n" << endl;
		return -1;
	}
         else
           cout << " Chaine de traitement en cours d'execution...\n" <<endl;

	Mat img1;
	
	cvtColor(img0, img1, COLOR_RGB2GRAY); // La Fx cvtColor nous permet de transformer l'img couleur en img en niveaux de gris

	//Ici, on va appliquer l'algorithme de OTSU pour etablir le seuil 
	threshold(img1, img1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
*/


// On commence par calculer l'hstogramme de nos images apres la segmentation manuelle effectuee.
float** histogramme(string type, int scale, float &nb_pixels) { 

   float scaleOfReduction = (float) scale / 256;  // réduction de l'espace des valeurs des pixels de 256*256*256 à la dimension de la valeur "scale"

	// Ici, nous arrangeons le chemin du repertoire de la base permettant de lire et charger les images
	char* PATH; // Declaration de la variable PATH contenant le chemin de la base

	if (type.compare("peau") == 0) {
		PATH = PATH_TO_PEAU_IMAGES;  // Chemin vers le repertoire des images peau segmentees manuellement
	} else if (type.compare("non_peau") == 0) {
		PATH = PATH_TO_NON_PEAU_IMAGES; // Chemin vers le repertoire des images non-peau segmentees manuellement
	} 
     else 
      {       cout << "Chemin non detecte, veuillez bien regarder le chemin de la base";
	      cout << " Ou veuillez bien prendre connaissance du nom des repertoires";
	}

	//Création de la matrice devant contenir l'histogramme
	float ** histogramme;
	histogramme = new float*[scale];
	for (int i = 0; i < scale; i++) {
		histogramme[i] = new float[scale];
		for (int j = 0; j < scale; j++) {
			histogramme[i][j] = 0;
		}
	}

	//Construction de l'histogramme
	for (int i = 1; i <= NB_IMAGE; i++) {

		//définition du nom de l'image
		char nom_image[50] = "";
		strcat(nom_image, PATH);
		char num[2] = "";
		sprintf(num, "%d", i);
		strcat(nom_image, num);
		strcat(nom_image, ".jpg");

		//chargement de l'image
		Mat image;
		image = imread(nom_image, 1);

		if (!image.data) {
			cout << "Image non valide " << endl;
			exit(0);
		} else {

			// Le code de onversion CV_BGR2Lab dans l'espace lab . Dans chaque cas, nous convertissons l'ensemble des images de
                        // l'espace de couleurs RGB à l’espace de couleur Lab
			Mat resultMatrice;
			cvtColor(image, resultMatrice, CV_BGR2Lab);


			// Parcours de l'image pour remplissage de l'histogramme
                        // nous avons associé à chaque case (bins) des axes utilisés, le nombre de fois que la valeur de
                        //couleur s'est produite dans la base de données des images.

			for (int k = 0; k < resultMatrice.rows; k++) {
				for (int l = 0; l < resultMatrice.cols; l++) {

					// choix des valeurs a et b
					int a = resultMatrice.at<Vec3b>(k, l).val[1]
							* scaleOfReduction;
					int b = resultMatrice.at<Vec3b>(k, l).val[2]
							* scaleOfReduction;

					// mise à jour des valeurs de l'histogramme
					if (image.at<Vec3b>(k, l) != Vec3b(0, 0, 0)) {

						histogramme[a][b] = histogramme[a][b] + 1;
					}
				}
			}
		}
	}

	                    // Lissage de l'histogramme pour améliorer la detection:
	                   //moyenne de la valeur des 8 pixels voisin + la valeur du pixel

 for (int i = 1; i < (scale - 1); i++) 
{
   // Second@forstarthere
  for (int j = 1; j < (scale - 1); j++) 
{

   histogramme[i][j] = histogramme[i][j] + histogramme[i - 1][j - 1] + histogramme[i - 1][j]
  + histogramme[i - 1][j + 1] + histogramme[i][j - 1] + histogramme[i][j + 1] + histogramme[i + 1][j - 1]
  + histogramme[i + 1][j] + histogramme[i + 1][j + 1] / 8;
  }
 // Second@forfinishere
}

	//  Ici on va calculer les proportions de pixel en fonction du niveau de gris
        //  des images c'est a dire normaisation de l'histogramme 
	for (int m = 0; m < scale; m++) 
        {     
       for (int n = 0; n < scale; n++) 

      {  // Second@forstarthere
	if(histogramme[m][n] !=0)
	nb_pixels += histogramme[m][n];
                      
      } // Second@forfinishere
	}

	for (int m = 0; m < scale; m++) {
        // Second@forstarthere
	for (int n = 0; n < scale; n++) {
	if(histogramme[m][n] !=0)
        histogramme[m][n] /= nb_pixels;
        }// Second@forfinishere
        }

      return histogramme;
}

//ICi on calcul les perforances du programme 
// par le rapport du nombre de pixel correct detectee et le nombre de faux posi, faux negatif

void estimation(Mat img_reference, Mat img_detected) {

	
	float tauxapprentissage;
        int nbre_pixel_peau_vrai = 0;
	int nbre_pixel_peau_faux_pos = 0;
	int nbre_pixel_peau_img_reference = 0;
	int nbre_pixel_peau_faux_neg = 0;

	for (int i = 0; i < img_detected.rows; i++) {
		for (int j = 0; j < img_detected.cols; j++) {

			Vec3b resultMatrice = img_detected.at<Vec3b>(i, j);
			Vec3b original = img_reference.at<Vec3b>(i, j);
			// Nombre de pixel peau correctement détecté dans le résultat
			// le pixel de l'image de résultat et de l'image de référence sont tous différent de noir
			if (resultMatrice != Vec3b(0, 0, 0) && original != Vec3b(0, 0, 0)) {

				nbre_pixel_peau_vrai++;
			}
			// Nombre de pixel peau mal détecté
			if (resultMatrice != Vec3b(0, 0, 0) && original == Vec3b(0, 0, 0)) {
				nbre_pixel_peau_faux_pos++;
			}
			// Nombre de pixel peau dans l'image de référence
			if (original != Vec3b(0, 0, 0)) {
				nbre_pixel_peau_img_reference++;
			}
		}
	}

	nbre_pixel_peau_faux_neg = nbre_pixel_peau_img_reference -nbre_pixel_peau_vrai;
	if(nbre_pixel_peau_faux_neg < 0.0)
		nbre_pixel_peau_faux_neg=0.0;

//  Mat img3 = Mat::zeros(check.size(), CV_8UC3);
//	for (int i = 0; i < check.rows; i++) 
//


//Calcul de la tauxapprentissage du programme
	tauxapprentissage = (float)nbre_pixel_peau_vrai/(nbre_pixel_peau_vrai
					+nbre_pixel_peau_faux_pos + nbre_pixel_peau_faux_neg);
	cout << "reference :"<<nbre_pixel_peau_img_reference<< endl;
	cout << "correct :"<<nbre_pixel_peau_vrai<< endl;
	cout << "faux_positif :"<<nbre_pixel_peau_faux_pos<< endl;
	cout << "faux_negatif :"<<nbre_pixel_peau_faux_neg<< endl;

	cout << "Perfomance du programme = " << tauxapprentissage * 100 << " %" << endl;

}

// Détection de la peau méthode simple
Mat detection_peau_simple(float** histog_peau, float** histog_non_peau,
		Mat test_image, int scale) {

	float scaleOfReduction = (float) scale / 256;
	//conversion de l'image test dans l'espace lab
	Mat resultMatrice;
	cvtColor(test_image, resultMatrice, CV_BGR2Lab);

	Mat mask(test_image.rows, test_image.cols, CV_8UC1);
	mask = Scalar(0);
	Mat output;
	test_image.copyTo(output);
	for (int k = 0; k < resultMatrice.rows; k++) {
		for (int l = 0; l < resultMatrice.cols; l++) {

			// choix des valeurs a et b
			int a = resultMatrice.at<Vec3b>(k, l).val[1] * scaleOfReduction;
			int b = resultMatrice.at<Vec3b>(k, l).val[2] * scaleOfReduction;

			
			if (histog_peau[a][b] < histog_non_peau[a][b]) {

				output.at<Vec3b>(k, l) = Vec3b(0, 0, 0);

			} else {
				mask.at<uchar>(k, l) = 255;
			}
		}
	}


// Affichage des differentes outputs a l'Ecran

	imshow("Image Entree", test_image);  // Affichage de l'image Entree representee par test_image
        imshow("mask", mask);           // Affichage du masque resultant de la segmentation manuelle des images
	imshow("output", output);          // Affichage de l'image output detectee

	return output;
}

// Détection de peau par calcul de probabilité
/*
  Après l'obtention des histogrammes de couleurs de pixels de peau et de non-
  peau, nous calculons la probabilité conditionnelle pour chaque couleur sachant que
  cette couleur est une couleur de peau ou non*/


Mat detection_peau_bayes(float** histog_peau, float** histog_non_peau,
		Mat test_image, int scale, float seuil, float nbre_pixel_peau,
		float nbre_pixel_non_peau) {

	float scaleOfReduction = (float) scale / 256;
	float probabilite_peau = 0.0;
	float probabilite_non_peau = 0.0;


	//calcul des probabilités peau et non peau
/*
   Dans cette étape nous calculons pour chacun des pixels de notre base d'images et
selon chaque combinaison d'axes, la probabilité qu'il soit un pixel de peau

*/

	probabilite_peau = nbre_pixel_peau / (nbre_pixel_peau + nbre_pixel_non_peau);
	probabilite_non_peau = nbre_pixel_non_peau / (nbre_pixel_peau + nbre_pixel_non_peau);
    // le code de conversion CV_BGR2Lab suivant nous permet en autre de convertir
       //  l'image test RGB dans l'espace L*A*B
	Mat resultMatrice;
	cvtColor(test_image, resultMatrice, CV_BGR2Lab);

	// création du mask
/*      Lorseque les régions de couleur de peau sont manuellement étiquetées, le résultat est donc un mask binaire
        celui ci identifie les pixels de la peau  */

	Mat mask(test_image.rows, test_image.cols, CV_8UC1);
	mask = Scalar(0);

//    création de l'image résultat  
   
	Mat output;
	test_image.copyTo(output);

	for (int k = 0; k < resultMatrice.rows; k++) {
		for (int l = 0; l < resultMatrice.cols; l++) {

			// choix des valeurs a et b
			int a =0, b=0;
			 a = resultMatrice.at<Vec3b>(k, l).val[1] * scaleOfReduction;
			 b = resultMatrice.at<Vec3b>(k, l).val[2] * scaleOfReduction;
			 //calcul de la probabilité de décision
			 float proba_decision = 0.0;
			 proba_decision = (histog_peau[a][b] * probabilite_peau)
					/ (histog_peau[a][b] * probabilite_peau
							+ histog_non_peau[a][b] * probabilite_non_peau);

			// mise à jour des valeurs de l'histogramme
			if (proba_decision > seuil) {
				mask.at<uchar>(k, l) = 255;

			} else {
				output.at<Vec3b>(k, l) = Vec3b(0, 0, 0);
			}
		}

	}

	//Post traitement
	int taille_erosion = 1;
	int taile_dilatation = 3;

	Mat dilate_element = getStructuringElement(MORPH_CROSS,
			Size(2 * taile_dilatation + 1, 2 * taile_dilatation + 1),
			Point(taile_dilatation, taile_dilatation));

	Mat erode_element = getStructuringElement(MORPH_CROSS,
			Size(2 * taille_erosion + 1, 2 * taille_erosion + 1),
			Point(taille_erosion, taille_erosion));
	dilate(output, output, dilate_element);

	erode(output, output, erode_element);

	imshow("image entree", test_image);

	imshow("mask", mask);
	imshow("output", output);


	return output;

}

// Affichage de l'histogramme
void histogramme_print(float ** histogramme, int scale, string type) {

	Mat grand_histo(256, 256, CV_8UC1);
	float val_maxi = 0.0;

	//Détermination de la valeur maximale de l'histogramme

	for (int i = 0; i < scale; i++) {
		for (int j = 0; j < scale; j++) {
			if (histogramme[i][j] > val_maxi)
				val_maxi = histogramme[i][j];
		}
	}

	//Agrandissement, normalisation de la matrice de l'histogramme et transformation en image

	for (int i = 0; i < scale; i++) {
		for (int j = 0; j < scale; j++) {
			for (int k = 0; k < 256/scale; k++) {
				for (int l = 0; l < 256/scale; l++)
					grand_histo.at<uchar>(i * 256/scale + k, j * 256/scale + l) =
							saturate_cast<uchar>(
									((histogramme[i][j]) / val_maxi)
											* 255);
						}
		}
	}

	// Enregistrement de l'histogramme
	char nom_histogramme[50] = "";
	strcat(nom_histogramme, "histogramme/histogramme_");
	if (type.compare("peau") == 0) {
		strcat(nom_histogramme, "peau");
	} else {
		strcat(nom_histogramme, "non-peau");
	}
	strcat(nom_histogramme, ".jpg");
	if (!imwrite(nom_histogramme, grand_histo))
		cout << "Erreur lors de l'enregistrement" << endl;

	// Affichage de l'histogramme
	imshow(nom_histogramme, grand_histo);
}

// Fonction principale
int main(int argc, char** argv) {

	int scale = 0;
	float seuil = 0.0;
	scale = atoi(argv[1]);
	seuil = atof(argv[2]);
	float ** histog_peau = NULL;
	float ** histog_non_peau = NULL;
	float nbre_pixel_peau = 0;
	float nbre_pixel_non_peau = 0;
	char* arg_nom = argv[3];
	char nom_test_image[50]= "";
	strcat(nom_test_image,"base/test/");
	strcat(nom_test_image,arg_nom);

	// Lecture de l'image entrée
	Mat img_input;
	img_input = imread(nom_test_image, 1);

	char name_reference_img[50] = PATH_TO_PEAU_IMAGES;
	strcat(name_reference_img,arg_nom);

	// Lecture de l'image de référence
	Mat img_reference;
	img_reference = imread(name_reference_img, 1);
	imshow("image reference", img_reference);


	Mat img_detected;

	// calcul des histogrammes
	histog_peau = histogramme("peau", scale, nbre_pixel_peau);

	histog_non_peau = histogramme("non_peau", scale, nbre_pixel_non_peau);

	

	img_detected = detection_peau_bayes(histog_peau, histog_non_peau,
			img_input, scale, seuil, nbre_pixel_peau, nbre_pixel_non_peau);

	char nom_image_resultMatrice[50] ="";
		strcat(nom_image_resultMatrice,"resultMatrice/");
		strcat(nom_image_resultMatrice,"resultMatrice_image_");
		strcat(nom_image_resultMatrice,arg_nom);
		if (!imwrite(nom_image_resultMatrice, img_detected))
				cout << "Erreur lors de l'enregistrement" << endl;

	estimation(img_reference, img_detected);
	histogramme_print(histog_peau,scale,"peau");
	histogramme_print(histog_non_peau,scale,"non_peau");
	waitKey(0);
	return 0;
}
