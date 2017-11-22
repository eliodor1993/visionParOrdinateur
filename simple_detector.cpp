
/****************************************************************************************************                    TP2: Détection et reconnaissance de visages  							    
            code de référence: documentation de opencv                    
	Auteurs: ELIODOR Ednalson Guy Mirlin 
             Promotion: 21                          				
             Promotion: Master 2- Promotion 21                                		
																									
          
																									 But: Ce programme permet de reconnaitre et d'étiquetter les images test passées en paramètres
c'est la deuxième partie de notre tp 																						       	   
                                   																	
****************************************************************************************************/

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;

//Variables globales
String face_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
//String mouth_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml";
String nose_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;

int nb_classes = 0;

//Cet fonction calcule la distance euclidienne entre un vecteur et un ensemble de vecteur passé en paramètre, puis retourne un vecteur les contenant
vector<double> dist_vec_to_set(vector<Mat> projected_training_images,
		Mat projected_test_image) {

	vector<double> distances;

	for (size_t i = 0; i < projected_training_images.size(); i++) {
		distances.push_back(
				norm(projected_training_images[i], projected_test_image,
						CV_L2));
	}

	return distances;
}
//Distance de mahalanobis d'un vecteur vers un ensemble de vecteurs
vector<double> dist_vec_to_set_mahalanobis(
		vector<Mat> projected_training_images, Mat projected_test_image) {

	vector<double> distances;
	cout << "ca passe" << endl;

	for (size_t i = 0; i < projected_training_images.size(); i++) {
		vector<Mat> merged_mat;
		Mat covar;
		Mat icovar;
		Mat mean;

		merged_mat.push_back(projected_training_images[i]);
		merged_mat.push_back(projected_test_image);

		calcCovarMatrix(merged_mat, covar, mean, COVAR_NORMAL, CV_64F);
	
		invert(covar, icovar, DECOMP_SVD);
		distances.push_back(
				Mahalanobis(projected_training_images[i], projected_test_image,
						icovar));
	}

	return distances;
}

//Prédiction de classe d'appartenance de l'image par l'algorithme du KNN


int kNN(vector<Mat> projected_training_images, Mat projected_test_image,
		vector<int> training_labels, int k) {
	
	//calcul des distances
	vector<double> distances_tab;
	distances_tab = dist_vec_to_set(projected_training_images,
			projected_test_image);

//tri croissant des distances
	vector<double> sorted_distances;
	sorted_distances = distances_tab;
	//sorting of the distance array obtained
	std::sort(sorted_distances.begin(), sorted_distances.end());
//
// vérification des images inconnues
	if (sorted_distances[0] > 9000) {

		int classe = 8;
		return classe;
	} else {

		//stoquage des k premiers resultats pour choisir la classe majoritaire parmis elles plutard
		vector<int> knn;

		for (int i = 0; i < k; i++) {
			int pos = find(distances_tab.begin(), distances_tab.end(),
					sorted_distances[i]) - distances_tab.begin();
			knn.push_back(training_labels[pos]);
		}
		std::sort(knn.begin(), knn.end());

//Determination du nombre de classes
		int * compteur;
		compteur = new int[k];
		for (int i = 0; i < k; i++) {
			compteur[i] = 0;
		}

		for (int i = 0; i < k; i++) {

			compteur[i] = std::count(knn.begin(), knn.end(), knn[i]);
			i = i + compteur[i] - 1;
		}

//Determination de la classe avec la plus grande occurence
		int max_index = 0;
		for (int m = 1; m < k; m++) {
			if (compteur[max_index] < compteur[m]) {
				max_index = m;
			}
		}
		int classe = knn[max_index];

		return classe;
	}
}


// detection de visage pour la base d'apprentissage et de test
Mat detect(Mat frame) {

	Size eye_size;
	Size nose_size;
	double echelle;
	int nb_voisins;

//redimensionnement des images
	if ((frame.cols > frame.rows) && (frame.cols * frame.rows > 1200000)) {
		resize(frame, frame, Size(1200, 1000), 1.0, 1.0, CV_INTER_AREA);
		eye_size = Size(5, 5);
		nose_size = Size(15, 15);
		echelle = 1.1;
		nb_voisins = 2;
	}
	if ((frame.cols > frame.rows) && (frame.cols * frame.rows < 1200000)) {
		resize(frame, frame, Size(1200, 1000), 1.0, 1.0, CV_INTER_LINEAR);
		eye_size = Size(5, 5);
		nose_size = Size(15, 15);
		echelle = 1.1;
		nb_voisins = 2;
	}
	if ((frame.cols <= frame.rows) && (frame.cols * frame.rows > 55000)) {
		resize(frame, frame, Size(220, 250), 1.0, 1.0, CV_INTER_AREA);
		eye_size = Size(30, 30);
		nose_size = Size(50, 50);
		echelle = 1.01;
		nb_voisins = 1;
	}
	if ((frame.cols <= frame.rows) && (frame.cols * frame.rows < 55000)) {
		resize(frame, frame, Size(220, 250), 1.0, 1.0, CV_INTER_LINEAR);
		eye_size = Size(30, 30);
		nose_size = Size(50, 50);
		echelle = 1.01;
		nb_voisins = 1;
	}
	std::vector<Rect> faces;
	Mat frame_gray;

	//conversion en image en niveau de gris
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	//-- Detection des faces
	face_cascade.detectMultiScale(frame_gray, faces, echelle, nb_voisins, 0,
			Size(30, 30), Size(200, 200));

	vector<int> faces_index;

	//discrimination des faces
	for (size_t i = 0; i < faces.size(); i++) {

		vector<Rect> eyes;
		Mat face_region = frame_gray(faces[i]);

		// détection des yeux dans la face considérée
		eyes_cascade.detectMultiScale(face_region, eyes, 1.1, 3, 0, eye_size);


		vector<Rect> nose;

		// détection du nez dans la face considérée
		nose_cascade.detectMultiScale(face_region, nose, 1.1, 3, 0, nose_size);

		//critère d'acceptation
		if (eyes.size() + nose.size() >= 0) {
			faces_index.push_back(i);



		}
	}
	//Normalisation de l'image
	equalizeHist(frame_gray, frame_gray);
	Mat face_detected = frame_gray(faces[faces_index[0]]);
	//Redimensionnement pour l acp
	resize(face_detected, face_detected, Size(100, 100), 1.0, 1.0, INTER_CUBIC);
	return face_detected;
}

// normalisation de 0 à 225 d'une image avec des valeurs de pixels flottant
static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

//Lecture des données
static void read_data(const string& filename, vector<Mat>& images,
		vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message =
				"No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		//cout << path << endl;
		getline(liness, classlabel);
		//cout << classlabel << endl;
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 1));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

//somme des éléments diagonaux
	//et aussi de tous les éléments de la matrice
float bon_taux(int **conf_mat) {
	float taux = 0;
	float somme_diag = 0;
	float somme_tot = 0;

	//sum of diagonal elements is divided by sum of
	//all the matrix elements
	for (int i = 0; i < nb_classes; i++) {
		for (int j = 0; j < nb_classes; j++) {
			if (i == j) {
				somme_diag += conf_mat[i][j];
			}
			somme_tot += conf_mat[i][j];
		}
	}
	taux = somme_diag / somme_tot;
	return taux;

}

//Fonction principale
int main(int argc, const char *argv[]) {

	// Vérifier la validité de la syntaxe d'appel du programme
	if (argc < 2) {
		cout << "usage: " << argv[0]
				<< " <apprentissage.txt> <test.txt> <dossier resultat> "
				<< endl;
		exit(1);
	}

	//Chargement des descripteurs
	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading face cascade\n");
		return -1;
	};

	// lire les fichiers descriptifs de la base d'apprentissage et de test
	string training_file = string(argv[1]);
	string test_file = string(argv[2]);


	// vecteurs images apprentissage et test.
	vector<Mat> training_images;
	vector<Mat> test_images;
	vector<Mat> processed_images;

	// Nom des personnes
	vector<string> person_names;
	person_names.push_back("Obama");
	person_names.push_back("Bela");
	person_names.push_back("Max");
	person_names.push_back("Michelle");
	//person_names.push_back("Bob");
	person_names.push_back("Elio");
	person_names.push_back("Sacha");
	person_names.push_back("Luco");
	person_names.push_back("Inconnu");
     
	// vecteurs labels d'images apprentissage et test.
	vector<int> training_labels;
	vector<int> test_labels;
       
	// Lecture des donnnees
	try {
		read_data(training_file, training_images, training_labels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << training_file << "\". Reason: "
				<< e.msg << endl;
		// nothing more we can do
		exit(1);
	}
           
	try {
		read_data(test_file, test_images, test_labels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << test_file << "\". Reason: " << e.msg
				<< endl;
	
		exit(1);
	}
   
	//Détection des images dans la base d'apprentissage
	for (size_t i = 0; i < training_images.size(); i++) {

		training_images.at(i) = detect(training_images[i]);

	}

	//Détection des images dans la base de test
	for (size_t i = 0; i < test_images.size(); i++) {
                //cout <<"nyanga"<<test_labels[i] <<"\n";
		processed_images.push_back(detect(test_images[i]));

	}
     
	//ACP
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();

	model->train(training_images, training_labels);
        
	// Récupération des valeurs propres
	Mat eigenvalues = model->getMat("eigenvalues");

	// Récupération des vecteurs propres
	Mat eigenvectors = model->getMat("eigenvectors");

	// Récupération de l'image moyenne
	Mat mean = model->getMat("mean");

	//projection des images de la base d'apprentissage
	vector<Mat> projected_training_images;

	projected_training_images = model->getMatVector("projections");

	//projection des images de la base de test
	vector<Mat> projected_test_images;
	//cout << "on est ici"<<endl;

	for (size_t i = 0; i < test_images.size(); i++) {
		//cout << "entre"<<endl;
		Mat projection = subspaceProject(eigenvectors, mean,
				processed_images[(int) i].reshape(1, 1));

		projected_test_images.push_back(projection);

		}

	//prédiction des labels
	vector<int> predicted_labels;
	for (size_t i = 0; i < test_labels.size(); i++) {
		predicted_labels.push_back(
				kNN(projected_training_images, projected_test_images[i],
						training_labels, 1));
		string identified_person = person_names[predicted_labels[i]];
		string result_message = format(
				" Classe prédite = %d / Classe réelle = %d.",
				predicted_labels[i], test_labels[i]);
		cout << "Image de " << person_names[test_labels[i]] << " : "
				<< result_message << endl;
		resize(test_images[i], test_images[i], Size(220, 250), 1.0, 1.0,
				INTER_CUBIC);
		putText(test_images[i], person_names[predicted_labels[i]],
				Point(test_images[i].cols / 110, 20), FONT_HERSHEY_PLAIN, 1.0,
				CV_RGB(0, 255, 0), 2.0);

		imshow(format("%d", i), test_images[i]);
		//   imwrite(format("%s/Test_%d", result_folder.c_str(), i), test_images[i]);

	}

	//déterminantion du nombre de classes
	sort(test_labels.begin(), test_labels.end());
	int compt = 0;
	std::vector<int> classes;
         
	for (size_t i = 0; i < test_labels.size(); i++) {

		compt = count(test_labels.begin(), test_labels.end(), test_labels[i]);
		classes.push_back(test_labels[i]);
		i = i + compt - 1;
		nb_classes++;
	}

//creation de la matrice de cofusion
	int **confusion_matrix;
	confusion_matrix = new int*[nb_classes];

	for (int j = 0; j < nb_classes; j++) {
		confusion_matrix[j] = new int[nb_classes];
		for (int n = 0; n < nb_classes; n++) {
			confusion_matrix[j][n] = 0;
		}
	}

	//remplissage matrice de confusion
	for (size_t m = 0; m < test_images.size(); m++) {

		confusion_matrix[test_labels[m]][predicted_labels[m]]++;
	}

	//Affichage matrice de confusion
	cout <<  "Matrice de confusion"<< endl;
	for (int a = 0; a < nb_classes; a++) {
		for (int b = 0; b < nb_classes; b++) {

			cout << confusion_matrix[a][b] << " ";
		}
		cout << endl;

	}

	//affichage taux de précision
	cout << endl << "le taux de précision est :"
			<< bon_taux(confusion_matrix) * 100 << "%" << endl;


	waitKey(0);

	return 0;
}
