/****************************************************************************************************                    TP2: Détection et reconnaissance de visages  							    
            code de référence: documentation de opencv                    
	Auteurs: ELIODOR Ednalson Guy Mirlin 
             Promotion: 21                           				
             Promotion: Master 2- Promotion 21                               		
																									
          
																									 But: Ce programme permet de reconnaitre et d'étiquetté de nombre visages présents dans une image,
c'est la troisième partie de notre tp 																						       	   
                                   																	
****************************************************************************************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"



using namespace cv;
using namespace std;

//Variables globales représentant les descripteurs caractérisant une figure, des yeux et un nez respectivement

String figure_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
String yeux_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String nez_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml";
CascadeClassifier figure_cascade;
CascadeClassifier yeux_cascade;
CascadeClassifier nez_cascade;
int nbre_classes = 0;

//Cet fonction calcule la distance euclidienne entre un vecteur et un ensemble de vecteur passé en paramètre, puis retourne un vecteur les contenant

vector<double> dist_vec_to_ensemble(vector<Mat> projected_image_entrainement,
		Mat projected_test_image) {

	vector<double> vecdistances;

	for (size_t i = 0; i < projected_image_entrainement.size(); i++) {
		vecdistances.push_back(
				norm(projected_image_entrainement[i], projected_test_image,
						CV_L2));
	}

	return vecdistances;
}

//Distance de mahalanobis d'un vecteur vers un ensemble de vecteurs
vector<double> dist_vec_to_ensemble_mahalanobis(
		vector<Mat> projected_image_entrainement, Mat projected_test_image) {

	vector<double> distancesvec;


	for (size_t i = 0; i < projected_image_entrainement.size(); i++) {
		vector<Mat> merged_mat;
		Mat covar;
		Mat icovar;
		Mat mean;
		merged_mat.push_back(projected_image_entrainement[i]);
		merged_mat.push_back(projected_test_image);
		calcCovarMatrix(merged_mat, covar, mean, COVAR_NORMAL, CV_64F);
		invert(covar, icovar, DECOMP_SVD);
		distancesvec.push_back(
				Mahalanobis(projected_image_entrainement[i], projected_test_image,
						icovar));
	}

	return distancesvec;
}

//Prédiction de classe d'appartenance de l'image par l'algorithme du KNN

int kNN(vector<Mat> projected_image_entrainement, Mat projected_test_image,
		vector<int> training_labels, int k) {

		//calcul des distances
	vector<double> distances_tab;
	distances_tab = dist_vec_to_ensemble(projected_image_entrainement,
			projected_test_image);

	//tri croissant des distances
	vector<double> sorted_distances;
	sorted_distances = distances_tab;
	std::sort(sorted_distances.begin(), sorted_distances.end());

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
		int * compteurClasse;
		compteurClasse = new int[k];
		for (int i = 0; i < k; i++) {
			compteurClasse[i] = 0;
		}

		for (int i = 0; i < k; i++) {

			compteurClasse[i] = std::count(knn.begin(), knn.end(), knn[i]);
			i = i + compteurClasse[i] - 1;
		}

		//Determination de la classe avec la plus grande occurence
		int max_index = 0;
		for (int m = 1; m < k; m++) {
			if (compteurClasse[max_index] < compteurClasse[m]) {
				max_index = m;
			}
		}
		int classe = knn[max_index];

		return classe;
	}
}

// construction du modèle d'apprentissage de reconnaissance de visage
Mat detect_training(Mat frame) {

	Size eye_size;
	Size nose_size;
	double echelle;
	int nb_voisins;

	//redimensionnement des images
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

	//ceci est fait pour faire abstraction des détails et focaliser beaucoup plus sur la forme
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	//-- Detection des figures
	figure_cascade.detectMultiScale(frame_gray, faces, echelle, nb_voisins, 0,
			Size(30, 30), Size(200, 200));

	vector<int> faces_index;

	//sélection  des figures
	for (size_t i = 0; i < faces.size(); i++) {

		vector<Rect> eyes;
		Mat face_region = frame_gray(faces[i]);

		// détection des yeux dans la figure courante
		yeux_cascade.detectMultiScale(face_region, eyes, 1.1, 3, 0, eye_size);

		vector<Rect> nose;

		// détection du nez dans la figure considérée
		nez_cascade.detectMultiScale(face_region, nose, 1.1, 3, 0, nose_size);

		//critère d'acceptation pour voir si il sagit d'un visage
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


// détection de visage pour l'image de test
vector<Rect> detecter_test(Mat frame) {

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
	figure_cascade.detectMultiScale(frame_gray, faces, echelle, nb_voisins, 0,
			Size(30, 30), Size(200, 200));

	vector<int> faces_index;

	//détermination  des figures
	for (size_t i = 0; i < faces.size(); i++) {

		vector<Rect> eyes;
		Mat face_region = frame_gray(faces[i]);

		// détection des yeux dans la face considérée
		yeux_cascade.detectMultiScale(face_region, eyes, 1.1, 3, 0, eye_size);

		vector<Rect> nose;

		// détection du nez dans la face considérée
		nez_cascade.detectMultiScale(face_region, nose, 1.1, 3, 0, nose_size);

		//critère d'acceptation
		if (eyes.size() + nose.size() >= 1) {
			faces_index.push_back(i);
			Point pt1(faces[i].x, faces[i].y);
			Point pt2((faces[i].x + faces[i].height),
					(faces[i].y + faces[i].width));
			rectangle(frame, pt1, pt2, Scalar(255, 0, 255), 2, 8, 0);

			
		}
	}

	
	vector<Rect> face_detected;
	for (size_t i = 0; i < faces_index.size(); i++) {

		face_detected.push_back(faces[faces_index[i]]);
		
	}
	imshow("test", frame);
	return face_detected;
}

//fonction permettant de lire les données de l'ensemble d'apprentissage et test
static void lire_donnees(const string& filename, vector<Mat>& images,
		vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message =
				"Le fichier en entrée ne respecte pas le format";
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


int main(int argc, const char *argv[]) {

	// checker le nombre de paramètres
	if (argc < 2) {
		cout << "usage: " << argv[0] << " <apprentissage.txt> <xxxx.jpgt> > "
				<< endl;
		exit(1);
	}

	//Chargement des descripteurs
	if (!figure_cascade.load(figure_cascade_name)) {
		printf("--(!)erreur chargement du descripteur face\n");
		return -1;
	};

	if (!yeux_cascade.load(yeux_cascade_name)) {
		printf("--(!)erreur chargement du descripteur yeux\n");
		return -1;
	};
	if (!nez_cascade.load(nez_cascade_name)) {
		printf("--(!)erreur chargement du descripteur nez\n");
		return -1;
	};

	// lire les fichiers descriptifs de la base d'apprentissage
	string fichier_entrainement = string(argv[1]);

	//lire le nom du fichier test
	string fichier_test = string(argv[2]);

	// vecteurs images
	vector<Mat> image_entrainement;
	Mat test_image;
	vector<Mat> image_traitees;

	// Noms des individus
	vector<string> person_names;
	person_names.push_back("Obama");
	person_names.push_back("Bela");
	person_names.push_back("Elio");
	person_names.push_back("Sacha");
	person_names.push_back("Bob");
	person_names.push_back("Michelle");
	person_names.push_back("Max");
	person_names.push_back("Luco");
	person_names.push_back("Inconnu");

	// vecteur contenant labels d'images apprentissage
	vector<int> training_labels;

	// Lecture des donnnees
	try {
		lire_donnees(fichier_entrainement, image_entrainement, training_labels);
	} catch (cv::Exception& e) {
		cerr << "erreur fichier \"" << fichier_entrainement << "\". Reason: "
				<< e.msg << endl;
		
		exit(1);
	}
	test_image = imread(fichier_test, 1);

	//Détection des images dans la base d'apprentissage
	for (size_t i = 0; i < image_entrainement.size(); i++) {

		image_entrainement.at(i) = detect_training(image_entrainement[i]);

	}

	//Détection des images dans la base de test
	vector<Rect> test_figures = detecter_test(test_image);
	cout << "taille test_face :" << test_figures.size();
	resize(test_image, test_image, Size(1200, 1000), 1.0, 1.0, INTER_CUBIC);
	for (size_t i = 0; i < test_figures.size(); i++) {

		Mat face = test_image(test_figures[i]);

		image_traitees.push_back(detect_training(face));
	}

	//ACP
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(image_entrainement, training_labels);

	// Récupération des valeurs propres
	Mat eigenvalues = model->getMat("eigenvalues");

	// Récupération des vecteurs propres
	Mat eigenvectors = model->getMat("eigenvectors");

	// Récupération de l'image moyenne
	Mat mean = model->getMat("mean");

	//projection des images de la base d'apprentissage
	vector<Mat> projected_image_entrainement;

	projected_image_entrainement = model->getMatVector("projections");

	//projection des images de la base de test
	vector<Mat> projected_test_images;
	//cout << "on est ici"<<endl;

	for (size_t i = 0; i < image_traitees.size(); i++) {
		//cout << "entre"<<endl;
		Mat projection = subspaceProject(eigenvectors, mean,
				image_traitees[(int) i].reshape(1, 1));

		projected_test_images.push_back(projection);


	}

	//prédiction des labels
	vector<int> predicted_labels;
	for (size_t i = 0; i < image_traitees.size(); i++) {
		predicted_labels.push_back(
				kNN(projected_image_entrainement, projected_test_images[i],
						training_labels, 1));
		string identified_person = person_names[predicted_labels[i]];
		string result_message = format(" Classe prédite = %d ",
				predicted_labels[i]);
		cout << result_message << endl;
	}

	// affichage des nom des personnes reconnues dans l image de test
	for (size_t i = 0; i < test_figures.size(); i++) {

		Point pt1(test_figures[i].x, test_figures[i].y); 
		Point pt2((test_figures[i].x + test_figures[i].height),
				(test_figures[i].y + test_figures[i].width));
		rectangle(test_image, pt1, pt2, Scalar(255, 0, 255), 2, 8, 0);
		putText(test_image, person_names[predicted_labels[i]], pt1,
				FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);

	}

	imshow(format("Resultat"), test_image);


	waitKey(0);

	return 0;
}
