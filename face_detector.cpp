/****************************************************************************************************                    TP2: Détection et reconnaissance de visages  							    
            code de référence: documentation de opencv                    
	Auteurs: ELIODOR Ednalson Guy Mirlin 
             Promotion: 21                          				
             Promotion: Master 2- Promotion 21                                		
																									
          
																									 But: Ce programme permet de détecter les visages dans les images 																						       	   
              Compiler:  ./face_detector nom_image                     																	
****************************************************************************************************/
#include <iostream>
#include <stdio.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);
//Variables  représentant les descripteur de face, yeux et nez respectivement
String figure_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
String yeux_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String nez_cascade_name =
		"/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;

String window_name = "Capture - Face detection";

int main(int argc, const char *argv[]) {

	Mat frame;
	string chemin_image = string(argv[1]);

	//Chargement des descripteurs de formes face, yeux et nez respectivement
	if (!face_cascade.load(figure_cascade_name)) {
		printf("--(!)Error loading face cascade\n");
		return -1;
	};
	if (!eyes_cascade.load(yeux_cascade_name)) {
		printf("--(!)Error loading eyes cascade\n");
		return -1;
	};
	if (!nose_cascade.load(nez_cascade_name)) {
		printf("--(!)Error loading eyes cascade\n");
		return -1;
	};

	//Lecture et chargement  de l'image
	frame = imread(chemin_image, 1);

	//Détection et affichage  des visages avec affichage des rectangles encadrant chaque zone détectée(face, yeux, nez) cette fonction est implémentée plus bas
	detectAndDisplay(frame);

	waitKey(0);

	return 0;
}
// fonction de détection de visages dans l'image test.
void detectAndDisplay(Mat frame) {

	Size eye_size; //taille de l'oeuil
	Size nose_size;//taille du nez
	double echelle;
	int nb_voisins;

	//étape de normalisation de la taille de l'image: redimensionnement des images
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

	//conversion en image en niveau de gris pour éviter la sensibilité de la lumière et se limiter à la forme
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	//Affichage de l'image originale pour la compararer avec l'image marquée de carrés.
	imshow("image originale", frame);

	//-- Détection des figurers
	face_cascade.detectMultiScale(frame_gray, faces, echelle, nb_voisins, 0,
			Size(30, 30), Size(200, 200));

	vector<int> faces_index;

	//sélection des figures
	for (size_t i = 0; i < faces.size(); i++) {

		vector<Rect> eyes;
		Mat face_region = frame_gray(faces[i]);

		// détection des yeux dans la figure considérée
		eyes_cascade.detectMultiScale(face_region, eyes, 1.1, 3, 0, eye_size);


		vector<Rect> nose;// vecteur représentant le nez

		// détection du nez dans la face courante
		nose_cascade.detectMultiScale(face_region, nose, 1.1, 3, 0, nose_size);

		//critère d'acceptation
		if (eyes.size() + nose.size() >= 1) {
			faces_index.push_back(i);
			Point pt1(faces[i].x, faces[i].y); // Display detected faces on main window - live stream from camera
			Point pt2((faces[i].x + faces[i].height),
					(faces[i].y + faces[i].width));
			rectangle(frame, pt1, pt2, Scalar(255, 0, 255), 2, 8, 0);

		for (size_t j = 0; j < eyes.size(); j++) {

				Point pt3(eyes[j].x + faces[i].x, eyes[j].y + faces[i].y); // Display detected faces on main window - live stream from camera
				Point pt4((eyes[j].x + faces[i].x + eyes[j].height),
						(eyes[j].y + faces[i].y + eyes[j].width));
				rectangle(frame, pt3, pt4, Scalar(0, 0, 255), 2, 8, 0);
		}
			for (size_t j = 0; j < nose.size(); j++) {

				Point pt7(nose[j].x + faces[i].x, nose[j].y + faces[i].y); // Display detected faces on main window - live stream from camera
				Point pt8((nose[j].x + faces[i].x + nose[j].height),
						(nose[j].y + faces[i].y + nose[j].width));
				rectangle(frame, pt7, pt8, Scalar(0, 255, 255), 2, 8, 0);
			}

		}
	}

	imshow(window_name, frame);
}
