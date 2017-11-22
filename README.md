# visionParOrdinateur
Ce repository contient les projets réalisés dans le cours de Vision Par Ordinateur

1- Detection de la peau : 
   L’objectif est de construire une application de détection de la peau à partir d’une base d’apprentissage. Pour cela, j'ai tout d'abord construit une base d'exemples de « peau » et de « nonpeau» et gerer l'apprentissage avec notre programme. J'ai fait un apprentissage de la couleur de la peau en utilisant l'espace couleur CIE LAB (les composantes AB seulement. 
   
2-Détection et Reconnaissance de visages : 
  L'objectif est de créer un programme qui peut détecter et reconnaître des personnes dans des photos. Pour détecter des visages, j'ai utilisé la méthode proposée par Violas et Jones [IJCV 2004] qui est déjà installé dans OpenCV. En suite, nj'ai utilisé la
méthode de ACP (Analyse en Composantes Principales) pour la reconnaissance des visages détectées.

3- Détection suivi de mouvement : 
   L'objectif est de créer un programme qui permet de détecter les mouvements dans une vidéo, ensuite de suivre les mouvements des objets en utilisant le filtre de Kalman. Comme paramètres d’entrée du programme on aura le nom d’une vidéo ainsi que 4 autres paramètres.
   
A la sortie du programme on aura:
Une image de fond de la vidéo, 
Les images binaires des mouvements détectés, 
La trajectoire du mouvement des objets.

Les bases d'apprentissage et de tests contiennent les images des gens qui ne sont pas des personnages publiques donc raison pour laquelle je ne les ai pas publiés. 
