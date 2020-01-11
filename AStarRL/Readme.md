MSE map, MSE play, BN, f(x)+x, Adam,
1 layer, no activation, lr=1e-1, batch=dataset entier
    loss -> 0
    0 erreurs
    play converge

sans f(x)+x
    play ne converge pas
    map converge

BCELogits - sans f(x)+x - sans player loss
    map converge

BCE, BCE Squared, Perceptron BCE
f(x)+x (donc custom clamp pour pouvoir utiliser une BCE)
    map très instable mais descent parfois au moins jusqu'à 1e-3
    
sans BN
    instable, play et map ne convergent pas
    play apprend mieux avec lr=1e-3

SGD au lieu de Adam
    très compliqué, l'entraînement s'arrête très vite
    il faut baisser le lr progressivement
    
réseau complexe (s, 1024, 512, s), lr=1e-2
    sigmoid/tanh: convergent de temps en temps
    relu convergent



done list:
    - BN, f(x)+x, MSE/BCE, BCE Squared, BCE Perceptron, one hot vector
    
    - affectation des poids du réseau à la main
    
    - custom clamp: instable.
    todo tester d'autres fonctions dans un autre cadre ex activation 'sign'
    
    - calcul des différentes transitions permet de voir
    le type d'erreurs faites par le NN, et voir si le dataset est déséquilibré
    
    - calcul de statistiques sur les prédictions entières
        
    - détection de fin d'apprentissage. todo: lr dynamique prenant en compte
        - le fait qu'on ait déjà obtenu une loss meilleure
        - le fait que la loss ne puisse pas passer sous 0
    
    - programmer:
        - list[i-1] avec i=0
        - a = b==c avec b et c non broadcastables
        - __eq__, __str__, list.index(), "bisect"
        - AStar
        

attend suffisament pour chaque action
accède aux bons indices