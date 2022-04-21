#Structure des spectres générés : 
Eps : [2, 3, 2, 3, 2, 3, 2, 3, 2, 3]
Mu : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Hauteur : int random entre 40 et 200 nm

Le dataset contient 10 000 spectres générés de manière aléatoire.
==> Features : Coef de reflexion en fonction de lambda ( 100 valeurs )
==> Labels : Epaisseurs des couches généreés de manière aléatoire

==> Batch size : 20 ( On peut le changer, c'est complétement arbitraire)

Model : 
=> Input : 100 ( les 100 points du spectre ) 
=> Output : 10 ( 10 épaisseurs ) 

== > [100, 256] => [256, 128] => [128, 64] => [64, 32] => [32, 16] => [16, 10]
On utilise la fonction d'activation LeakyReLU entre chaque couche ( j'ai pas trouvé mieux, j'ai pas eu de bons résultats avec les autres )

Fonction de coût : Mean Square Error ( https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss ) 

Optimizer : SGD ( https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD ) 

Entrainement du model avec un backpropagation

