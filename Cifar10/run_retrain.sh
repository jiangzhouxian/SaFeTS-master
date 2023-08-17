
#echo 'Retrain result of ResNet18'
#for sample in 0.01 0.05 0.1 0.2 0.5
#do
    #python cifar10_main.py '0' ResNet18 pgd phase kmeans "$sample" high_uc >>RS18-Retrain-highuc-result.txt
#done

python cifar10_main.py '0' VGG16 PGD phase kmeans 0.2 uniform 
#python cifar10_main.py '1' ResNet18 PGD highpass kmeans 0.2 uniform &
#python cifar10_main.py '2' ResNet18 PGD residual kmeans 0.2 uniform 
#python cifar10_main.py '0' ResNet18 PGD quaternion_fourier kmeans 0.2 uniform 
