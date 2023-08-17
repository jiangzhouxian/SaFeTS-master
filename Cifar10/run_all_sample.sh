
#echo 'result of ResNet18-pgd-saliency-kmeans-0.2-uniform'>>RS18-sample-result.txt
#python sample_data.py ResNet18 PGD phase kmeans uniform >>RS18-sample-result.txt
#python sample_data.py ResNet18 PGD highpass kmeans uniform >>RS18-sample-result.txt
#python sample_data.py ResNet18 PGD residual kmeans uniform >>RS18-sample-result.txt
#python sample_data.py ResNet18 PGD quaternion_fourier kmeans uniform >>RS18-sample-result.txt

echo 'result of VGG16-pgd-saliency-kmeans-0.2-uniform'>>VGG16-sample-result.txt
python sample_data.py VGG16 PGD phase kmeans uniform >>VGG16-sample-result.txt
python sample_data.py VGG16 PGD highpass kmeans uniform >>VGG16-sample-result.txt
python sample_data.py VGG16 PGD residual kmeans uniform >>VGG16-sample-result.txt
python sample_data.py VGG16 PGD quaternion_fourier kmeans uniform >>VGG16-sample-result.txt

echo 'result of DenseNet121-pgd-saliency-kmeans-0.2-uniform'>>DenseNet121-sample-result.txt
python sample_data.py DenseNet121 PGD phase kmeans uniform >>DenseNet121-sample-result.txt
python sample_data.py DenseNet121 PGD highpass kmeans uniform >>DenseNet121-sample-result.txt
python sample_data.py DenseNet121 PGD residual kmeans uniform >>DenseNet121-sample-result.txt
python sample_data.py DenseNet121 PGD quaternion_fourier kmeans uniform >>DenseNet121-sample-result.txt


