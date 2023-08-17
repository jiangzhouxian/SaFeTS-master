
#echo 'result of resnet50'
#python sample_data.py resnet50 pgd phase kmeans high_uc >>resnet50-sample-result.txt
#python sample_data.py resnet50 pgd phase kmeans low_uc >>resnet50-sample-result.txt
#python sample_data.py resnet50 pgd phase kmeans uniform >>resnet50-sample-result.txt


#for sample in 0.001 0.005 0.01 0.02 0.05 0.1
#do
   # python retrain_main.py '1' resnet50 pgd phase kmeans "$sample" low_uc >>resnet50-lowuc-result.txt 
   # python retrain_main.py '1' resnet50 pgd phase kmeans "$sample" high_uc >>resnet50-highuc-result.txt 
   # python retrain_main.py '1' resnet50 pgd phase kmeans "$sample" uniform >>resnet50-uniform-result.txt 
#done

for slic in residual quaternion_fourier
do 
    python sample_data.py resnet50 pgd "$slic" kmeans uniform >>sliency-sample-result.txt
    python retrain_main.py '0' resnet50 pgd "$slic" kmeans 0.1 uniform >>sliency-retrain-result.txt 
done