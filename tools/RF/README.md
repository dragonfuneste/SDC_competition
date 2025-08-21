# TRAINING tool 
I have a folder Training where I have a jupyter notebook where I had done all of my training that are working 

# Evaluation Tool
This my tool using a random tree forest. The training is done on a jupyter notebook in ./notebook/training. There is also a overview of the dataset at the same place. 

I made 3 RF : 
-> The first one : rf_model_1M,  with only the mean curvature to look at something simple
-> The second one : rf_model_2M with the mean curvature and the total lenght of the road as metric because in the article it say that the result aren't the best with it 
-> The third one : rf_model_2M_BIS with a try to equilibrate the class when training the RF (deleted because it didn't give something credible)
-> The fourth one : rf_model_3M the mean curvature and the total lenght of the road and the mean Radius best model and currently it the one who see the more credible 

## Usage
The required test data is stored with Git LFS.
Hence you need to have Git LFS installed.
For this, refer to the offical documentation: https://git-lfs.com
  b
After you have installed Git LFS, you can pull the large test data file(s) of this repository.
```bash
git lfs pull
```

Then build and run the evaluator tool that uses the sample dataset and performs an initial evaluation:

### For the one with one metrics
```bash
cd tools/RF
docker build -t main-image .
docker run --rm --name main-container -t -p 4545:4545 main-image -p 4545  -m RF_model_3M.joblib -name RF-Selector
```
### OTHER run command (for the other model )
1 metric model
```bash
docker run --rm --name main-container -t -p 4545:4545 main-image -p 4545  -m RF_model_1M.joblib -name RF-Selector
```

2 metric model
```bash
docker run --rm --name main-container -t -p 4545:4545 main-image -p 4545  -m RF_model_2M.joblib -name RF-Selector
```

3 metric model
```bash
docker run --rm --name main-container -t -p 4545:4545 main-image -p 4545  -m RF_model_3M.joblib -name RF-Selector
```
##  Next step
After it you should open another terminal and put the evaluator command line : 
```bash
cd evaluator
docker build -t evaluator-image .
docker run --rm --name evaluator-container -t evaluator-image -u host.docker.internal:4545 -t sample_tests/sdc-test-data.json
```
