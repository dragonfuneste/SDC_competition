# Evaluation Tool
This my tool using a random tree forest. The training is done on a jupyter notebook in ./notebook/training. There is also a overview of the dataset at the same place. 

I made 3 RF : 
-> The first one : rf_model,  with only the mean curvature to look at something simple
-> The second one : rf_model_RL1 with the mean curvature and the total lenght of the road as metric because in the article it say that the result aren't the best with it 
-> The third one : rf_model_RL2 with a try to equilibrate the class when training the RF 
-> A fourth one was in devellopement ( with local curvature and radius ) but because of time constraint it wasn't finished 

## Usage
The required test data is stored with Git LFS.
Hence you need to have Git LFS installed.
For this, refer to the offical documentation: https://git-lfs.com

After you have installed Git LFS, you can pull the large test data file(s) of this repository.
```bash
git lfs pull
```

Then build and run the evaluator tool that uses the sample dataset and performs an initial evaluation:

### For the one with one metrics
```bash
cd tools/RF
docker build -t main-image .
docker run --rm --name main-container -t -p 4545:4545 main-image -p 4545 --curv-only  -m rf_model_Curv.joblib
```

### for the models with two metric 
```bash
cd tools/RF
docker build -t main-image .
docker run --rm --name main-container -t -p 4545:4545 main-image -p 4545   -m rf_model_RL1.joblib
```
##  Next step
After it you should open another terminal and put the evaluator command line : 
```bash
cd evaluator
docker build -t evaluator-image .
docker run --rm --name evaluator-container -t evaluator-image -u host.docker.internal:4545
```
