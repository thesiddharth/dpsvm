---
#DPSVM#
---

###An OpenMPI and CUDA based, distributed, parallel SVM implementation###

We have implemented a distributed and parallel Support Vector Machine training algorithm for binary classification using the OpenMPI library and the CUDA parallel programming model. We parallelize the modified Sequential Minimal Optimization algorithm used by popular tools like LIBSVM, and distribute the parallelism over GPUs in a cluster.


---
**Dependencies**

 - [OpenMPI](http://www.open-mpi.org/)
 - [CUDA](https://developer.nvidia.com/cuda-toolkit)
 - [Thrust](https://developer.nvidia.com/Thrust)
 - [cuBLAS](https://developer.nvidia.com/cuBLAS)
 - [CBLAS](http://www.netlib.org/blas/blast-forum/cblas.tgz)
 - [BLAS](http://www.netlib.org/blas/blas.tgz)

---
**Performance**

**DPSVM** took *137 seconds* in training on the [MNIST](http://yann.lecun.com/exdb/mnist/) even-odd dataset, on a single Nvidia GTX 780 GPU. This reduced to *46 seconds*, when using 10 GPUs via OpenMPI on a cluster connected with an Ethernet backbone. The gaussian kernel paramters chosen were c = 10 and gamma = 0.25, with a tolerance of 0.001. Herrero-Lopez et. al. use a single Nvidia Tesla C1060 GPU, whereas the Spark implementation was the SVMWithSGD() call from MLLib on 10 AWS m3.2xlarge machines.

The same job using **LibSVM** takes *13,963 seconds*, using an Intel Core i7 920.

The DPSVM training phase has the same number of Support Vectors as LibSVM.

![alt tag](https://github.com/thesiddharth/dpsvm/blob/gh-pages/images/mnist_1.png)
