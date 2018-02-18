#include <stdio.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "math.h"
#include <time.h>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

//Directories
#define CASCADE_DIR  "haarcascade_frontalface_default.xml"
#define LR_DIR "LR/"
#define HR_DIR "HR/"
#define SAVE_DIR "Results/"

using namespace std;

int pixelValue(IplImage * image, int i, int j);
void writePixelF(IplImage * image, int i, int j, float val);
float euclideanNorm(IplImage * LRimage, IplImage * InpImg, int tl_x, int tl_y, int patch_size);
void checkCudaError(cudaError_t err);
void LoadDB(IplImage **& databaseHR,IplImage **& databaseLR, int &db_full);
void calcPixelWeight(int pixelWeight[15][15], int U, int V, int imrow, int imcol, int patch_size, int skip);
void detectFace(IplImage * InpImg, IplImage * LRface);
void setInputMatrices(IplImage * LRface, IplImage **& databaseLR, float tau, int sortedList[], float **&devPtrInpM, int db_full, int db_size, int U, int V, int imrow, int imcol, int patch_size, int skip);
void sortList(float eNormList[], int sortedList[], int patchNum, int db_size, int db_full);
void reconstructHR(float HRf[60][60], IplImage **& databaseHR, int pixelWeight[15][15], int sortedList[], float **&devPtrInvM, int db_size, int U, int V, int imrow, int imcol, int patch_size, int skip);

int main(int argc, char** argv)
{
	IplImage * InpImg = 0, *LRface, **databaseLR, **databaseHR, *HRface;
	CvSize LRsize, HRsize;
	clock_t fbegin, fend;
	float HRf[60][60], tau;
	int i, j, pixelWeight[15][15], db_size, U, V, patch_size, imrow, imcol, overlap, skip, *sortedList, db_full, batchsize;
	cublasHandle_t handle;
	cublasCreate(&handle);
	float **devPtrInpM, **d_devPtrInpM, **devPtrInvM, **d_devPtrInvM;
	int *d_pivotarray, *d_infoarray;
	string saveName;
	stringstream ss;
	
	db_size=atoi(argv[2]);//number of image patches that will be used to resolve each patch
	patch_size=atoi(argv[3]);
	overlap=atoi(argv[4]);//how many pixels of each patch should overlap with the patch in previous position
	tau = atof(argv[5]);//regularization parameter
	
	if(overlap>patch_size){
		printf("Error: overlap greater than patch size\n");
		return EXIT_FAILURE;
	}	

	imrow=imcol=15;
	skip = patch_size - overlap;
	U=ceil((imrow - overlap)/(float)skip);//number of patches in one row
	V=ceil((imcol - overlap)/(float)skip);//number of patches in one column
	batchsize=U*V;//total number of patches
	
	//Initialize Pivot & Info arrays for cuBLAS
	cudaMalloc((void **)&d_pivotarray, batchsize * db_size * sizeof(int));
	cudaMalloc((void **)&d_infoarray, batchsize * sizeof(int));
	
	//Initialize pointers and memory for input & inverse matrices
	devPtrInpM = (float **)malloc(batchsize * sizeof(*devPtrInpM));//devPtrInpM -> pointers -> matrices
	devPtrInvM = (float **)malloc(batchsize * sizeof(*devPtrInvM));
	for(i=0;i<batchsize;i++){
		cudaMalloc((void **) &devPtrInpM[i], db_size*db_size*sizeof(float));
		cudaMalloc((void **) &devPtrInvM[i], db_size*db_size*sizeof(float));
	}
	cudaMalloc((void **) &d_devPtrInpM, batchsize * sizeof(*devPtrInpM));
	cudaMalloc((void **) &d_devPtrInvM, batchsize * sizeof(*devPtrInvM));

	//Load Image database
	LoadDB(databaseHR, databaseLR, db_full);//db_full is the number of images in the database
	if(db_size>db_full){
		printf("Input database size is smaller than images required for solving\n");
		return EXIT_FAILURE;
	}	
	
	//get image sizes
	LRsize = cvGetSize(databaseLR[0]);
	HRsize = cvGetSize(databaseHR[0]);
	
	//calculate how many patches overlap on each pixel, the reciprocal will be used as pixel weight
	calcPixelWeight(pixelWeight, U, V, imrow, imcol, patch_size, skip);
	
	InpImg = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);//load input image
	LRface = cvCreateImage(LRsize, 8, 1);//create image for low resolution face
	
	//detect face or use input image directly, transferring to LRface
	if(*argv[6]=='y' || *argv[6]=='Y')
		detectFace(InpImg, LRface);
	else
		cvResize(InpImg, LRface, CV_INTER_CUBIC);

	HRface = cvCreateImage(HRsize, 8, 1);//create a blank image for HR face
	for(i=0;i<60;i++){//2D float array to hold HR face intensity values temporarily
		for(j=0;j<60;j++)
			HRf[i][j]=0;
	}
	fbegin = clock();
	//patch processing start
	printf("Begin Processing\n");
	sortedList = (int*)malloc(batchsize*db_size*sizeof(int));
	
	//finds closest image patches to each input patch and sets matrices to be inverted
	setInputMatrices(LRface, databaseLR, tau, sortedList, devPtrInpM, db_full, db_size, U, V, imrow, imcol, patch_size, skip);
	
	printf("Inversion\n");
	//copy arrays of matrix pointers to GPU
	cudaMemcpy(d_devPtrInpM, devPtrInpM, batchsize * sizeof(*devPtrInpM), cudaMemcpyHostToDevice);
	cudaMemcpy(d_devPtrInvM, devPtrInvM, batchsize * sizeof(*devPtrInvM), cudaMemcpyHostToDevice);
	
	//batched factorization and inversion
	cudaDeviceSynchronize();
	cublasSgetrfBatched(handle, db_size, d_devPtrInpM, db_size,d_pivotarray,d_infoarray,batchsize);
	cudaDeviceSynchronize();
	cublasSgetriBatched(handle, db_size, d_devPtrInpM, db_size, d_pivotarray, d_devPtrInvM, db_size, d_infoarray, batchsize);
	cudaDeviceSynchronize(); //can have effect on timing
	printf("HR reconstruction\n");
	
	//multiply with ones vector and reconstruct HR face
	reconstructHR(HRf, databaseHR, pixelWeight, sortedList, devPtrInvM, db_size, U, V, imrow, imcol, patch_size, skip);
	
	//transfer from 2D float array HRf to IplImage HR
	for(i=0;i<60;i++){
		for(j=0;j<60;j++){
			if(HRf[i][j]>255)
				writePixelF(HRface, j, i, 255);
			else if(HRf[i][j]<0)
				writePixelF(HRface, j, i, 0);
			else
				writePixelF(HRface, j, i, HRf[i][j]);
		}
	}
	
	fend = clock();
	printf("time taken: %fs\n", (double)(fend-fbegin) / CLOCKS_PER_SEC);
	ss << SAVE_DIR << argv[1]<< " db" << argv[2] << " ps" << argv[3] << " o" << argv[4] << " rp" << argv[5] << ".pgm" ;
	saveName = ss.str();
	cvSaveImage(saveName.c_str(), HRface);
	
	//Uncomment to print timing statistics to file
	/*FILE * pFile;
	pFile = fopen ("Results/timings.txt","a");
	if (pFile!=NULL)
	{
		fprintf (pFile, "%s\t\t\t%s\t\t%s\t\t%f\n",argv[2],argv[3],argv[4],(double)(fend-fbegin) / CLOCKS_PER_SEC);
		fclose (pFile);
	}*/
	
	//free memory
	cublasDestroy(handle);
	cvReleaseImage (&LRface);
	cvReleaseImage (&HRface);
	cvReleaseImage(&InpImg);
	free(devPtrInpM);
	free(devPtrInvM);
	free(sortedList);
	for(i=0;i<batchsize;i++){
		cudaFree(devPtrInpM[i]);
		cudaFree(devPtrInvM[i]);
	}
	cudaFree(d_devPtrInpM);
	cudaFree(d_devPtrInvM);
	cudaFree(d_infoarray);
	cudaFree(d_pivotarray);	
	for (i=0;i<db_full;i++){
		cvReleaseImage(&databaseLR[i]);
		cvReleaseImage(&databaseHR[i]);
	}

	return 0;
}

int pixelValue(IplImage * image, int i, int j){//retrieves pixel values
	return ((uchar *)(image->imageData + i*image->widthStep))[j*image->nChannels];
}

void writePixelF(IplImage * image, int i, int j, float val){//writes pixel values
	((uchar *)(image->imageData + i*image->widthStep))[j*image->nChannels] = (uchar)val;
	return;
}

float euclideanNorm(IplImage * LRimage, IplImage * InpImg, int tl_x, int tl_y, int patch_size){//calculates euclidean/L2 norm for a patch using top left x/y coordinates
	int i, j;
	float result=0;
	int nval=0;
	for (i=0; i<patch_size; i++){//loops over patch_size pixels in both x and y direction and squares the difference
		for(j=0; j<patch_size; j++){
			nval=pixelValue(InpImg, tl_y+j, tl_x+i)-pixelValue(LRimage,tl_y+j,tl_x+i);
			result+=nval*nval;
		}
	}
	return result;
}

void LoadDB(IplImage **& databaseHR,IplImage **& databaseLR, int &db_full){//loads HR and LR training image databases by reading from input.txt file
	string temp, HRfile, LRfile;
	int i;
	ifstream infile("input.txt");
	if(!infile.is_open()){
		cout << "could not open database input file" << endl;
		exit(1);
	}
	infile >> temp;//first line of input.txt specifies number of images in database
	db_full = atoi(temp.c_str());
	databaseLR = (IplImage **)malloc(db_full * sizeof(*databaseLR));
	databaseHR = (IplImage **)malloc(db_full * sizeof(*databaseHR));
	
	for(i=0;i<db_full;i++){
		infile >> temp;//input.txt contains the names of all images in database which are then loaded
		LRfile = LR_DIR + temp;//LR/HR directories defined on top
		HRfile = HR_DIR + temp;
		databaseLR[i]=cvLoadImage(LRfile.c_str(),0);
		databaseHR[i]=cvLoadImage(HRfile.c_str(),0);
	}
	infile.close();
}

void calcPixelWeight(int pixelWeight[15][15], int U, int V, int imrow, int imcol, int patch_size, int skip){
	int i,j,k,l,x,y;
	x=y=0;
	for(i=0;i<15;i++){//set to 0
		for(j=0;j<15;j++)
			pixelWeight[i][j]=0;
	}
	  
	for(i=0;i<V;i++){//go through each patch and add 1 to each pixel
		for(j=0;j<U;j++){
			for(k=0;k<patch_size;k++){
				for(l=0;l<patch_size;l++){
					pixelWeight[y+k][x+l]+=1;
				}
			}
			y+=skip;//increment y by skip (patch_size - overlap)
			if(y+patch_size>imrow)
				y=imrow-patch_size;//if skip puts new patch beyond image boundary, correct it by placing it just within boundary
		}
		y=0;//set y to 0 and start the next column
		x+=skip;//similar to y increment
		if(x+patch_size>imcol)
			x=imcol-patch_size;
	}
	//uncomment to print and check pixel weights
	/*
	for(i=0;i<15;i++){
		cout<<"[ ";
		for(j=0;j<15;j++){
			if(pixelWeight[i][j]<10)
				cout<<" ";
			cout<<pixelWeight[i][j]<<" ";
		}
		cout<<"]\n";
	}
	*/
}

void detectFace(IplImage * InpImg, IplImage * LRface){//detects face in InpImg and saves it in LRface
	IplImage * FaceCrop;
	CvHaarClassifierCascade * Cascade = 0;
	CvMemStorage * Storage = 0;
	CvSeq * FaceRectSeq;
	CvRect * FaceRect;
	
	Storage = cvCreateMemStorage(0);
	Cascade = (CvHaarClassifierCascade *)cvLoad((CASCADE_DIR),0, 0, 0 );//load cascade
	//detect face from 5x5 pixels to image size, storing faces in FaceRectSeq
	FaceRectSeq = cvHaarDetectObjects(InpImg, Cascade, Storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, cvSize(5,5), cvGetSize(InpImg));
	
	if(FaceRectSeq->total==0){//if no face found, exit
		printf("Error: No face detected in input image\n");
		exit(1);
	}
	printf("Faces detected: %d\n", FaceRectSeq->total);
	
	//crop first face detected	
	FaceRect = (CvRect*)cvGetSeqElem(FaceRectSeq, 1);//select first face
	cvSetImageROI(InpImg, *FaceRect);
	FaceCrop = cvCreateImage(cvGetSize(InpImg), 8, 1);
	cvCopy(InpImg, FaceCrop, NULL);//crop face and copy into FaceCrop
	cvResetImageROI(InpImg);
	cvResize(FaceCrop, LRface, CV_INTER_CUBIC);//resize FaceCrop and copy to LRface
	
	cvReleaseHaarClassifierCascade(&Cascade);//free memory
	cvReleaseMemStorage(&Storage);
	cvReleaseImage(&FaceCrop);
}

void setInputMatrices(IplImage * LRface, IplImage **& databaseLR, float tau, int sortedList[], float **&devPtrInpM, int db_full, int db_size, int U, int V, int imrow, int imcol, int patch_size, int skip){
	cublasHandle_t handle;
	cublasCreate(&handle);
	int i, j, k, l, x, y;
	float one, *eNormM, *DiffM, *d_DiffM, *eNormList;
	
	DiffM = (float*)malloc(db_size * patch_size * patch_size * sizeof(float));
	eNormM = (float*)malloc(db_size * db_size * sizeof(float));
	eNormList = (float*)malloc(db_full * sizeof(float));
	cudaMalloc((void **)&d_DiffM, db_size * patch_size * patch_size * sizeof(float));
	one = 1.0;
	
	x=y=0;
	for(i=0;i<V;i++){
		for(j=0;j<U;j++){
			for(k=0;k<db_full;k++)
			eNormList[k]=euclideanNorm(databaseLR[k], LRface, x, y, patch_size);//find euclidean distance of input patch and each training patch
			sortList(eNormList, sortedList, i*U+j, db_size, db_full);//sort out the closest db_size patches
			
			//calculating Difference and Euclidean Norm matrices
			memset(eNormM, 0, db_size*db_size*sizeof(float));
			for(k=0;k<db_size;k++){
				for(l=0;l<patch_size * patch_size;l++){
					DiffM[k * patch_size * patch_size + l]	= 
					(float)(pixelValue(LRface, y+l/patch_size, x+l%patch_size)-pixelValue(databaseLR[sortedList[(j+i*V)*db_size+k]], y+l/patch_size, x+l%patch_size));
					}//difference matrix subtracts each rasterized training patch from rasterized input patch duplicates
				eNormM[k + k*db_size] = euclideanNorm(databaseLR[sortedList[(j+i*V)*db_size+k]], LRface, x, y, patch_size);//diagonal euclidean norm matrix
				}
			cudaMemcpy(d_DiffM, DiffM, db_size * patch_size * patch_size * sizeof(float), cudaMemcpyHostToDevice); 
			cudaMemcpy(devPtrInpM[i*U+j], eNormM, db_size * db_size * sizeof(float), cudaMemcpyHostToDevice);
			//result matrix = transpose(DiffM) * DiffM + tau * eNormM
			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, db_size, db_size, patch_size * patch_size, &one, d_DiffM, patch_size * patch_size, d_DiffM, patch_size * patch_size, &tau, devPtrInpM[i*U+j], db_size);

			y+=skip;//increment x and y for new patch positions
			if(y+patch_size>imrow)
			y=imrow-patch_size;
		}
		y=0;
		x+=skip;
		if(x+patch_size>imcol)
			x=imcol-patch_size;
	}
	free(DiffM);
	free(eNormList);
	free(eNormM);
	cudaFree(d_DiffM);
	cublasDestroy(handle);
}

void sortList(float eNormList[], int sortedList[], int patchNum, int db_size, int db_full){
	float minval, maxval;
	int minidx, i, j;
	maxval=0;
	for(i=0;i<db_full;i++){//find maximum value
		if(eNormList[i]>maxval)
			maxval=eNormList[i];
	}
	for(i=0;i<db_size;i++){
		minval = maxval;
		for(j=0;j<db_full;j++){//find smallest value, skip previously used values marked -1
			if(eNormList[j] !=-1 && eNormList[j]<minval){
			minidx=j;
			minval=eNormList[minidx];
			}
		}
	eNormList[minidx]=-1;//set smallest value to -1 to skip
	sortedList[patchNum*db_size+i]=minidx;//add index to the sorted list
	}
}

void reconstructHR(float HRf[60][60], IplImage **& databaseHR, int pixelWeight[15][15], int sortedList[], float **&devPtrInvM, int db_size, int U, int V, int imrow, int imcol, int patch_size, int skip){
	cublasHandle_t handle;
	cublasCreate(&handle);
	int i, j, k, l, m, n, o, x, y;
	float *onesV, *weightsV, *zeroesV, *d_devPtrOnesV, *d_devPtrWeightsV, one, weightSum;
	one = 1.0;
	onesV = (float*)malloc(db_size*sizeof(float));//onesV=[1 1 1...]
	for(i=0;i<db_size;i++)
		onesV[i]=1.0;
	cudaMalloc((void **)&d_devPtrOnesV, db_size*sizeof(float));
	cudaMemcpy(d_devPtrOnesV, onesV, db_size * sizeof(float), cudaMemcpyHostToDevice);
	weightsV = (float*)malloc(db_size*sizeof(float));//weightsV = weights vector
	memset(weightsV, 0, db_size*sizeof(float));
	zeroesV = (float*)malloc(db_size*sizeof(float));//zeroesV = [0 0 0...], used to reset GPU memory after solving for weights
	memset(zeroesV, 0, db_size*sizeof(float));
	cudaMalloc((void **)&d_devPtrWeightsV, db_size*sizeof(float));
	cudaMemcpy(d_devPtrWeightsV, zeroesV, db_size * sizeof(float), cudaMemcpyHostToDevice);
	
	x=y=0;
	for(i=0;i<V;i++){
		for(j=0;j<U;j++){
		cublasSgemv(handle, CUBLAS_OP_N, db_size, db_size, &one, devPtrInvM[i*V+j], db_size, d_devPtrOnesV, 1, &one, d_devPtrWeightsV, 1);//weightsV = Inverted Matrix * onesV
		cudaMemcpy(weightsV, d_devPtrWeightsV, db_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_devPtrWeightsV, zeroesV, db_size * sizeof(float), cudaMemcpyHostToDevice);	
		
			weightSum=0;//calculate sum of weights
			for(k=0;k<db_size;k++)
				weightSum+=weightsV[k];
			
			//fill HRf with pixel values
			for(k=0;k<patch_size;k++){//for each LR pixel
				for(l=0;l<patch_size;l++){
					for(m=0;m<4;m++){//fill 4 (magnification factor) HR pixels
						for(n=0;n<4;n++){
							for(o=0;o<db_size;o++){//HR pixel += normalized weight * HR training image / number of patches overlapping pixel
								HRf[(4*(x+k))+m][(4*(y+l))+n]+=
									weightsV[o]/weightSum * (float)pixelValue(databaseHR[sortedList[(j+i*V)*db_size+o]], (4*(y+l))+n, (4*(x+k))+m) / pixelWeight[x+k][y+l];
							}
						}
					}
				}
			}
		y+=skip;//increment x and y for position of next patch
		if(y+patch_size>imrow)
			y=imrow-patch_size;
		}
		y=0;
		x+=skip;
		if(x+patch_size>imcol)
			x=imcol-patch_size;
	}
	cublasDestroy(handle);
	free(onesV);
	free(zeroesV);
	free(weightsV);
	cudaFree(d_devPtrOnesV);
	cudaFree(d_devPtrWeightsV);
}