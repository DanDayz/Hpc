# include <bits/stdc++.h>
# include <cuda.h>

#define TILE_WIDTH 32

using namespace std;
// ::::::::::::::::::::::::::::::::::::::::::GPU::::::::::::::::::::::::::::::::

__global__ void KernelNormalMul(int *Mat1,int *Mat2,int *Mat3,int m,int n,int p){
  int j = threadIdx.y + blockDim.y * blockIdx.y; // cols
  int i = threadIdx.x + blockDim.x * blockIdx.x; // row

  if((i<m) && (j<p)){
    int value=0;
    for(int k=0;k<n;k++){
      value+=Mat1[n*i+k]*Mat2[p*k+j];
    }
    Mat3[p*i+j]=value;
  }
}


__global__ void  KernelTilesMul(int *Mat1,int *Mat2,int *Mat3,int m,int n,int p){

      __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
      __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];
      int bx = blockIdx.x;
      int by = blockIdx.y;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int row = by * TILE_WIDTH + ty;
      int col = bx * TILE_WIDTH + tx;
      int Pvalue = 0;


        for(int k = 0; k < (m + TILE_WIDTH-1)/TILE_WIDTH; ++k){

          if(k*TILE_WIDTH + tx < n && row < m){
            Mds[ty][tx] = Mat1[row*n + k*TILE_WIDTH + tx];
          }else{
            Mds[ty][tx] = 0;
          }
          if(k*TILE_WIDTH + threadIdx.y < n && col < p){
            Nds[ty][tx] = Mat2[(k*TILE_WIDTH + ty) * p + col];
          }else{
            Nds[ty][tx] =0;
          }

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k){
          Pvalue += Mds[ty][k] * Nds[k][tx];
        }
          __syncthreads();
      }

      if (row < m && col < p){
        Mat3[row*p+col] = Pvalue;
        }

}



void d_MatrixMult(int *Mat1,int *Mat2,int *Mat3,int m,int n,int p, int op ){
  int * d_Mat1;
  int * d_Mat2;
  int * d_Mat3;
  float Blocksize=32; // Bloque de 2 dimensiones 32*32=256  número de blokes= 1024 (1024/256=4)
  int size1=m*n;
  int size2=n*p;
  int size3=m*p;

  // 1. Separamos memoria en el device
  cudaMalloc(&d_Mat1,size1*sizeof(int));
  cudaMalloc(&d_Mat2,size2*sizeof(int));
  cudaMalloc(&d_Mat3,size3*sizeof(int));

  // 2. Copiamos el valor de las variables de host a las variables del device.
  cudaMemcpy(d_Mat1, Mat1,size1*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mat2, Mat2,size2*sizeof(int),cudaMemcpyHostToDevice);
  // 3. Lógica de bloques e hilos, elementos para realizar la parelelización.
  dim3 dimGrid(ceil(m/Blocksize),ceil(p/Blocksize),1);
  //dim3 dimGrid((m+Blocksize-1)/Blocksize,(p+Blocksize-1)/Blocksize,1);
  dim3 dimBlock(Blocksize,Blocksize,1);
  // 4. Invocación del kernel (invoción del host, ejecutadas en el device), <<<<#dimGrid,#dimBlock>>>
  if(op==1){KernelNormalMul<<<dimGrid,dimBlock>>>(d_Mat1,d_Mat2,d_Mat3,m,n,p);}else{
  	KernelTilesMul<<<dimGrid,dimBlock>>>(d_Mat1,d_Mat2,d_Mat3,m,n,p);
  }
  // 5. Copiamos el resultado para mostrar en el I/O del host.
  cudaMemcpy (Mat3,d_Mat3,size3*sizeof(int),cudaMemcpyDeviceToHost);
  // 6. Liberamos memoria.
  cudaFree(d_Mat3);
}

// :::::::::::::::::::::::::::::::::::::::Normal::::::::::::::::::::::::::::::::


void h_Mul_Mat(int *Mat1,int *Mat2, int *Mat3,int m,int n,int p){

  for(int i=0;i<m;i++){
    for(int j=0;j<p;j++){
      int value=0;
      for(int k=0;k<n;k++){
        value+=Mat1[n*i+k]*Mat2[p*k+j];
        }
			Mat3[p*i+j]=value;
    }
  }
}

void llena_mat(int *Mat, int Value,int m,int n){// ver matriz como chorizo grande
 int size=n*m; // matriz lineal
  for(int i =0 ; i<size ; i++){
          Mat[i]=Value;
      }
}

void mostrar_mat(int *Mat,int m,int n){// ver matriz como chorizo grande
 int size=n*m; // matriz lineal
  for (int i=0;i<size;i++) {
      if(i%n==0 && n!=0){
        cout<<endl;
      }
    cout<<"["<<Mat[i]<<"] ";
  }
  cout<<endl;
}


int check_mat(int *Mat1,int *Mat2,int m,int p){
  for(int i=0; i<(m*p);i++){
    if(Mat1[i]!=Mat2[i]){
      cout<<"Error, Las matrices no son iguales"<<endl;
      return 1;
    }
  }
  return 0;
}


// :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::

double diffclock(clock_t clock1,clock_t clock2){
    double diffticks=clock2-clock1;
    double diffms=(diffticks)/(CLOCKS_PER_SEC/1); //  /1000 mili
    return diffms;
}

// :::::::::::::::::::::::::::::::::::::::Main::::::::::::::::::::::::::::::::.

int main(){
  // Malloc   (Fila,Columna)
	double T1,T2,T3;

  int n=32; // columna Mat1, fila Mat2
  int m=64; // Fila Mat1 , Fila Mat3
  int p=64; // colunma Mat2m Columna Mat3
  int *Mat1 = (int*)malloc((m*n)*sizeof(int));
  int *Mat2 = (int*)malloc((n*p)*sizeof(int));
  int *Mat3 = (int*)malloc((m*p)*sizeof(int));
  int *Mat4 = (int*)malloc((m*p)*sizeof(int));
  int *Mat5 = (int*)malloc((m*p)*sizeof(int));

  llena_mat(Mat1,1,m,n);
  llena_mat(Mat2,2,n,p);

	clock_t start = clock();
  h_Mul_Mat(Mat1,Mat2,Mat3,m,n,p);
  clock_t end = clock();
  T1=diffclock(start,end);
  cout <<"Tiempo secuencial: "<<T1<<endl;
  //mostrar_mat(Mat3,m,p);
  clock_t start2 = clock();
  d_MatrixMult(Mat1,Mat2,Mat4,m,n,p,1); // paralelo
  clock_t end2 = clock();
  //mostrar_mat(Mat4,m,p);
  T2=diffclock(start2,end2);
  cout <<"Tiempo Paralelo: "<<T2<<endl;
  cout<<"Aceleración lograda: "<<T1/T2<<endl;

  if(check_mat(Mat3,Mat4,m,p)==0){
   cout<<"Matrices M1 Y M2 son iguales"<<endl;
  }

  clock_t start3 = clock();
  d_MatrixMult(Mat1,Mat2,Mat5,m,n,p,2); // tiles
  //mostrar_mat(Mat5,m,p);
  clock_t end3 = clock();
  T3=diffclock(start3,end3);

  cout <<"Tiempo Paralelo con Tiles: "<<T3<<endl;
	cout<<"Aceleración lograda Respecto a el tiempo paralelo: "<<T2/T3<<endl;

   if(check_mat(Mat4,Mat5,m,p)==0){
   cout<<"Matrices M2 Y M3 son iguales"<<endl;
  }

return 0;
}

// http://www.techdarting.com/2014/03/matrix-multiplication-in-cuda-using.html
