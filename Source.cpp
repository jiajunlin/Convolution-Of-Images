
#include <iostream>
#include <cstdlib>   // for rand(), srand()
#include <ctime>     // for time()
#include <fstream>
#include <vector>
using namespace std;

class ConvNet
{
private: 
	vector<vector<vector<float>>> D1, D2, D5, D6, D7, D10, D11, D12, D15, D16;
	vector<vector<vector<vector<float>>>> D3, D8, D13, D18;
	vector<float> D4, D9, D14, D17, D19, D20;

public: 
	

	void M1(vector<vector<vector<float>>>& tn3d, int s1, int s2, int s3, istream& input_image);
	void M2(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4, istream& input_file);
	void M3(vector<float>& tn1d, int s1, istream& input_file);
	void M4(vector<vector<vector< float >>>& D1, vector<vector<vector< float>>>& D2,
		vector<vector<vector< vector< float >>>>& D3, vector< float >& D4, ofstream& file);
	void M5(vector<vector<vector< float >>>& D2, vector<vector<vector< float >>>& D5);
	void M6(vector<vector<vector< float >>>& D5, vector<vector<vector< float >>>& D6);
	void M7(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4, istream& input_file);
	void M8(vector<float>& tn1d, int s1, istream& input_file);
	void M9(vector<vector<vector< float >>>& D6, vector<vector<vector< float>>>& D7,
		vector<vector<vector< vector< float >>>>& D8, vector< float >& D9, ofstream& file);
	void M10(vector<vector<vector< float >>>& D7, vector<vector<vector< float >>>& D10);
	void M11(vector<vector<vector< float >>>& D10, vector<vector<vector< float >>>& D11);
	void M12(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4, istream& input_file);
	void M13(vector<float>& tn1d, int s1, istream& input_file);
	void M14(vector<vector<vector< float >>>& D12, vector<vector<vector< float>>>& D11,
		vector<vector<vector< vector< float >>>>& D13, vector< float >& D14, ofstream& file);
	void M15(vector<vector<vector< float >>>& D12, vector<vector<vector< float >>>& D15);
	void M16(vector<vector<vector< float >>>& D15, vector<vector<vector< float >>>& D16);
	void M17(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4, istream& input_file);
	void M18(vector<float>& tn1d, int s1, istream& input_file);
	void M19(vector< float >& D17, vector<vector<vector< float >>>& D16,
		vector<vector<vector< vector< float >>>>& D18, vector< float >& D19, ofstream& file);
	void M20(vector < float >& D17);
	void M21(vector < float >& D17, ofstream& file);
	void M22(vector < float >& D17, vector < float >& D20, ofstream& file);

	//layers
	void Layer1(ifstream& file);
	void Layer2(ifstream& infile, ofstream& outfile);
	void Layer3(ofstream& outfile);
	void Layer4();
	void Layer5(ifstream& infile, ofstream& outfile);
	void Layer6(ifstream& infile, ofstream& outfile);
	void Layer7();
	void Layer8(ifstream& infile, ofstream& outfile);
	void Layer9();
	void Layer10();
	void Layer11(ifstream& infile, ofstream& outfile);
	void Layer12(ofstream& outfile);


};


//M1 reads input file and initializes the input layer in 1a.
void ConvNet::M1(vector<vector<vector<float>>>& tn3d, int s1, int s2, int s3, istream& input_image)
	{
		for (int i3 = 0; i3 < s3; i3++) {
			for (int i2 = 0; i2 < s2; i2++) {
				for (int i1 = 0; i1 < s1; i1++)
				{
					input_image >> tn3d[i1][i2][i3];	//stores the 3d
				}
			}
		}
	}

// read a 4d tensor tn4d of size s1, s2, s3, s4
void ConvNet::M2(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4, istream& input_file) 
{
	for (int i4 = 0; i4 < s4; i4++) {
		for (int i3 = 0; i3 < s3; i3++) {
			for (int i2 = 0; i2 < s2; i2++) {
				for (int i1 = 0; i1 < s1; i1++) {
					input_file >> tn4d[i1][i2][i3][i4];
				}
			}
		}
	}
}

// read data for a 1d tensor
void ConvNet::M3(vector<float>& tn1d, int s1, istream& input_file) 
{
	for (int i1 = 0; i1 < s1; i1++)
	{
		input_file >> tn1d[i1];
	}
}

//computing for D2 by convolving D1 with D3 and adding D4, then storing output to D2.
//set stride to 1	
//D2 is 32*32*16 output of L2			tn3
//D1 is 32*32*3							tn1
//D3 is 5*5*3*16 stores the 16 convultion filters with zero padding.	tn2
//D4 is 16*1	 Bias vector 
void ConvNet::M4(vector<vector<vector< float >>>& D1, vector<vector<vector< float>>>& D2, 
	vector<vector<vector< vector< float >>>>& D3, vector< float >& D4, ofstream& file)
{
	int stride = 1; 
	int D1s1 = 32, D1s2 = 32, D1s3 = 3; 
	int D2s1 = 32, D2s2 = 32, D2s3 = 16; 
	int D3s1 = 5, D3s2 = 5, D3s3 = 3, D3s4 = 16;
	int D3s1by2 = D3s1 / 2; 
	int D3s2by2 = D3s2 / 2; 

	file << "Output of Convolution layer: D2" << endl;
	for (int D3i4 = 0, D2i3 = 0; D3i4 < D3s4; D3i4++, D2i3++) {
		for (int D1i1 = 0, D2i1 = 0; D1i1 < D1s1; D1i1 += stride, D2i1++) {
			for (int D1i2 = 0, D2i2 = 0; D1i2 < D1s2; D1i2 += stride, D2i2++) {
				float tmpsum = 0.0;
				for (int D3i3 = 0; D3i3 < D3s3; D3i3++) {
					// note D1s3=D3s3
					for (int D3i1 = -D3s1by2; D3i1 <= D3s1by2; D3i1++) {
						for (int D3i2 = -D3s2by2; D3i2 <= D3s2by2; D3i2++) {
							if (((D1i1 + D3i1) >= 0) && ((D1i1 + D3i1) < D1s1) 
								&& ((D1i2 + D3i2) >= 0) && ((D1i2 + D3i2) < D1s1)) { // zero padding of tn1
								tmpsum += D3[D3i1 + D3s1by2][D3i2 + D3s2by2][D3i3][D3i4] 
									* D1[D1i1 + D3i1][D1i2 + D3i2][D3i3];
							}
						}
					}
				}
				D2[D2i1][D2i2][D2i3] = tmpsum + D4[D2i3];
				file << D2[D2i1][D2i2][D2i3] << "  ";
			}
			file << endl;
		}
		file << endl << endl;
	}
}

//Computes D5 using D2. Z = max(0, x)
//D5 is a 32*32*16 array 
//D2 32*32*16 array
void ConvNet::M5(vector<vector<vector< float >>>& D2, vector<vector<vector< float >>>& D5)
{
	int s1 = 32, s2 = 32, s3 = 16;

	for (int i3 = 0; i3 < s3; i3++) {
		for (int i2 = 0; i2 < s2; i2++) {
			for (int i1 = 0; i1 < s1; i1++)
			{
				D5[i1][i2][i3] = fmax(0.00, D2[i1][i2][i3]); 
			}
		}
	}
}

//D6 16x16x16 array for storing output
//M6 Proccess D5 to compute and store D6. Each element is the max of 4 elements in a 2x2 block
//with stride 2.

float maximum(float e1, float e2, float e3, float e4) {
	float max = INT_MIN;
	vector <float> nums;
	nums.push_back(e1);
	nums.push_back(e2);
	nums.push_back(e3);
	nums.push_back(e4);
	for (int i = 0; i < 4; i++) {
		if (nums[i] > max)
			max = nums[i];
	}

	return max;
}

void ConvNet::M6(vector<vector<vector< float >>>& D5, vector<vector<vector< float >>>& D6)
{

	int stride = 2; 
	for (int k = 0; k < 16; k++) {  // for each cross section k
		for (int m = 0, i = 0; m < 32; m += stride, i++) { //for row m
			for (int n = 0, j = 0; n < 32; n += stride, j++) { // for column n
				D6[i][j][k] = maximum(D5[m][n][k], D5[m + 1][n][k], D5[m][n + 1][k], D5[m + 1][n + 1][k]);
			}
		}
	}
}
	//D7 : 16x16x20 array of floating point numbers to store the output of L5
	//D8 : 5x5x16x20  array for storing the 20 convolution filters with zero padding.
	//D9 : 20x1 array for storing the bias vector
	//M7 : for reading input file to initialize the data member D8 in L5.
	//M8 : for reading input file to initialize the data member D9 in L5.
	//M9 : for computing the data member D7 by convolving D6  with D8, stride = 1,
	//		and adding D9, and storing the output in D7.

void ConvNet::M7(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4, istream& input_file)
{
	for (int i4 = 0; i4 < s4; i4++) {
		for (int i3 = 0; i3 < s3; i3++) {
			for (int i2 = 0; i2 < s2; i2++) {
				for (int i1 = 0; i1 < s1; i1++) {
					input_file >> tn4d[i1][i2][i3][i4];
				}
			}
		}
	}
}

//M8 (20*1): for reading input file to initialize the data member D9 in L5.
void ConvNet::M8(vector<float>& tn1d, int s1, istream& input_file)
{
	for (int i1 = 0; i1 < s1; i1++)
	{
		input_file >> tn1d[i1];
	}
}

//M9 : for computing the data member D7 by convolving D6  with D8, stride = 1,
//		and adding D9, and storing the output in D7.
//D7	16*16*20
//D6	16*16*16														
//D8	5*5*16*20
//D9	20*1
void ConvNet::M9(vector<vector<vector< float >>>& D6, vector<vector<vector< float>>>& D7,
	vector<vector<vector< vector< float >>>>& D8, vector< float >& D9, ofstream& file)
{
	int stride = 1;
	int D6s1 = 16, D6s2 = 16, D6s3 = 16;			//D1
	int D7s1 = 16, D7s2 = 16, D7s3 = 20;			//D2
	int D8s1 = 5, D8s2 = 5, D8s3 = 16, D8s4 = 20;	//D3
	int D8s1by2 = D8s1 / 2;		
	int D8s2by2 = D8s2 / 2;

	file << "Output of Convolution layer: D7\n" << endl;
	for (int D8i4 = 0, D7i3 = 0; D8i4 < D8s4; D8i4++, D7i3++) {
		for (int D6i1 = 0, D7i1 = 0; D6i1 < D6s1; D6i1 += stride, D7i1++) {
			for (int D6i2 = 0, D7i2 = 0; D6i2 < D6s2; D6i2 += stride, D7i2++) {
				float tmpsum = 0.0;
				for (int D8i3 = 0; D8i3 < D8s3; D8i3++) {
					// note D6s3=D8s3
					for (int D8i1 = -D8s1by2; D8i1 <= D8s1by2; D8i1++) {
						for (int D8i2 = -D8s2by2; D8i2 <= D8s2by2; D8i2++) {
							if (((D6i1 + D8i1) >= 0) && ((D6i1 + D8i1) < D6s1)
								&& ((D6i2 + D8i2) >= 0) && ((D6i2 + D8i2) < D6s1)) { // zero padding of tn1
								tmpsum += D8[D8i1 + D8s1by2][D8i2 + D8s2by2][D8i3][D8i4]
									* D6[D6i1 + D8i1][D6i2 + D8i2][D8i3];
							}
						}
					}
				}
				D7[D7i1][D7i2][D7i3] = tmpsum + D9[D7i3];
				file << D7[D7i1][D7i2][D7i3] << "  ";
			}
			file << endl;
		}
		file << endl << endl;
	}
}

//Layer 6: ReLu 
//Data member D10 : 16x16x20 array for storing the output of this layer L6.
//Method member M10 : Computes and stores D10 using D7.Z = max(0, x).
void ConvNet::M10(vector<vector<vector< float >>>& D7, vector<vector<vector< float >>>& D10)
{

	int s1 = 16, s2 = 16, s3 = 20;

	for (int i3 = 0; i3 < s3; i3++) {
		for (int i2 = 0; i2 < s2; i2++) {
			for (int i1 = 0; i1 < s1; i1++)
			{
				D10[i1][i2][i3] = fmax(0.00, D7[i1][i2][i3]);
			}
		}
	}
}

//Layer L7 : Maxpooling: filter size 2x2, stride = 2.
//Data member D11 : 8x8x20 array for storing the output of this layer L7.
//Method member M11 : Process D10 to computeand store D11.Each element is the maximum of 2x2 block
//with stride 2.

void ConvNet::M11(vector<vector<vector< float >>>& D10, vector<vector<vector< float >>>& D11)
{
	int stride = 2;
	for (int k = 0; k < 20; k++) {  // for each cross section k
		for (int m = 0, i = 0; m < 16; m += stride, i++) { //for row m
			for (int n = 0, j = 0; n < 16; n += stride, j++) { // for column n
				D11[i][j][k] = maximum(D10[m][n][k], D10[m + 1][n][k], D10[m][n + 1][k], D10[m + 1][n + 1][k]);
			}
		}
	}
}

//to initialize data member D13 in L8 
void ConvNet::M12(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4, istream& input_file)
{
	for (int i4 = 0; i4 < s4; i4++) {
		for (int i3 = 0; i3 < s3; i3++) {
			for (int i2 = 0; i2 < s2; i2++) {
				for (int i1 = 0; i1 < s1; i1++) {
					input_file >> tn4d[i1][i2][i3][i4];
				}
			}
		}
	}
}


//M13 (20*1): for reading input file to initialize the data member D9 in L5.
void ConvNet::M13(vector<float>& tn1d, int s1, istream& input_file)
{
	for (int i1 = 0; i1 < s1; i1++)
	{
		input_file >> tn1d[i1];
	}
}

//M14: for computing the data member D12 by convolving D11  with D13, stride=1, 
//		and adding D14, and storing the output in D12.
//D12 : 8x8x20 array of floating point numbers to store the output of L8
//D13 : 5x5x20x20  array for storing the 20 convolution filters with zero padding.
//D14 : 20x1 array for storing the bias vector

void ConvNet::M14(vector<vector<vector< float >>>& D11, vector<vector<vector< float>>>& D12,
	vector<vector<vector< vector< float >>>>& D13, vector< float >& D14, ofstream& file)
{
	int stride = 1;
	int D11s1 = 8, D11s2 = 8, D11s3 = 20;	//D6
	int D12s1 = 8, D12s2 = 8, D12s3 = 20;	//D7
	int D13s1 = 5, D13s2 = 5, D13s3 = 20, D13s4 = 20;	//D8
	int D13s1by2 = D13s1 / 2;
	int D13s2by2 = D13s2 / 2;

	file << "Output of Convolution layer: D12" << endl;
	for (int D13i4 = 0, D12i3 = 0; D13i4 < D13s4; D13i4++, D12i3++) {
		for (int D11i1 = 0, D12i1 = 0; D11i1 < D11s1; D11i1 += stride, D12i1++) {
			for (int D11i2 = 0, D12i2 = 0; D11i2 < D11s2; D11i2 += stride, D12i2++) {
				float tmpsum = 0.0;
				for (int D13i3 = 0; D13i3 < D13s3; D13i3++) {
					// note D11s3=D13s3
					for (int D13i1 = -D13s1by2; D13i1 <= D13s1by2; D13i1++) {
						for (int D13i2 = -D13s2by2; D13i2 <= D13s2by2; D13i2++) {
							if (((D11i1 + D13i1) >= 0) && ((D11i1 + D13i1) < D11s1)
								&& ((D11i2 + D13i2) >= 0) && ((D11i2 + D13i2) < D11s1)) { // zero padding of tn1
								tmpsum += D13[D13i1 + D13s1by2][D13i2 + D13s2by2][D13i3][D13i4]
									* D11[D11i1 + D13i1][D11i2 + D13i2][D13i3];
							}
						}
					}
				}
				D12[D12i1][D12i2][D12i3] = tmpsum + D14[D12i3];
				file << D12[D12i1][D12i2][D12i3] << "  ";
			}
			file << endl;
		}
		file << endl << endl;
	}
}

//Layer 9: ReLu 
//Data member D15 : 8x8x20 array for storing the output of this layer L6.
//Method member M10 : Computes and stores D15 using D12.Z = max(0, x).
void ConvNet::M15(vector<vector<vector< float >>>& D12, vector<vector<vector< float >>>& D15)
{

	int s1 = 8, s2 = 8, s3 = 20;

	for (int i3 = 0; i3 < s3; i3++) {
		for (int i2 = 0; i2 < s2; i2++) {
			for (int i1 = 0; i1 < s1; i1++)
			{
				D15[i1][i2][i3] = fmax(0.00, D12[i1][i2][i3]);
			}
		}
	}
}

//D16 8x8x20
void ConvNet::M16(vector<vector<vector< float >>>& D15, vector<vector<vector< float >>>& D16)
{
	int stride = 2;
	for (int k = 0; k < 20; k++) {  // for each cross section k
		for (int m = 0, i = 0; m < 8; m += stride, i++) { //for row m
			for (int n = 0, j = 0; n < 8; n += stride, j++) { // for column n
				D16[i][j][k] = maximum(D15[m][n][k], D15[m + 1][n][k], D15[m][n + 1][k], D15[m + 1][n + 1][k]);
			}
		}
	}
}

//to initialize data member D13 in L8 
void ConvNet::M17(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4, istream& input_file)
{
	for (int i4 = 0; i4 < s4; i4++) {
		for (int i3 = 0; i3 < s3; i3++) {
			for (int i2 = 0; i2 < s2; i2++) {
				for (int i1 = 0; i1 < s1; i1++) {
					input_file >> tn4d[i1][i2][i3][i4];
				}
			}
		}
	}
}

//M18 (10*1): for reading input file to initialize the data member D19 in L11.
void ConvNet::M18(vector<float>& tn1d, int s1, istream& input_file)
{
	for (int i1 = 0; i1 < s1; i1++)
	{
		input_file >> tn1d[i1];
	}
}

//D17: output array of size 10 to store layer 11.					tn5
//D18: 4x4x20x10 array for storing 10 full connection filters		tn4
//D19: array of 10 bias vector										bias2
//D16: 4*4*20 output from maxpooling								tn1
//M19: for computing the data member D17 by taking dot-product of D16  
//		with the 10 different 4x4x20 filters stored in D18,
//		and adding D19, and storing the output in D17.


int D18s1 = 4, D18s2 = 4, D18s3 = 20, D18s4 = 10;
void ConvNet::M19(vector< float >& D17, vector<vector<vector< float >>>& D16,
	vector<vector<vector< vector< float >>>>& D18, vector< float >& D19, ofstream& file)
{
	file << "Output of M19" << endl; 
	
	for (int D18i4 = 0; D18i4 < D18s4; D18i4++) {
		// note tn1s1=tn4s1 tn1s2=tn4s2, tn1s3=tn4s3
		float tmpsum = 0.0;
		for (int D18i1 = 0; D18i1 < D18s1; D18i1++) {
			for (int D18i2 = 0; D18i2 < D18s2; D18i2++) {
				for (int D18i3 = 0; D18i3 < D18s3; D18i3++) {
					tmpsum += D18[D18i1][D18i2][D18i3][D18i4] * D16[D18i1][D18i2][D18i3];
				}
			}
		}
		D17[D18i4] = tmpsum + D19[D18i4];
		file << D17[D18i4] << "  ";
	}
	file << endl;
}



//D17: output array of size 10 to store layer 11.					tn5
//D18: 4x4x20x10 array for storing 10 full connection filters		tn4
//D19: array of 10 bias vector										bias2
//D16: 4*4*20 output from maxpooling								tn1
//M19: for computing the data member D17 by taking dot-product of D16  
//		with the 10 different 4x4x20 filters stored in D18,
//		and adding D19, and storing the output in D17.
//Layer L12 : Softmax layer
//D20 : array of size 10 to store the output of this layer L12
//M20 : In order to avoid taking the exponent of large numbers that cause overflow, 
//		normalize the contents of D17, by dividing each element by the 
//		square - root of the sum of the squares of each element in D17.Use this result in computing 
//		probabilities in the next method M21.
void ConvNet::M20(vector < float >& D17)
{ 

	// Computing softmax of tn5 after normalizing
	float tmpsum = 0.0;
	for (int i = 0; i < D18s4; i++) {
		tmpsum += (D17[i] * D17[i]);
	}
	tmpsum = sqrt(tmpsum);
	for (int i = 0; i < D18s4; i++) {
		D17[i] /= tmpsum;
	}

}

float tmpsum = 0.0;
void ConvNet::M21(vector < float > & D17, ofstream& file)
{
	// compute softmax
	file << "\nComputing softmax" << endl;
	for (int i = 0; i < D18s4; i++) {
		tmpsum += (exp(D17[i]));
		file << tmpsum << " ";
	}
	file << endl; 
}

void ConvNet::M22(vector < float >& D17, vector < float >& D20, ofstream& file)
{
	file << "\nSoftmax probabilities or Output of D20: " << endl;
	for (int i = 0; i < D18s4; i++) {
		D20[i] = (exp(D17[i])) / tmpsum;
		file << D20[i] << "   ";
	}
}

void alloc1d(vector<float>& tn1d, int s1) {
	// allocate memory for a 1d tensor tn1d of size s1
	tn1d.resize(s1);
}

// allocate memory for a 3d tensor
void alloc3d(vector<vector<vector<float>>>& tn3d, int s1, int s2, int s3) 
{
	// allocate memory for a 3d tensor tn3d of size s1, s2, s3
	tn3d.resize(s1);
	for (int i1 = 0; i1 < s1; i1++) {
		tn3d[i1].resize(s2);
		for (int i2 = 0; i2 < s2; i2++) {
			tn3d[i1][i2].resize(s3);
		}
	}
}

// print data for a 1d tensor
void print1d(vector<float>& tn1d, int s1) {
	cout << endl << endl;
	for (int i1 = 0; i1 < s1; i1++)
	{
		cout << tn1d[i1] << "   ";
	}
	cout << endl << endl;
}

// print data for a 3d tensor
void print3d(vector<vector<vector< float >>>& tn3d, int s1, int s2, int s3) 
{
	cout << endl << endl;
	for (int i3 = 0; i3 < s3; i3++) {
		for (int i2 = 0; i2 < s2; i2++) {
			for (int i1 = 0; i1 < s1; i1++)
			{
				cout << tn3d[i1][i2][i3] << "  ";
			}
			cout << endl;
		}
		cout << endl << endl;
	}
	cout << endl << endl;
}


// print a 4d tensor tn4d of size s1, s2, s3, s4
void print4d(vector<vector<vector<vector<float>>>>& tn4d, int s1, int s2, int s3, int s4) {
	cout << endl << endl;
	for (int i4 = 0; i4 < s4; i4++) {
		for (int i3 = 0; i3 < s3; i3++) {
			for (int i2 = 0; i2 < s2; i2++) {
				for (int i1 = 0; i1 < s1; i1++) {
					cout << tn4d[i1][i2][i3][i4] << "   ";
				}
				cout << endl;
			}
			cout << endl << endl;
		}
		cout << endl << endl;
	}
	cout << endl << endl;
}

// allocate memory for a 4d tensor tn4d of size s1, s2, s3, s4
void alloc4d(vector<vector<vector<vector< float >>>>& tn4d, int s1, int s2, int s3, int s4) {

	tn4d.resize(s1);

	for (int i1 = 0; i1 < s1; i1++) {
		tn4d[i1].resize(s2);
		for (int i2 = 0; i2 < s2; i2++) {
			tn4d[i1][i2].resize(s3);
			for (int i3 = 0; i3 < s3; i3++)
			{
				tn4d[i1][i2][i3].resize(s4);
			}
		}
	}
}

void ConvNet::Layer1(ifstream &file)
{
	//Stage 1
	//layer 1
	alloc3d(D1, 32, 32, 3);				//32*32*3 to store image 
	M1(D1, 32, 32, 3, file);		//reads input image 

}

void ConvNet::Layer2(ifstream& infile, ofstream &outfile)
{
	//Stage 2
	//Layer 2 Convolution, Stride 1
	alloc1d(D4, 16);
	alloc3d(D2, 32, 32, 16);			//32*32*16 space to store layer 2
	alloc4d(D3, 5, 5, 3, 16);

	M2(D3, 5, 5, 3, 16, infile);	//reading 4d 
	M3(D4, 16, infile);
	M4(D1, D2, D3, D4, outfile);
}

void ConvNet::Layer3(ofstream &outfile)
{
	//Layer 3 ReLu activation function
	alloc3d(D5, 32, 32, 16);

	M5(D2, D5);

	//outfile << "\n\n\n\n\n\n\n Output of D2 after ReLu: D5\n" << endl;
	//print3d(D5, 32, 32, 16);

}

void ConvNet::Layer4()
{
//Layer 4 Maxpooling filter size 2*2, stride = 2. 
//D6 16x16x16 array for storing output
//M6 Proccess D5 to compute and store D6. Each element is the max of 4 elements in a 2x2 block
//with stride 2.
	alloc3d(D6, 16, 16, 16);
	M6(D5, D6);
}

void ConvNet::Layer5(ifstream &infile, ofstream &outfile)
{
	//Layer 5 Second Conv+Relu+Max Pool
//D7 : 16x16x20 array of floating point numbers to store the output of L5
//D8 : 5x5x16x20  array for storing the 20 convolution filters with zero padding.
//D9 : 20x1 array for storing the bias vector
//M7 : for reading input file to initialize the data member D8 in L5.
//M8 : for reading input file to initialize the data member D9 in L5.
//M9 : for computing the data member D7 by convolving D6  with D8, stride = 1, 
//	   and adding D9, and storing the output in D7.
	alloc3d(D7, 16, 16, 20);
	alloc4d(D8, 5, 5, 16, 20);
	alloc1d(D9, 20);

	M7(D8, 5, 5, 16, 20, infile);
	M8(D9, 20, infile);
	M9(D6, D7, D8, D9, outfile);
}

void ConvNet::Layer6(ifstream& infile, ofstream& outfile)
{
	//Layer 6: ReLu 
	//Data member D10 : 16x16x20 array for storing the output of this layer L6.
	//Method member M10 : Computes and stores D10 using D7.Z = max(0, x).
	alloc3d(D10, 16, 16, 20);
	M10(D7, D10);
}

void ConvNet::Layer7()
{
	//Layer L7: Maxpooling: filter size 2x2, stride=2.
	//D11 : 8x8x20 array for storing the output of this layer L7.
	//M11 : Process D10 to computeand store D11.Each element is the maximum of 2x2 block, with stride 2.
	alloc3d(D11, 8, 8, 20);
	M11(D10, D11);
}

void ConvNet::Layer8(ifstream& infile, ofstream& outfile)
{
	//Layer 8 Third Conv+Relu+Max Pool
	//D12 : 8x8x20 array of floating point numbers to store the output of L8
	//D13 : 5x5x20x20  array for storing the 20 convolution filters with zero padding.
	//D14 : 20x1 array for storing the bias vector
	// 
	//M12 : for reading input file to initialize the data member D13 in L8.
	//M13 : for reading input file to initialize the data member D14 in L8.
	//M14 : for computing the data member D12 by convolving D11  with D13, stride = 1, 
	//		and adding D14, and storing the output in D12.

	alloc3d(D12, 8, 8, 20);
	alloc4d(D13, 5, 5, 20, 20);
	alloc1d(D14, 20);

	M12(D13, 5, 5, 20, 20, infile);
	M13(D14, 20, infile);
	M14(D11, D12, D13, D14, outfile);
}

void ConvNet::Layer9()
{
	//Layer 9: ReLu 
	//Data member D15 : 8x8x20 array for storing the output of this layer L9.
	//Method member M15 : Computes and stores D15 using D12.Z = max(0, x).
	alloc3d(D15, 8, 8, 20);
	M15(D12, D15);
}

void ConvNet::Layer10()
{
	//Layer L10: Maxpooling: filter size 2x2, stride=2.
	//D16: 4x4x20 array for storing the output of this layer L10.
	//M16 : Process D15 to compute and store D16. Each element is the maximum of 
	//		4 elements in a 2x2 block, with stride 2.

	alloc3d(D16, 4, 4, 20);
	M16(D15, D16);
}
void ConvNet::Layer11(ifstream& infile, ofstream& outfile)
{
	//Layer L11 : Fully Connected Layer
	//D17 : array of size 10 to store the output of this layer L11
	//D18 : 4x4x20x10 array for storing 10  full connection filters(dot product).
	//D19 : array of size 10 for storing a bias vector.
	//M17 : for reading input file to initialize the data member D18 in L11.
	//M18 : for reading input file to initialize the data member D19 in L11.
	//M19 : for computing the data member D17 by taking dot-product of D16 
	//		with the 10 different 4x4x20 filters stored in D18, 
	//		and adding D19, and storing the output in D17.
	Layer10(); 
	alloc1d(D17, 10); 
	alloc4d(D18, 4, 4, 20, 10); 
	alloc1d(D19, 10); 

	M17(D18, 4, 4, 20, 10, infile); 

	M18(D19, 10, infile); 
	//cout << "\n\n\n\n\n\nOutput of D19\n\n\n\n\n\n"; 
	//print1d(D19, 10); 
	M19(D17, D16, D18, D19, outfile); 
}

//Layer L12 : Softmax layer
//D20 : array of size 10 to store the output of this layer L12
//M20 : In order to avoid taking the exponent of large numbers that cause overflow, 
//normalize the contents of D17, by dividing each element by the square-root 
//of the sum of the squares of each element in D17.
//Use this result in computing probabilities in the next method M21.
//M21 : for each element x of normalized D17, compute the corresponding softmax 
//function of x that gives it’s probability : (exp(x) / (sum of exp(xi) for all i).
//M22 : Method that prints the output of this CNN stored in D20.
void ConvNet::Layer12(ofstream& out)
{
	alloc1d(D20, 10); 

	M20(D17); 
	M21(D17, out); 
	M22(D17, D20, out); 
}



int main()
{
	ifstream input_file, input_image; 
	input_file.open("CNN_weights.txt");
	input_image.open("Test_image.txt");

	ofstream OutputFile; 
	OutputFile.open("OutputFile.txt"); 

	if (OutputFile.fail())
	{
		cout << "\nError opening file!" << endl; 
		exit(0); 
	}
	ConvNet layers; 

	layers.Layer1(input_image);
	layers.Layer2(input_file, OutputFile); 
	layers.Layer3(OutputFile);
	layers.Layer4();
	layers.Layer5(input_file, OutputFile);
	layers.Layer6(input_file, OutputFile);
	layers.Layer7();
	layers.Layer8(input_file, OutputFile);
	layers.Layer9();
	layers.Layer10();
	layers.Layer11(input_file, OutputFile);
	layers.Layer12(OutputFile);

	cout << "\nOutputed to Text file done!" << endl; 

}

