# Convolution-Of-Images<br>
***The test weights, and test images are attched*** <br>
Defining a ConvNet class with the following class members:<br>
(1) one data member for storing the output of each layer where the first layer stores the result of reading the input image data (32x32x3 array of floating point numbers) and the last layer stores the output of the ConvNet (which is the probability of the 10 classes), <br>
(2) one method/function member to compute the output of each of the layers from 2 to the last layer. <br><br>
***Stage 1 Input***<br>
1)  Layer L1 : input layer <br>
a)	Data member D1:  float 3D matrix array of size 32x32x3 to store input image.<br>
b)	Method member M1: reads input file and initializes the input layer in 1a.<br>

***Stage 2: First Conv+ReLU+Max pool***<br>
2)  Layer L2 : Convolution, Stride 1 <br>
a. Data member D2: 32x32x16 array of floating point numbers to store the output of L2 <br>
b. Data member D3:  5x5x3x16 array for storing the 16 convolution filters with zero padding.<br>
c. Data member D4: 16x1 array for storing the bias vector<br>
d. Method member M2: for reading input file to initialize the data member D3 in L2.<br>
e. Method member M3: for reading input file to initialize the data member D4 in L2.<br>
f. Method M4 : for computing the data member D2 by convolving D1 with D3 and adding D4, and storing the output in D2. Set Stride = 1.<br><br>

3)	Layer L3: ReLu activation function <br>
a.	Data member D5: 32x32x16 array for storing the output of this layer L3.<br>
b.	Method member M5: Computes D5 using D2. Z= max(0,x).<br>
4)	Layer L4: Maxpooling: filter size 2x2, stride=2. <br>
a.	Data member D6: 16x16x16 array for storing the output of this layer L4.<br>
b.	Method member M6: Process D5 to compute and store D6. Each element is the maximum of 4 elements in a 2x2 block, with stride 2.<br>
The way this CNN works is such that it considers max-pooling of 2x2 regions in 2D cross-sections of 3D arrays, for each cross-section separately. Only 2D max pooling is used. For example, the first max pooling in layer L4 does the following:<br>
Input: D5: 32x32x16 say D5[m][n][k]<br>
Output: D6: 16x16x16, say D6[i][j][k]<br>
Stride: 2<br>

***Stage 3: Second Conv+ReLU+Max pool***<br>

5)  Layer L5 : Convolution, stride 1<br>
a. Data member D7: 16x16x20 array of floating point numbers to store the output of L5 <br>
b. Data member D8:  5x5x16x20  array for storing the 20 convolution filters with zero padding.<br>
c. Data member D9: 20x1 array for storing the bias vector<br>
d. Method member M7: for reading input file to initialize the data member D8 in L5.<br>
e. Method member M8: for reading input file to initialize the data member D9 in L5.<br>
f. Method M9: for computing the data member D7 by convolving D6  with D8, stride=1, and adding D9, and storing the output in D7.<br>
6)	Layer L6: ReLu activation function<br>
a.	Data member D10: 16x16x20 array for storing the output of this layer L6.<br>
b.	Method member M10: Computes and stores D10 using D7. Z= max(0,x).<br>
7)	Layer L7: Maxpooling: filter size 2x2, stride=2.<br>
a.	Data member D11: 8x8x20 array for storing the output of this layer L7.<br>
b.	Method member M11: Process D10 to compute and store D11. Each element is the maximum of 2x2 block, with stride 2.<br>

***Stage 4: Third Conv+ReLU+Max pool***

8)  Layer L8 : Convolution, stride 1<br>
a. Data member D12: 8x8x20 array of floating point numbers to store the output of L8<br>
b. Data member D13:  5x5x20x20  array for storing the 20 convolution filters with zero padding.<br>
c. Data member D14: 20x1 array for storing the bias vector<br>
d. Method member M12: for reading input file to initialize the data member D13 in L8.<br>
e. Method member M13: for reading input file to initialize the data member D14 in L8.<br>
f. Method M14: for computing the data member D12 by convolving D11  with D13, stride=1, and adding D14, and storing the output in D12.<br>
9)	Layer L9: ReLu activation function<br>
a.	Data member D15: 8x8x20 array for storing the output of this layer L9.<br>
b.	Method member M15: Computes and stores D15 using D12. Z= max(0,x).<br>
10)	Layer L10: Maxpooling: filter size 2x2, stride=2.<br>
a.	Data member D16: 4x4x20 array for storing the output of this layer L10.<br>
b.	Method member M16: Process D15 to compute and store D16. Each element is the maximum of 4 elements in a 2x2 block, with stride 2.<br>

***Stage 5: Last layer: Fully connected + Softmax***<br>
11)	Layer L11 : Fully Connected Layer<br>
a.	Data member D17:  array of size 10 to store the output of this layer L11<br>
b.	Data member D18: 4x4x20x10 array for storing 10  full connection filters (dot product).<br>
c.	Data member D19: array of size 10 for storing a bias vector.<br>
d. Method member M17: for reading input file to initialize the data member D18 in L11.<br>
e. Method member M18: for reading input file to initialize the data member D19 in L11.<br>
f. Method M19: for computing the data member D17 by taking dot-product of D16  with the 10 different 4x4x20 filters stored in D18, and adding D19, and storing the output in D17.<br>
12) Layer L12: Softmax layer<br>
a.	Data member D20: array of size 10 to store the output of this layer L12<br>
b.	Method member M20: In order to avoid taking the exponent of large numbers that cause overflow, normalize the contents of D17, by dividing each element by the square-root of the sum of the squares of each element in D17. Using this result in computing probabilities in the next method M21. <br>
c.	Method M21: for each element x of normalized D17, compute the corresponding softmax function of x that gives it’s probability: ( exp(x)/(sum of exp(xi) for all i).
d.	Method M22: Method that prints the output of this CNN stored in D20.<br><br>
Main function creates a ConvNet object defined above, and calls methods to initialize the input, filters, and bias vectors, and then calls the methods to compute the output of layers in sequence, and prints the output to a file. <br>
