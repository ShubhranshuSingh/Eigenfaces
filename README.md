Python 3.6.5 (System: Intel Core i5-7200U @ 2.5GHz, 64-bit, 8 GB RAM) 

## Dataset
	http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z
	Unzip to get the orl_faces folder into the same directory as the code

## Code
	- The file Face_detection.py is the main program that trains and validates the images.

	Some of the constants in the starting of the file can be changed.
	image_train can be any value less than 10 and greater than 0.
	number_of_persons can be any values less than 41 and greater than 0.

	- Inititally, average face is plotted
	- Then top 5 eigenfaces are shown 
	- After that the mismatched images name are shown
	- Then the percentage accuracy is displayed
	- Then user is asked to enter the image location to test the image
	- The input should be the absolute path to the image or the path should be relative to the folder containing the code
	Example of inputs absolute - G:\\Sem-5\\Prob and Rand\\Ass 1\\orl_faces\\s1\\9.pgm 
	or relative - orl_faces\\s1\\9.pgm(Here code is already in Ass 1 folder) (Remember to put \\ between directories)
	The path of the images of dataset is of the form orl_faces\\sX\\Y.pgm X can be 1,2,3 ... 40 Y can be 1,2,3 ... 10
	
	The output will tell the folder of the test image. for example if orl_faces\\s1\\9.pgm is given as input, the desired output would be:
	"This face is from folder:  1"

	The user will be asked to input images until y is entered when asked to exit.