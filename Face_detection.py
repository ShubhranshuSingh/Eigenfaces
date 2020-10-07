import numpy as np
import cv2
import matplotlib.pyplot as plt

# This function calculates the weight vector of each image
def weights(I_mean, u):
    w = np.zeros(len(u))
    for k in range(len(u)):
        temp_u = u[k].reshape((1,size[0]*size[1]))
        temp_I = I_mean.reshape((size[0]*size[1],1))
        w[k] = temp_u@temp_I
    return w

size = (112,92) # Size of images
image_each_person = 10 # Images of each person
image_train = 8 # Images of each person used for training
number_of_persons = 40 # Total number of different people

# Saving Images
images = []
average = np.zeros(size)
for i in range(1,number_of_persons+1):
    for j in range(1,image_train+1):
        img = 'orl_faces\\s'+str(i)+'\\'+str(j)+'.pgm'
        img = cv2.imread(img,0)
        images.append(img)
        average = average+img

# Average Image       
average = average / len(images)

plt.imshow(np.uint8(average),cmap = 'gray')
plt.title("Average Face")
plt.show()

# Normalise Image
images = images - average

# Find L matrix (A.T A)
L = np.zeros((len(images),len(images)))

for m in range(len(images)):
    for n in range(len(images)):
        phi_m = images[m].reshape((1,size[0]*size[1]))
        phi_n = images[n].reshape((size[0]*size[1],1))
        temp = phi_m@phi_n
        L[m,n] = temp[0,0]

# Finding eigenvalues and eigenvectors
eigen_values, U= np.linalg.eig(L)

# Sorting eigenvectors based on eigenvalues (largest to smallest)
idx = sorted(range(len(eigen_values)), key=lambda k: eigen_values[k])[::-1]
U = U.T[idx].T

# Eigen Faces
eigenfaces = []

for l in range(len(images)):
    temp = np.zeros(size)
    for k in range(len(images)):
        phi_k = images[k]
        temp = temp + U[k,l]*phi_k
    temp = temp.reshape((1,size[0]*size[1]))
    temp = temp/np.linalg.norm(temp)    # Eigen faces are normalised
    temp = temp.reshape(size)
    eigenfaces.append(temp)

# 100 best eigenfaces are selected
eigenfaces = eigenfaces[0:100]

# Show 5 best eigenfaces
for i in range(5):
    plt.imshow(eigenfaces[i],cmap='gray')
    plt.title("Eigen face "+str(i+1))
    plt.show()

# Calculating weights for each image
all_weights = []

for i in range(0,len(images),image_train):
    for j in range(image_train):
        all_weights.append(weights(images[i+j],eigenfaces))

# Test accuracy 
test_match = 0
num_test = 0
s = "orl_faces\\"
for i in range(1,number_of_persons+1):
    folder = s+'s'+str(i)+"\\"
    for j in range(image_train + 1,11): # Taking all images except the one which have been used for training
        img = folder + str(j) + ".pgm"
        gray = cv2.imread(img,0)
        test = gray

        # Normalise and find weight vector
        test = test - average
        w_test = weights(test,eigenfaces)

        minn = np.Inf
        match_class = 0

        # Finding the closest matching class
        for k in range(len(all_weights)):
            temp = np.linalg.norm(all_weights[k]-w_test)
            if(temp<minn):
                minn = temp
                match_class = k

        # TO check if the matched class is actually correct
        if((match_class//image_train)+1 == i):
            test_match += 1
        else:
            print("Image "+img+" is incorrectly recognised")

        # Number of test images
        num_test += 1

print("Test Accuracy:",(test_match/num_test)*100)
print()
print()

# User Input
while(True):
    # Image location
    a = input("Please give the test image:")
    gray = cv2.imread(a,0)
    test = gray

    # Normalise and find weight vector
    test = test - average
    w_test = weights(test,eigenfaces)
    
    minn = np.Inf
    match_class = 0
    # Finding the closest matching class
    for i in range(len(all_weights)):
        temp = np.linalg.norm(all_weights[i]-w_test)
        if(temp<minn):
            minn = temp
            match_class = i
    
    print("This face is from folder: ",(match_class//image_train)+1)

    q = input("Dow you want to exit? (y or n): ")
    if(q == 'y'):
        break
