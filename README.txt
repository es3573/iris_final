README
==========================
1. This project compares python implementation of algorithms used in the Canny Edge Method
for iris detection with those of pyCUDA implementations. There is only one file: iris_canny.py
All the images used are found on /home/es3573/ as I have uploaded them onto the server.

2. To compile and run the program:
sbatch --gres=gpu:1 --time=8 --wrap="python iris_canny.py"

3. The expected output should be 15 figures. The first 9 feature step by step images from
each stage of the Canny Edge Method, with the final edge map shown at the very bottom. The
figures are numbered in increasing order (ie. Figure 1 is the smallest image, Figure 9 is the largest).

The next 6 figures will be time comparison subplots between python/pyCUDA. 
The figures are arranged according to each step of the Canny Edge Method (ie. Figure 6 is the first step,
and Figure 15 is the last step)

Finally, the .out file produced will show the results of each step for each image size with
a final table at the end to show the average speedups. 