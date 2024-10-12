CUBu:

You need to have the CUDa libraries and compile the source. It took me some time to get the right includes but it worked for me. I didn't write down the process.

Once you have CUBu you can open a file with the following cmd:  ./cubu -f gowalla_SF_12_overlap_free.trl -i 1000 
-f specifies the file
-i specifies the image size

fd:

Force-directed bundling is 


FD:

force-directed bundling is a web application. In the folder example are two html pages that you open and they will compute the output and visualize it. I added some code to display the edge information which I copied by hand into a txt file. The input file is hardcoded in the example .html page.

WR:

winding roads is implemented in the tulip software and should run out of the box. This is the one approach I am worried the most because I don't know how to automate the process of getting the output. (https://tulip.labri.fr/site/)

At the moment, I wrote two scripts that can be loaded into tulip and generate the output. The scripts are in the folder.

EPB:

that is the latest iteration of my code. It can produce EPB, Confluent and Straight-line drawings. It also contains the metrics and a loader for the output format of the other algorithms. I can walk you through an example but if you want to figure it out look at the code in main.py.
