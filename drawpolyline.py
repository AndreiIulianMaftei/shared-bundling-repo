import matplotlib.pyplot as plt
"""
drawpolyline.py
This script generates a polyline plot from a given list of 2D points using Matplotlib. 
The polyline is drawn by connecting the points in the order they are provided. 
The resulting plot is saved as a PNG file and can optionally be displayed interactively.
Modules:
    - matplotlib.pyplot: Used for plotting the polyline and saving the figure.
Data:
    - points: A list of tuples representing the (x, y) coordinates of the polyline.
Functions:
    - None
Workflow:
    1. Extracts the x and y coordinates from the `points` list.
    2. Creates a Matplotlib figure and plots the polyline with markers at each point.
    3. Configures the plot with labels, a title, and a grid.
    4. Saves the plot to a file named 'polyline.png' with a resolution of 300 DPI.
    5. Optionally, the plot can be displayed interactively by uncommenting the `plt.show()` line.
Usage:
    Run the script to generate and save the polyline plot. Ensure that the `points` list 
    contains the desired coordinates before execution.
Output:
    - A PNG file named 'polyline.png' containing the polyline plot.
Note:
    - The script assumes that the `points` list is pre-defined and contains valid (x, y) coordinates.
"""

# The list of coordinates you provided:
points = [(701.5516798324147, 604.1508267598136), (713.3273809510902, 610.0747459558773), (722.2768129981067, 611.6115218883436), (730.2219933954967, 610.2678470415802), (737.6663080077844, 607.6616124044458), (744.980208509989, 604.9337839352047), (752.5996686420987, 602.6935908341009), (760.9279011116172, 601.1544523271765), (770.2131337416017, 600.294160214526), (780.4881275645367, 599.9791502867886), (791.5897074378545, 600.0424672822638), (803.2312143273139, 600.3231411278778), (815.1001583845997, 600.6816224256428), (826.9376014491118, 601.0023025320794), (838.5898186761142, 601.189733439083), (850.0227889934116, 601.1629575952254), (861.3014044461177, 600.8533028362597), (872.549778945761, 600.2026498305182), (883.9090369399678, 599.1665821781903), (895.5026623204111, 597.718481540382), (907.4131876881582, 595.8559853426352), (919.6733740756937, 593.6066569536271), (932.2712511057636, 591.0295607348082), (945.1595672892072, 588.2141595056562), (958.2677604022115, 585.2724392954815), (971.5126678238538, 582.3242613817881), (984.809236873645, 579.480194249212), (998.0730448909515, 576.8227704988202), (1011.2247093739835, 574.3901063318689), (1024.1942981187833, 572.1664512519714), (1036.9175491000926, 570.0817155502059), (1049.3371305110632, 568.0210781079124), (1061.402640763257, 565.8418394278258), (1073.0630482483807, 563.397834903476), (1084.2679513779963, 560.5644781084429), (1094.9669485636657, 557.26333157072), (1105.1140483559395, 553.4811648733421), (1114.6752296826219, 549.2800349720737), (1123.6435623253026, 544.7952386308779), (1132.048657217189, 540.2171993510717), (1139.9686368203602, 535.7552412296369), (1147.532655205188, 531.5863998459615), (1154.920268030483, 527.7971454252096), (1162.3500921860948, 524.3373913888709), (1170.0621652369593, 521.0053738782296), (1178.2788841920587, 517.4791527471384), (1187.1350733004651, 513.3807160823123), (1196.5639504574979, 508.294565789066), (1206.09174073184, 501.5837517935145), (1214.309089706795, 491.7001578057931)]
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]

plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, marker='o', markersize=2, linewidth=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polyline from Given Points')
plt.grid(True)

# Save the figure to a file (e.g., a PNG file)
plt.savefig('polyline.png', dpi=300)

# If you want to display the plot interactively (optional):
# plt.show()
