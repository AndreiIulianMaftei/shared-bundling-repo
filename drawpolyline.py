import matplotlib.pyplot as plt

# The list of coordinates you provided:
points = [
(1178.8620177974278, 360.61000098563727), (1130.0253377732483, 349.13163315899556), (1090.0824726054564, 339.67242563008546), (1057.7387356395964, 331.8656221560693), (1031.8794272178116, 325.4189595339084), (1011.5450601319342, 320.10153483471544), (995.9098510495462, 315.7324313695341), (984.2631121196151, 312.17094170127405), (975.9932055292795, 309.3082344518865), (970.5737511071303, 307.06031988149124), (967.551803150762, 305.3621772370474), (966.5377374976113, 304.1629146823146), (967.1966134579905, 303.42184022825387), (969.2407975879079, 303.10532948368717), (972.4236573966399, 303.1843832399675), (976.5341529601451, 303.63277489060187), (981.3921730462488, 304.42569446722143), (986.844479751118, 305.5388026470106), (992.7611417988385, 306.94761445367766), (999.0323515669533, 308.62713853429284), (1005.5655345705802, 310.5517038478092), (1012.2826725662334, 312.69491134785216), (1019.1177726236835, 315.02965378237076), (1026.0144244601652, 317.5281520660386), (1032.9233970359126, 320.1619618078262), (1039.800232873425, 322.90190849597553), (1046.602804785009, 325.7179145556703), (1053.2888046740168, 328.5786860010255), (1059.8131378148103, 331.45123070260473), (1066.125198514808, 334.300184384525), (1072.166004319039, 337.0869243513182), (1077.8651659334294, 339.76845462409216), (1083.1376688175528, 342.2960496381646), (1087.8804399308547, 344.61364692023835), (1091.968669408322, 346.65598222234536), (1095.2518519922892, 348.3464634421967), (1097.5495068565294, 349.5947823052619), (1098.6465270269275, 350.2942652228337), (1098.2881009299515, 350.31896697253853), (1096.1741386857689, 349.5205128732143), (1091.9531246072117, 347.7246969447988), (1085.21530496889, 344.72784515585823), (1075.4851064725738, 340.2929542666294), (1062.2126659555147, 334.14561797395635), (1044.764335767663, 325.96975305627143), (1022.4120118817399, 315.40313900179837), (994.3211131968777, 302.0327851814466), (959.5370206519951, 285.39013999941795), (916.9697636792915, 264.94615661935825), (865.3767192011621, 240.10622982196423)     # ... add the rest of the points here ...
]

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