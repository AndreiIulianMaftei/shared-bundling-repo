# When the plugin development is finished, you can copy the associated
# Python file to /home/markus/.Tulip-5.5/plugins/python
# and it will be automatically loaded at Tulip startup

from tulip import tlp
import tulipplugins


class EdgeExport(tlp.ExportModule):
    def __init__(self, context):
        tlp.ExportModule.__init__(self, context)
        # You can add parameters to the plugin here through the
        # following syntax:
        # self.add<Type>Parameter('<paramName>', '<paramDoc>',
        #                         '<paramDefaultValue>')
        # (see the documentation of class tlp.WithParameter to see what
        #  parameter types are supported).
    
    def exportGraph(self, os):
        # This method is called to export a graph.
        # The graph to export is accessible through the 'graph' class attribute
        # (see documentation of class tlp.Graph).
        #
        # The parameters provided by the user are stored in dictionary
        # that can be accessed through the 'dataSet' class attribute.
        #
        # The os parameter is an output file stream (initialized by the
        # Tulip GUI or by the tlp.exportGraph function).
        # To write data to the file, you have to use the following syntax:
        #
        # write the number of nodes and edges to the file
        # os << self.graph.numberOfNodes() << '\n'
        # os << self.graph.numberOfEdges() << '\n'
        #
        # The method must return a Boolean indicating if the algorithm
        # has been successfully applied on the input graph.
        
        y = self.graph.getDoubleProperty("y")
        x = self.graph.getDoubleProperty("x")
        points = self.graph.getLayoutProperty('viewLayout')  
        
        for e in self.graph.edges():
            source = self.graph.source(e)
            target = self.graph.target(e)
            sourceS = str(source).replace('<node ', '').replace('>','')
            targetS = str(target).replace('<node ', '').replace('>','')
            os << sourceS << ' ' << targetS
            
            controlPoints = points.getEdgeValue(e)
            start = tlp.Vec3f()
            start[0] = x.getNodeValue(source)
            start[1] = y.getNodeValue(source)      
            
            end = tlp.Vec3f()
            end[0] = x.getNodeValue(target)
            end[1] = y.getNodeValue(target)      
            
            controlPoints = [start] + controlPoints
            controlPoints.append(end)
            
            #pointList = tlp.computeOpenUniformBsplinePoints(controlPoints, curveDegree=3, nbCurvePoints=50)         
            pointList = tlp.computeBezierPoints(controlPoints, nbCurvePoints=50)         
            
            for point in pointList:
                #x.append(point.x)
                #y.append(point.y)
                
                os << ' ' << point[0] << ' ' << point[1]
            
            os  << '\n'
            
        return True

# The line below does the magic to register the plugin into the plugin database
# and updates the GUI to make it accessible through the menus.
tulipplugins.registerPlugin('EdgeExport', 'Edge Export', '', '02/03/2021', '', '1.0')
