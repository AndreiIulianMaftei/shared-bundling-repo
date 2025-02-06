# When the plugin development is finished, you can copy the associated
# Python file to /home/markus/.Tulip-5.5/plugins/python
# and it will be automatically loaded at Tulip startup

from tulip import tlp
import tulipplugins


class NodeExport(tlp.ExportModule):
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
        
        for n in self.graph.nodes():
            s = str(n).replace('<node ', '').replace('>','')
            os << s << ' ' << x.getNodeValue(n) << ' ' << y.getNodeValue(n) << '\n'
        
        return True

# The line below does the magic to register the plugin into the plugin database
# and updates the GUI to make it accessible through the menus.
tulipplugins.registerPlugin('NodeExport', 'Node Export', '', '02/03/2021', '', '1.0')
