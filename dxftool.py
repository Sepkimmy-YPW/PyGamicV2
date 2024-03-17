import ezdxf
import numpy as np
import math
from utils import *

class OrigamiToDxfConverter:
    def __init__(self, filename=None) -> None:
        self.filename = filename
    
    def setFileName(self, filename):
        self.filename = filename

    def ExportAsDxf(self, lines: list):
        doc = ezdxf.new(setup=True, units=4)
        msp = doc.modelspace()
        doc.layers.new(name='Mountain', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 25,
                'color': 1
            }
        )
        doc.layers.new(name='Valley', 
            dxfattribs={
                'linetype': 'DASHEDX2', 
                'lineweight': 25,
                'color': 5
            }
        )
        doc.layers.new(name='Cutting', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 25,
                'color': 3
            }
        )
        doc.layers.new(name='Border', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 30,
                'color': 7
            }
        )
        for i in range(len(lines)):
            line = lines[i]
            type_crease = line.getType()
            if distance(line[START], line[END]) < 1e-5:
                continue
            not_duplicate = True
            for j in range(i):
                other_line = lines[j]
                if (distance(line[START], other_line[START]) < 1e-5 and distance(line[END], other_line[END]) < 1e-5) or (distance(line[START], other_line[END]) < 1e-5 and distance(line[END], other_line[START]) < 1e-5):
                    not_duplicate = False
                    break
            if not not_duplicate:
                continue
            if type_crease == MOUNTAIN:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Mountain'})
            elif type_crease == VALLEY:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Valley'})
            elif type_crease == CUTTING:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Cutting'})
            else:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Border'})
        doc.saveas(self.filename)

            
