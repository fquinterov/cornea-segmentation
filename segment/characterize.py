from skimage.measure import label, regionprops
from skimage.morphology import selem, binary_dilation
import math
import numpy as np
import matplotlib.pyplot as plt

class Characterize:
    
    def __init__(self, seg, FC, d=4):
        self.d = d
        self.seg = seg
        self.FC = FC
        self.seg_label = label(self.seg, connectivity=1)
        
        props = regionprops(self.seg_label)
        self.areas = [prop.area for prop in props] # areas in pix of each region
        
        kernel = selem.diamond(self.d)
        self.hex_shape = 0
        
        if len(self.areas)>0:
            
            # The list "sizes" will contain the number of regions in each range (>0, >100, >200, etc.)
            self.sizes = [0 for i in range(math.floor( max(self.areas) * FC * 10**6 / 100) + 1)] # *10^6 to convert mm2->um2
            
            for i, region in enumerate(props):
                                
                # Extract one region to be dilated
                img = self.seg_label == i+1
                img_dilation = binary_dilation(img, kernel)
                
                # The dilation merged the initial region and the surrounding regions
                neighbors = img_dilation == seg
                neighbors[seg==0] = 0
                neighbors = neighbors - img*1
                
                # This way we can count them and know the shape of the initial region using the label function
                neighbors_label = label(neighbors, connectivity=1)
                region_shape = np.amax(neighbors_label)
                
                if region_shape==6:
                    self.hex_shape += 1
                
                self.sizes[math.floor(self.areas[i] * FC * 10**6 / 100)] += 1
                
            self.N_reg = np.amax(self.seg_label)
            self.areas_mm2 = [area*FC for area in self.areas]
            self.areas_um2 = [area*FC*10**6 for area in self.areas] # *10^6 to convert mm2->um2
            
        else:
            
            ####### Try this #################################
            self.sizes = [0]
            self.N_reg = 0
            self.areas = [0]
            self.areas_mm2 = [0]
            self.areas_um2 = [0]
            
            
    def get_N(self): # --> cell, guttae
        
        # Get number of regions in the binary image
        return self.N_reg


    def get_avg(self, mode='um'): # --> cell, guttae, seg
        
        # Get mean area
        # mode can be 'pix', 'um' or 'mm'
        if mode=='pix':
            return np.mean(self.areas)
        elif mode=='mm':
            return np.mean(self.areas_mm2)
        else:
            return np.mean(self.areas_um2)


    def get_sizes(self, mode='um'): # --> cell, guttae
        
        # Get min, max and average areas
        if mode=='pix':
            return min(self.areas), max(self.areas), np.mean(self.areas)
        elif mode=='mm':
            return min(self.areas_mm2), max(self.areas_mm2), np.mean(self.areas_mm2)
        else:
            return min(self.areas_um2), max(self.areas_um2), np.mean(self.areas_um2)
            
            
    def get_std(self, mode='um'): # --> cell, guttae
        
        # Get standard deviation of region areas --> std_size
        if mode=='pix':
            return np.std(self.areas)
        elif mode=='mm':
            return np.std(self.areas_mm2)
        else:
            return np.std(self.areas_um2)
            
            
    def get_cv(self, mode='um'): # --> cell, guttae
        
        # Get coefficient of variation (polymegathism) of region areas --> cv_size
        if np.mean(self.areas)==0:
            return 0
        elif mode=='pix':
            return np.std(self.areas) / np.mean(self.areas) * 100
        elif mode=='mm':
            return np.std(self.areas_mm2) / np.mean(self.areas_mm2) * 100
        else:
            return np.std(self.areas_um2) / np.mean(self.areas_um2) * 100
            
    
    def get_hex(self): # --> seg
        
        # DON'T USE FOR CELL OR GUTTAE REGIONS, BUT FOR THEIR SUM
        # Get percentage of hexagonal regions (pleomorphism) --> hexagonality
        return self.hex_shape / self.N_reg * 100
        
        
    def get_areas(self, mode='um'): # --> cell, guttae
        
        # Get a list that contains the regions areas in the unit specified by the mode
        # mode can be 'pix', 'um' or 'mm'
        if mode=='pix':
            return self.areas
        elif mode=='mm':
            return self.areas_mm2
        else:
            return self.areas_um2
            
            
    def get_apercentages(self): # --> cell, guttae
        
        # Get percentage of region in each range (>0, >100, >200, etc. um2) --> porc
        return self.sizes / self.N_reg * 100
        
        
def density(cell_areas, guttae_areas, skel=None): # region density --> cell, guttae, seg 
    
    N_cell = len(cell_areas)
    N_guttae = len(guttae_areas)
    
    total_area = sum(cell_areas) + sum(guttae_areas)
    
    if not skel==None:
        total_area += sum(sum(skel))

    if N_guttae==1 and guttae_areas[0]==0:
        N_guttae = 0

    if N_cell==1 and cell_areas[0]==0:
        N_guttae = 0

    CD = N_cell / total_area # Cell Density, number of cells per unit area
    GD = N_guttae / total_area # Guttae Density, number of guttae per unit area
    
    return CD, GD
    

def class_hex(cell, guttae, d=4, skel=None):
    
    try:
        seg = 1 - skel
    except TypeError:
        seg = cell + guttae        
    
    cell_label = label(cell, connectivity=1)
    guttae_label = label(guttae, connectivity=1)
    
    cell_props = regionprops(cell_label)
    guttae_props = regionprops(guttae_label)
    
    kernel = selem.diamond(d)
    cell_hex = 0
    guttae_hex = 0
    
    # Calculate hexagonality of cells
    for i, region in enumerate(cell_props):
    
        # Extract one region to be dilated
        img = cell_label == i+1
        img_dilation = binary_dilation(img, kernel)

        # The dilation merged the initial region and the surrounding regions
        neighbors = img_dilation == seg
        neighbors[seg==0] = 0
        neighbors = neighbors - img*1

        # This way we can count them and know the shape of the initial region using the label function
        neighbors_label = label(neighbors, connectivity=1)
        region_shape = np.amax(neighbors_label)

        if region_shape==6:
            cell_hex += 1
            
     # Calculate hexagonality of guttae
    for i, region in enumerate(cell_props):
    
        # Extract one region to be dilated
        img = guttae_label == i+1
        img_dilation = binary_dilation(img, kernel)

        # The dilation merged the initial region and the surrounding regions
        neighbors = img_dilation == seg
        neighbors[seg==0] = 0
        neighbors = neighbors - img*1

        # This way we can count them and know the shape of the initial region using the label function
        neighbors_label = label(neighbors, connectivity=1)
        region_shape = np.amax(neighbors_label)

        if region_shape==6:
            guttae_hex += 1
            
    if len(cell_props)>0 and len(guttae_props)>0:
        return cell_hex * 100 / len(cell_props), guttae_hex * 100 / len(guttae_props)

    elif len(cell_props)>0 and len(guttae_props)==0:
        return cell_hex * 100 / len(cell_props), 0

    elif len(cell_props)==0 and len(guttae_props)>0:
        return 0, guttae_hex * 100 / len(guttae_props)

    else:
        return 0, 0

        

        

        

        
        

        
        

        