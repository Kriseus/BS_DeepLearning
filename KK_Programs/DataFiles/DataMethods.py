import sys
import numpy as np
sys.path.append('../../KG_Programs/mods')
from fitting import BandStructure


class EnchancedBS(BandStructure):
    def __init__(self, mat,path,Ham):
        super().__init__(mat = mat, path = path, Ham = Ham)
    def SaveBSToList(self):
        fulldata = []
        for i in range(0,40):
            fulldata.append([])
        for n in range(len(self.path)):
            p = self.path[n]
            E = self.energies[n]
            k = (np.sign(p.kx)*np.sqrt(p.kx**2+p.ky**2+p.kz**2)) 
            fulldata[0].append(k)
            fulldata[1].append(p.kx)
            fulldata[2].append(p.ky)
            fulldata[3].append(p.kz)
            fulldata[4].append(p.exx)
            fulldata[5].append(p.eyy)
            fulldata[6].append(p.ezz)
            fulldata[7].append(p.exy)
            fulldata[8].append(p.eyz)
            fulldata[9].append(p.ezx)
            for j in range(0,len(E)):
                fulldata[j+10].append(E[j])
        return fulldata
    
    def saveBS(self,dir,suffix):

        filename = "bulk_%s_%s.dat"%(self.mat.name,suffix)

        f = open(dir + filename,"a")


        for n in range(len(self.path)):
            p = self.path[n]
            E = self.energies[n]
            k = (np.sign(p.kx)*np.sqrt(p.kx**2+p.ky**2+p.kz**2)) 
            line = "%e %e %e %e %e %e %e %e %e %e "%(k,p.kx,p.ky,p.kz,p.exx,p.eyy,p.ezz,p.exy,p.eyz,p.ezx) + " ".join(map(str, E))
            line += "\n"
            f.write(line)
        f.write("#\n")
        f.close()
        return 0



def EnterChange(Input):
    lottery = np.random.randint(8,19)
    for i in range(0,8):
        Input[i]=Input[i]*np.random.uniform(-1.2,1.2)
    for j in range(20,24):
        Input[j]=Input[j]*np.random.uniform(-1.2,1.2)
    Input[lottery]=Input[lottery]*np.random.uniform(0.95,1.05)        
    OptInput = Input[9:21]
    return Input.copy(), OptInput.copy()

