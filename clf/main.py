# -*- coding: utf-8 -*-

import AllElectronics
import ID3

# config: using which one 
USEID3=1 
USEAllElectronics=1

if __name__=='__main__':
    if (USEID3):
        mytree = ID3.run()
        print mytree
        ID3.test(mytree)
   
    if (USEAllElectronics):
	    AllElectronics.run()