# -*- coding: utf-8 -*-
#
# Sylvain Lobry 2015
# JM Nicolas 2015#
# 
# Septembre 2017 : on introduit mat2imz
#   sauvegarde en 8 bits ou en float. Pas d'autre choix
#   possibilitÃ© de gÃ©nÃ©rer un fichier .hdr en mettant une option
#
# Janvier : introduction de dat2mat pour le projet biomass
"""
read and display radar images
"""


import numpy as np
import os.path
import struct
import matplotlib.pyplot as plt

from math import sin,cos

#General IO functions

############################################################################
# Septembre 2017 : une liste de paramÃ¨tres sont requis
#
def imz2matbase(namima, listeparam):
#
# on rajoute un test sur l'existence mÃªme du fichier
#

    print('imz2mat, appel imz2matbase : version Septembre 2017')

    try :
        ftest=open(namima);
    except IOError:
        legx=namima+': is a non openable folder'
        print(legx)
        print(u'Echec Ã  l\'appel de imz2matbase')
        return 0,0,0,0
    ftest.close()
    
    nparam=np.size(listeparam)
    if(nparam!=7):
        print(u'Echec appel imz2matbase : nombre erronÃ© de paramÃ¨tres (%d'%nparam+' au lieu de 7')
        return 0,0,0,0
#    for iut in range(nparam) :
#        print(listeparam[iut])

##############################################################################
# Septembre 2017
#  



    nblok=    listeparam[0]*listeparam[1]*  listeparam[5]  *  (1+listeparam[6])
    ncan=0
    return( _readImage( namima, listeparam[0], listeparam[1], listeparam[2], 1, listeparam[3], listeparam[4], listeparam[5], listeparam[6], nblok, ncan))
        

def imz2mat(imgName,ncan=0):
    """
    lecture d'images plutot radar
    Formats Telecom et .dat
    argument 1 : nom du fichier image
    argument 2 (facultatif) : si multicanal, renvoie uniquement le canal indiquÃ©
    """

#
# on rajoute un test dur l'existence mÃªme du fichier
#

    print(u'imz2mat : version janvier 2018.  Folder to open : %s'%imgName)

    try :
        ftest=open(imgName);
    except IOError:
        legx=imgName+': is a non openable folder'
        print(legx)
        print(u'failure to call imz2mat')
        return 0,0,0,0
     
    ftest.close()

##############################################################################
# Septembre 2017
#  
    if imgName.endswith('.dim') : 
        ncolZ, nligZ, nplantotZ, nzzZ = dimimabase(imgName )
        offsetZ, nbBytesZ, typeZ, komplexZ, radarZ, endianZ, namima = dimimadim(imgName)
        print(dimimabase(imgName))
        print(dimimadim(imgName))
        nblok=ncolZ*nligZ*nbBytesZ*(1+komplexZ)
        return( _readImage( namima, ncolZ, nligZ, nplantotZ, nzzZ, offsetZ, endianZ+typeZ, nbBytesZ, komplexZ, nblok, ncan))
 
##############################################################################
# Janvier 2018 : donnÃ©es .dat de l'ONERA (projet Biomass)
#  
    if imgName.endswith('.dat') : 
        return( dat2mat(imgName ))  # voir Ã  la fin de ce fichier, avant les outils de visu
        
############################################################################## 
##############################################################################
# Septembre 2017
#  
        # faire la meme chose avec les .hdr !!!
###############################################################################
        
        
    ncolZ, nligZ, nplantotZ, nzzZ = dimimabase(imgName )

    if(nplantotZ==1):
        print("Dans ximaread : image monocanal")
        
    if(nplantotZ>0):
        print("Dans ximaread : lecture du canal "+"%d"%ncan+'/'+'%d'%nplantotZ)
        
        
    """ Reads a file in a xima format. """
    if imgName.endswith('.ima'):
        print("image en .ima")
        return imaread(imgName,ncan)
    elif imgName.endswith('.IMA'):
        print("image en .IMA")
        return imaread(imgName,ncan)
    elif imgName.endswith('.imw'):
        print("image en .imw")
        return imwread(imgName,ncan)
    elif imgName.endswith('.IMW'):
        print("image en .IMW")
        return imwread(imgName,ncan) 
    elif imgName.endswith('.iml'):
        print("image en .iml")
        return imlread(imgName,ncan)
    elif imgName.endswith('.IML'):
        print("image en .IML")
        return imlread(imgName,ncan) 
    elif imgName.endswith('.rvb'):
        print("image en .rvb")
        return imaread(imgName, 1)  ########### TODO
    elif imgName.endswith('.cxs'):
        print("image en .cxs")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.cxb'):
        print("image en .cxb")
        return cxbread(imgName,ncan)
    elif imgName.endswith('.cxbtivo'):
        print("image en .cxbtivo")
        return cxbread(imgName,ncan)
    elif imgName.endswith('.cxbadts'):
        print("image en .cxbadts")
        return cxbread(imgName,ncan)
    elif imgName.endswith('.CXS'):
        print("image en .CXS")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.cxstivo'):
        print("image en .cxstivo")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.CXSTIVO'):
        print("image en .CXSTIVO")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.cxsadts'):
        print("image en .cxsadts")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.CXSADTS'):
        print("image en .CXSADTS")
        return cxsread(imgName,ncan)
    elif imgName.endswith('.imf'):
        print("image en .imf")
        return imfread(imgName,ncan)
    elif imgName.endswith('.IMF'):
        print("image en .IMF")
        return imfread(imgName,ncan)
    elif imgName.endswith('.imd'):
        print("image en .imd")
        return imdread(imgName,ncan)
    elif imgName.endswith('.IMD'):
        print("image en .IMD")
        return imdread(imgName,ncan)
    elif imgName.endswith('.cxf'):
        print("image en .cxf")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.CXF'):
        print("image en .CXF")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.cxftivo'):
        print("image en .cxftivo")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.CXFTIVO'):
        print("image en .CXFTIVO")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.cxfadts'):
        print("image en .cxfadts")
        return cxfread(imgName,ncan)
    elif imgName.endswith('.CXFADTS'):
        print("image en .CXFADTS")
    else:
#        raise Exception("Format not currently supported.")
        print("Format non pris en compte actuellement");
        return 0,0,0,0,0
#IO operations for ima format. both IMA and ima
def imaread(imgName,ncan):
    """ Reads a *ima file. ImgName can be with or without extension. """
    if imgName.endswith('.ima'):
        print("image en .ima")
        extension='.ima'
    if imgName.endswith('.IMA'):
        print("image en .IMA")
        extension='.IMA'
    if imgName.endswith('.rvb'):
        print("image en .rvb")
        extension='.rvb'
    imgName = os.path.splitext(imgName)[0]
    return _imaread(imgName, extension,ncan)
#IO operations for imw format.
def imwread(imgName,ncan):
    """ Reads a *.imw file. ImgName can be with or without extension. """
    if imgName.endswith('.imw'):
        print("image en .imw")
        extension='.imw'
    if imgName.endswith('.IMW'):
        print("image en .IMW")
        extension='.IMW'
    imgName = os.path.splitext(imgName)[0]
    return _imwread(imgName, extension,ncan)
#IO operations for iml format.
def imlread(imgName,ncan):
    """ Reads a *.iml file. ImgName can be with or without extension. """
    if imgName.endswith('.iml'):
        print("image en .iml")
        extension='.iml'
    if imgName.endswith('.IML'):
        print("image en .IML")
        extension='.IML'
    imgName = os.path.splitext(imgName)[0]
    return _imlread(imgName, extension,ncan)
#IO operations for cxb format.
def cxbread(imgName,ncan):
    """ Reads a *.imw file. ImgName can be with or without extension. """
    if imgName.endswith('.cxb'):
        print("image en .cxb")
        extension='.cxb'
    if imgName.endswith('.cxbtivo'):
        print("image en .cxbtivo")
        extension='.cxbtivo'
    if imgName.endswith('.cxbadts'):
        print("image en .cxbadts")
        extension='.cxbadts'
    imgName = os.path.splitext(imgName)[0]
    return _cxbread(imgName, extension,ncan)
#IO operations for cxs format.
def cxsread(imgName,ncan):
    """ Reads a *.cxs file. ImgName can be with or without extension. """
    if imgName.endswith('.cxs'):
        print("image en .cxs")
        extension='.cxs'
    if imgName.endswith('.CXS'):
        print("image en .CXS")
        extension='.CXS'
    if imgName.endswith('.cxstivo'):
        print("image en .cxstivo")
        extension='.cxstivo'
    if imgName.endswith('.CXSTIVO'):
        print("image en .CXSTIVO")
        extension='.CXSTIVO'
    if imgName.endswith('.cxsadts'):
        print("image en .cxsadts")
        extension='.cxsadts'
    if imgName.endswith('.CXSADTS'):
        print("image en .CXSADTS")
        extension='.CXSADTS'
    imgName = os.path.splitext(imgName)[0]
    return _cxsread(imgName, extension,ncan)
#IO operations for imf format.
def imfread(imgName,ncan):
    """ Reads a *.imf file. ImgName can be with or without extension. """
    if imgName.endswith('.imf'):
        print("image en .imf")
        extension='.imf'
    if imgName.endswith('.IMF'):
        print("image en .IMF")
        extension='.IMF'
    imgName = os.path.splitext(imgName)[0]
    return _imfread(imgName, extension,ncan)
#IO operations for imd format.
def imdread(imgName,ncan):
    """ Reads a *.imf file. ImgName can be with or without extension. """
    if imgName.endswith('.imd'):
        print("image en .imd")
        extension='.imd'
    if imgName.endswith('.IMD'):
        print("image en .IMD")
        extension='.IMD'
    imgName = os.path.splitext(imgName)[0]
    return _imdread(imgName, extension,ncan)
#IO operations for cxf format.
def cxfread(imgName,ncan):
    """ Reads a *.cxf file. ImgName can be with or without extension. """
    if imgName.endswith('.cxf'):
        print("image en .cxf")
        extension='.cxf'
    if imgName.endswith('.CXF'):
        print("image en .CXF")
        extension='.CXF'
    if imgName.endswith('.cxftivo'):
        print("image en .cxftivo")
        extension='.cxftivo'
    if imgName.endswith('.CXFTIVO'):
        print("image en .CXFTIVO")
        extension='.CXFTIVO'
    if imgName.endswith('.cxfadts'):
        print("image en .cxfadts")
        extension='.cxfadts'
    if imgName.endswith('.CXFADTS'):
        print("image en .CXFADTS")
        extension='.CXFADTS'
    imgName = os.path.splitext(imgName)[0]
    return _cxfread(imgName, extension,ncan)
#Internal functions.
def _imaread(imgName, extension, ncan):
    """ Reads a *.ima file. imgName should come with no extension. """
        #

    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en .ima ',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=1
    offset=0
    type='B'
    komplex=0
    nblok=w*h
    indien = '>' #en minuscule
    
    if extension == '.ima':
        indien='>'
    if extension == '.IMA':
        indien='<'
    if extension == '.rvb':
        komplex=999
        nbBytes=1
    if nktemps>0:
        offset, nbBytes, type, komplex, radar, indienZ, namerien = _readDimparamZV2(imgName + '.dim')
        if indienZ != 'Z':
            indien=indienZ
    nblok=nblok*nbBytes*(1+komplex) ######## ATTENTION : ne marchera que pour des tivoli
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, indien+type, nbBytes, komplex, nblok, ncan)
    
    
    
def _imwread(imgName, extension, ncan):
    """ Reads a *.imw file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en unsigned short',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=2
    offset=0
    if extension == '.imw':
        typeA='H'
        endian='>'
    if extension == '.IMW':
        typeA='H'
        endian='<'
    komplex=0
    nblok=w*h*2
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
def _imlread(imgName, extension, ncan):
    """ Reads a *.iml file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en int',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=4
    offset=0
    if extension == '.iml':
        typeA='i'
        endian='>'
    if extension == '.IML':
        typeA='i'
        endian='<'
    komplex=0
    nblok=w*h*4
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
def _imfread(imgName, extension, ncan):
    """ Reads a *.imf file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en float',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=4
    offset=0
    if extension == '.imf':
        typeA='f'
        endian='>'
    if extension == '.IMF':
        typeA='f'
        endian='<'
    komplex=0
    nblok=w*h*4
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
def _imdread(imgName, extension, ncan):
    """ Reads a *.imd file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en double',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=8
    offset=0
    if extension == '.imd':
        typeA='d'
        endian='>'
    if extension == '.IMD':
        typeA='d'
        endian='<'
    komplex=0
    nblok=w*h*4
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
def _cxbread(imgName, extension, ncan):
    """ Reads a *.cxb file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en complex signed char',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=1
    offset=0
    nblok=w*h*2
    if extension == '.cxb':
        typeA='b'
        endian='>'
        komplex=1
    if extension == '.cxbtivo':
        typeA='b'
        endian='>'
        komplex=2
    if extension == '.cxbadts':
        typeA='b'
        endian='>'
        komplex=3
    if nktemps>0:
        offset, nbBytes, type, komplex, radar, indien, namerien = _readDimparamZV2(imgName + '.dim')
    if radar==1:
        komplex=11
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, type, nbBytes, komplex, nblok, ncan)
def _cxsread(imgName, extension, ncan):
    """ Reads a *.cxs file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en complex signed short',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=2
    offset=0
    nblok=w*h*4
    if extension == '.cxs':
        typeA='h'
        endian='>'
        komplex=1
    if extension == '.CXS':
        typeA='h'
        endian='<'
        komplex=1
    if extension == '.cxstivo':
        typeA='h'
        endian='>'
        komplex=2
    if extension == '.CXSTIVO':
        typeA='h'
        endian='<'
        komplex=2
    if extension == '.cxsadts':
        typeA='h'
        endian='>'
        komplex=3
    if extension == '.CXSADTS':
        typeA='h'
        endian='<'
        komplex=3
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)

def _cxfread(imgName, extension, ncan):
    """ Reads a *.cxf file. imgName should come with no extension. """
    w, h, nk, nktemps = _readDimZ(imgName + '.dim')
    if w==0:
        print('le fichier .dim n''est pas lisible')
        return 0,0,0,0,0
    print('image en float',w, ' ',h,'  canaux:',nk,' verif : ', nktemps)
    nbBytes=4
    offset=0
    if extension == '.cxf':
        typeA='f'
        endian='>'
        komplex=1
    if extension == '.CXF':
        typeA='f'
        endian='<'
        komplex=1
    if extension == '.cxftivo':
        typeA='f'
        endian='>'
        komplex=2
    if extension == '.CXFTIVO':
        typeA='f'
        endian='<'
        komplex=2
    if extension == '.cxfadts':
        typeA='f'
        endian='>'
        komplex=3
    if extension == '.CXFADTS':
        typeA='f'
        endian='<'
        komplex=3
    nblok=w*h*8
    return _readImage(imgName + extension, w, h, nk, nktemps, offset, endian+typeA, nbBytes, komplex, nblok, ncan)
    
def _readDimZ(dimFile):
    """ Reads a *.dim file and return width and height. """
    try :
        f=open(dimFile);
    except IOError:
        legx=dimFile+': est un fichier non ouvrable'
        print(legx)
        return 0,0,0,0
    else:
        tmp = f.readline().split()
        w = int(tmp[0])
        h = int(tmp[1])
        nk=1
        nktemps=0 
# sert de code retour si le fichier .dim n'a que 2 valeurs 
        if len(tmp)>2:
            print('Fichier .dim version longue (lecture 3eme parametre) ')
            nk = int(tmp[2]);
        if len(tmp)>3:
            print('Fichier .dim version longue (lecture 4eme parametre)')
            nktemps = int(tmp[3]);
        return w, h, nk, nktemps
        
def _readDimparamZV2(dimFile):
    """ Reads a *.dim file and return width and height. """     
    offsetZ=0
    nbBytesZ=1
    typeZ='B'
    komplexZ=0
    radarZ=0
    endianZ='Z'
    namima=""
    with open(dimFile) as f:
        tmpK = f.readline()
        while tmpK!='':
            tmp = tmpK.split()
            print(tmp[0], tmp[1])
            if tmp[0]=="-offset":
                # print("Lecture de l''offset  ",tmp[1])
                offsetZ=tmp[1]
            if tmp[0]=="-radar":
                # print("Lecture du radar  ",tmp[1])
                if tmp[1]=="ERS":
                    radarZ=1
            if tmp[0]=="-image":
                print("Le .dim contient le nom de l\'image : ",tmp[1])
                namima=tmp[1]
            if tmp[0]=="-bo":  # bug corrigÃ© !!!
                # print("endian ?? :   ",tmp[1])
                if tmp[1]=="SUN":
                    endianZ='>'
                if tmp[1]=="DEC":
                    endianZ='<'
            if tmp[0]=="-type":
                # print("Lecture du type  ",tmp[1])
                if tmp[1]=="U8":
                    nbBytesZ=1
                    typeZ='B'
                if tmp[1]=="U16":
                    nbBytesZ=2
                    typeZ='H'
                if tmp[1]=="S16":
                    nbBytesZ=2
                    typeZ='h'
                if tmp[1]=="S32":
                    nbBytesZ=4
                    typeZ='i'
                if tmp[1]=="U32":
                    nbBytesZ=4
                    typeZ='I'
                if tmp[1]=="FLOAT":
                    nbBytesZ=4
                    typeZ='f'
                if tmp[1]=="DOUBLE":
                    nbBytesZ=8
                    typeZ='d'
                if tmp[1]=="C8":
                    nbBytesZ=1
                    komplexZ=1
                    typeZ='b'
                if tmp[1]=="CS8":
                    nbBytesZ=1
                    komplexZ=1
                    typeZ='b'
                if tmp[1]=="CS8TIVO":
                    nbBytesZ=1
                    komplexZ=2
                    typeZ='b'
                if tmp[1]=="CS8ADTS":
                    nbBytesZ=1
                    komplexZ=3
                    typeZ='b'
                if tmp[1]=="CS16":
                    nbBytesZ=2
                    komplexZ=1
                    typeZ='h'
                if tmp[1]=="CS16TIVO":
                    nbBytesZ=2
                    komplexZ=2
                    typeZ='h'
                if tmp[1]=="CS16ADTS":
                    nbBytesZ=2
                    komplexZ=3
                    typeZ='h'
                if tmp[1]=="C32TIVO":
                    nbBytesZ=4
                    komplexZ=2
                    typeZ='f'
                if tmp[1]=="C32ADTS":
                    nbBytesZ=4
                    komplexZ=3
                    typeZ='f'
                if tmp[1]=="CFLOAT":
                    nbBytesZ=4
                    komplexZ=1
                    typeZ='f'
                    
            tmpK = f.readline()  

                
        return offsetZ, nbBytesZ, typeZ, komplexZ, radarZ, endianZ, namima
        
        
def _readImage(imgName, w, h, nkparam, nktemps, offset, typeA, nbBytes, komplex, nblok, ncan):
    print("lecture de ", imgName,' en quelconque', w, h, nkparam, ' offset ', offset, typeA, nbBytes,' complex',komplex,'blocksize',nblok)
    if ncan>0:
        print('lecture specifique du canal %d'%ncan)
#    print('parametre ncan : %d'%ncan)
    """ Reads an image coded in any binary format. """
    
    tagRNSAT=0
    nk=nkparam
    
    if  nkparam<0 :
        print(u'Fichier RNSat : procÃ©dure en test')
        nk=-nkparam
        tagRNSAT=1
    
    try :
        f=open(imgName,'rb');
    except IOError:
        legx=imgName+': est un fichier non ouvrable'
        print(legx)
        return 0,0,0,0,0
    else:
        f.seek(offset,0)
        if komplex==999:
            imgligne=np.empty([3*w])
            img = np.empty([h, w, 3])
            for i in range(0, h):
#                if i%100==0:
#                    print(u'Ligne lue %d'%i)
                record=f.read(nbBytes*3*w)
                imgligne = np.ndarray( 3*w, typeA, record)
                img[i, 0:h, 0]= imgligne[0:3*h:3]/255.
                img[i, 0:h, 1]= imgligne[1:3*h:3]/255.
                img[i, 0:h, 2]= imgligne[2:3*h:3]/255.

#            imgligne=np.empty([3*w])
#            for i in range(0, h):
#                for j in range(0, 3*w):
#                    imgligne[j] = struct.unpack( typeA, f.read(nbBytes))[0]
#                for j in range(0, w):
#                    img[i, j, 0]=imgligne[3*j]/255.
#                    img[i, j, 1]=imgligne[3*j+1]/255.
#                    img[i, j, 2]=imgligne[3*j+2]/255.
            return img,w,h,nk,nktemps
  
      
        if nk>1:
            tag3=1
            
        if nk==1 :
            tag3=0 
            nkmin=0
            nkmax=1    
            
        if  ncan==0:
            tag3=0 
            nkmin=0
            nkmax=nk
            if nkmax>1:
                tag3=1
#            print(nkmin)
#            print(nkmax)
#            print(tag3)
            
        if ncan>0:
            if ncan>nk:
                ncan=nk
            tag3=2
            nkmin=ncan-1
            nkmax=ncan


        if(ncan>1):
            f.seek(nblok*(ncan-1))
            
        if tag3==1:
            nkmin=0
            nkmax=nk
            if komplex==0:
                imgtot = np.empty([h, w, nk])
            else:
                imgtot = np.zeros([h, w, nk])+ 1j * np.zeros([h, w, nk])
            
            if tagRNSAT==1 :
                if komplex==0:
                    imgtotstep = np.empty([h*w*nk])
                if komplex==1:
                    imgtotstep = np.zeros([h*w*nk]) + 1j * np.zeros([h*w*nk])
                iutrnsat=0
                iblocRNSAT=h*w
            
            
        if komplex==0:
            img = np.empty([h, w])
            
        if komplex==1 or komplex==2 or komplex==11:
            #print(h,w)
            imgligne=np.empty([2*w])
            img = np.zeros([h, w])+ 1j * np.zeros([h, w])
            
        if komplex==3:
            imgampli = np.empty([h, w])
            imgphase = np.empty([h, w])
            img = np.zeros([h, w])+ 1j * np.zeros([h, w])
            
#        print('Boucle de lecture entre %d'%nkmin+' et %d'%nkmax+'   sur %d'%nk+' canaux'+'  (tag3=%d'%tag3+')')           
        if nk>1:
            print('Boucle de lecture entre %d'%nkmin+' et %d'%nkmax+'   sur %d'%nk+' canaux')
        
#        print('Verif typeA = %s'%typeA )
 

        for nkz in range(nkmin,nkmax):
            if tag3==1 or tag3==2:
                print('Lecture du canal %d'%(nkz+1)+'/%d'%nk)
#            else:
#                print('Lecture monocanal  %d'%(nkz+1)+'/%d'%nk)
                
#############                
            if komplex==0:           
                print(u'DonnÃ©es rÃ©elles. Nouvelle version de imz2mat  '+'%s'%typeA)
                record=np.zeros(nbBytes*w, dtype=np.byte() )             
                for i in range(0, h):
                    record = f.read(nbBytes*w)
                    img[i, :] = np.ndarray( w, typeA, record)
#                    for j in range(0, w):
#                        img[i, j] = struct.unpack( typeA, record[nbBytes*j:nbBytes*j+nbBytes])[0]

############  komplex > 0  : trois cas 
                    
            if komplex==1 or komplex==11:   # cas des complexes standards :
                                            # partie rÃ©elle puis partie imaginaire
#                for i in range(0, h):
#                    for j in range(0, 2*w):
#                        imgligne[j] = struct.unpack( typeA, f.read(nbBytes))[0]
#                    for j in range(0, w):
#                        img[i, j] = imgligne[2*j]+imgligne[2*j+1]*1j
            
#>>>>>>>>>>>>  a incorporer !!
#   imgligneZ =  numpy.ndarray( 2*nfencboucle, '<h', record[4*vigdebcol:4*(vigdebcol+nfencboucle)]) 
                
                print(u'DonnÃ©es complexes (standard). Nouvelle version de imz2mat  '+'%s'%typeA)
                record=np.zeros(nbBytes*w*2, dtype=np.byte() )                       
                for i in range(0, h):
                    record = f.read(nbBytes*w*2)
                    imgligne =  np.ndarray( 2*w, typeA, record)
                    img[i,:]=imgligne[0:2*w:2]+1j*imgligne[1:2*w:2]
#                    for j in range(0, w):
#                        img[i, j] = imgligne[2*j]+imgligne[2*j+1]*1j
                    
            if komplex==11:
                valmoyR = np.mean(img.real)
                valmoyI = np.mean(img.imag)
                img = img.real-valmoyR+(img.imag-valmoyI)*1j                
## komplex==2                  
            if komplex==2:  # d'abord la partie rÃ©elle, puis la partir imaginaire
                for i in range(0, h):
                    for j in range(0, w):
                        imgligne[j] = struct.unpack( typeA, f.read(nbBytes))[0]
                    for j in range(0, w):
                        img[i, j] =  imgligne[j]
                for i in range(0, h):
                    for j in range(0, w):
                        imgligne[j] = struct.unpack( typeA, f.read(nbBytes))[0]
                    for j in range(0, w):
                        img[i, j] =  img[i, j] + imgligne[j]*1j           
## komplex==3                        
            if komplex==3: # d'abord l'amplitude, puis la phase
                imgampli = np.empty([h, w])
                imgphase = np.empty([h, w])
                for i in range(0, h):
                    for j in range(0, w):
                        imgampli[i, j] = struct.unpack( typeA, f.read(nbBytes))[0]
                for i in range(0, h):
                    for j in range(0, w):
                        imgphase[i, j] = struct.unpack( typeA, f.read(nbBytes))[0]
                for i in range(0, h):
                    for j in range(0, w):
                        img[i, j] = imgampli[i, j]*(cos(imgphase[i, j])+sin(imgphase[i, j])*1j)
###################################################################################
                        
                        
            if tag3==1 and tagRNSAT==0:
                imgtot[:,:,nkz]=img[:,:]   

                    
            if tag3==1 and tagRNSAT==1:  #horrible verrue
                for iutloop in range(iblocRNSAT) :
                    jbase=iutloop%w
                    ibase=int(iutloop/w)
                    jspe=(iutrnsat%nk)*iblocRNSAT
                    ispe=int(iutrnsat/nk)
                    imgtotstep[ispe+jspe]=img[ibase,jbase]
                    iutrnsat=iutrnsat+1
 

        if tagRNSAT==1:  # je ne m'en suis pas sorti avec les reshape
            ispe=w*h
            for iut in range(nk):
                isk=iut*iblocRNSAT
                for jut in range(h) :
                    imgtot[jut,:,iut]=imgtotstep[jut*w+isk:(jut+1)*w+isk]
               
        if tag3==0 or tag3==2:
            return img,w,h,nk,nktemps
        else:
            print('retour tableau 3-D (%dx%dx%d)'%(w,h,nk))
            return imgtot,w,h,nk,nktemps
#
def dat2mat(imgName):
#
# on rajoute un test dur l'existence mÃªme du fichier
#

    print('dat2mat : version Janvier 2018')

    try :
        fin=open(imgName,'rb');
    except IOError:
        legx=imgName+': est un fichier non ouvrable'
        print(legx)
        print(u'Echec Ã  l\'appel de dat2mat')
        return 0,0,0,0
     

    firm = fin.read(4)  
    nlig = struct.unpack("h",fin.read(2))[0]
    ncol = struct.unpack("h",fin.read(2))[0]

#    
    firm=np.zeros(8*ncol, dtype=np.byte() ) 
    imgcxs = np.empty([nlig-1, ncol], dtype=np.complex64())
    
# on elimine la premiÃ¨re ligne : cf la doc !!
    firm = fin.read(8*ncol)
    
    for iut in range(nlig-1):
        firm = fin.read(8*ncol)
        imgligne =  np.ndarray( 2*ncol, 'f', firm)
        imgcxs[iut,:]=imgligne[0:2*ncol:2]+1j*imgligne[1:2*ncol:2]


    return imgcxs, ncol, nlig-1, 1, 1
#
#######################################################################################
#######################################################################################
    #  FIN DES LECTURES
#######################################################################################
#######################################################################################
#  OUTILS VISU
#######################################################################################
#######################################################################################
#
def visusarbase(tabima,zparam,tagspe):
#
# on commence par eliminer les tableaux manifestement trop petits...
# ainsi que les tableaux reduit a la valeur 0 (entiere)
#
    if isinstance(tabima,int)==True:
        legspe='Pas de visualisation : Tableau nul'
        print(legspe)
        return
        
    RSI=tabima.size
    if RSI<16:
        legspe='Pas de visualisation : Tableau manifestement beaucoup trop petit (%d) pour etre une image : pas d''affichage' %(RSI)
        print(legspe)
        return 0
        
    kparam=3
    malegende='Essai de titre'
    if zparam!=-999:
        kparam=zparam
#    else:
#        print("visusar sans second parametre'
        
    R=tabima.shape
    ZZ=len(R)
    if ZZ==3:  # on traite Ã  part les couleurs  
        malegende='3 canaux couleurs (RVB)'
        print("Affichage comme image en couleur (3 canaux)")       
        if tagspe > 0 :
            plt.figure()
        plt.imshow(tabima)       
        if tagspe > 0 :
            plt.show()
        return 0
        
        

    if np.isrealobj(tabima)==True:
        print("Affichage d'une image reelle")
        BB=tabima
    if np.isrealobj(tabima)==False:
        print("Affichage d'une image complexe : on prend le module")
        BB=abs(tabima)
    valmin=np.min(BB)
    valmax=np.max(BB)
    valsig=np.std(BB)
    valmoy=np.mean(BB)
    legx='Min %.3f   Max %.3f    Moy %.3f   Ect %.3f ' %(valmin,valmax,valmoy,valsig)
 
    
    
    if kparam>0:
        valseuil=valmoy+kparam*valsig
        malegende='Image seuillee : valmoy + %.3f sigma  (%.2f)' %(kparam,valseuil)
        if valseuil>valmax:
            valseuil=valmax
            malegende='Image sans seuillage'
        masque=BB<valseuil
        BB=BB*masque+(1-masque)*valseuil
        
    if kparam<0:
        valseuil=-kparam
        malegende='Image seuillee : %.2f' %(valseuil)
        masque=BB<valseuil
        BB=BB*masque+(1-masque)*valseuil
        
    if kparam==0:
        malegende='Image sans seuillage'
              
    if tagspe <0 :
        print('Visusar sans affichage')
        print(legx)
        return(BB)              
              
    if tagspe > -1 :
        if tagspe>0 :
            plt.figure()
            plt.xlabel(legx)
        plt.imshow(BB)   
        plt.gray()        
 
    if kparam>0: 
        print('Image seuillee : valmoy + %.3f sigma  (%.2f)' %(kparam,valseuil) )
    else :
        print('Image non seuillee : entre %.3f et %.3f ' %(valmin, valmax))
    
    if tagspe>0:              
        plt.title(malegende)    
        ncol=np.size(tabima,1)
        nlig=np.size(tabima,0)
        print('plt.show dans visusar : image %d x %d'%(ncol,nlig))
        plt.show()
        
    return BB
    
    
def visusar(tabima,zparam=-999):
    """
    affichage d'images plutot radar.  Si image complexe : affichage de la valeur absolue
    
    plt.show() incorporÃ© dans cette routine
    
    Arguments en entrÃ©e : 1 ou 2
    
        argument 1 : tableau 2D image
        
        argument 2 (facultatif) : facteur de la formule <<valeur moyenne + fak * Ã©cart type >>
        Si ce facteur est nul, l'image ne sera pas seuillÃ©e
        
    Argument en sortie : le tableau affichÃ© (avec seuillage)
        
    Utilisez visusarZ (mÃªme syntaxe) pour Ã©viter le plt.show()     
    
    Utilisez visusarW (mÃªme syntaxe) pour n'avoir aucun affichage 
    """
    return visusarbase(tabima,zparam,1) 
    
def visusarZ(tabima, *therest):
    nnn=-999
    if(len(therest)==1):
        nnn=therest[0]
    return visusarbase(tabima,nnn,0) 
    
    
def visusarW(tabima, *therest):
    nnn=-999
    if(len(therest)==1):
        nnn=therest[0]
    return visusarbase(tabima,nnn,-1)
    
####################################################################
    # fin outils visu
####################################################################
     

#######################################
def dimimabase(imgName):
    
    if imgName.endswith('.dim') :
        w, h, nk, nktemps  = _readDimZ(imgName)
    else :       
        imgName = os.path.splitext(imgName)[0]
        w, h, nk, nktemps  = _readDimZ(imgName + '.dim')
#    print('n colonnes '+'%d'%w+'  nlignes '+'%d'%h+' ncanaux '+'%d'%nk)
    return w, h, nk, nktemps    
    
    
#######################################
def dimimadim(imgNameParam):
    
    if imgNameParam.endswith('.dim') :
        return(_readDimparamZV2(imgNameParam))
        
    else :
        return( 0, 0, 0 )
        w, h, nk, nktemps, imgName  = _readDimZ(imgNameParam)
      
    
    #
#######################################
    # compatibilitÃ© ascendante
#
def dimima(imgName):
    return(dimimabase(imgName))
#
#
######################################################
######################################################
# SEPTEMBRE 2017
#

typecode='<f'  # < : little endian
hdrcode='byte order = 0'  # pour .hdr d'IDL/ENVI
imacode='-byteorder = 0'  # pour .dim (mesure conservatoire)


def mat2imz( tabimage, nomimage, *therest):
    
    """
    Procedure pour ecrire un tableau dans un fichier au format TelecomParisTech
    Le tableau sera archivÃ© en :
        .ima si tableau 8 bits
        .IMF sinon
        .CXF si complexe   
    Si le tableau est Ã  3 dimensions (pile TIVOLI), l'archivage se fera en .IMA
    Exemple d'appel :
    mat2imz( montableau2d, 'MaSortie')
    Pour avoir aussi  le fichier .hdr d'IDL
    mat2imz( montableau2d, 'MaSortie', 'idl')
    """




    nomdim=nomimage+'.dim'
    
    taghdr=0
    testchar=0
       
    if(len(therest)==1):
        if therest[0]=="idl" :
            taghdr=1
    
    ndim=np.ndim(tabimage)
    if(ndim<2):
        print('mat2imz demande un tableau 2D ou 3D')
        return
    if(ndim>3):
        print('mat2imz demande un tableau 2D ou 3D')
        return

 
    nlig=np.size(tabimage,0)
    ncol=np.size(tabimage,1)
    nplan=1  # par defaut.. pour idl
#
# Cas image 2D
#      
    if ndim==2:      
        fp=open(nomdim,'w')
        fp.write('%d'%ncol+'  %d'%nlig)
        fp.close()        
        imode=np.iscomplex(tabimage[0][0])
        if imode==True :
            nomimagetot=nomimage+'.CXF'
            fp=open(nomimagetot,'wb')
            for iut in range(nlig):
                for jut in range(ncol):
                    fbuff=float(tabimage.real[iut][jut])
                    record=struct.pack( typecode, fbuff)
                    fp.write(record)
                    fbuff=float(tabimage.imag[iut][jut])
                    record=struct.pack( typecode, fbuff)
                    fp.write(record)
            fp.close()
            
        else :
            mintab=np.min(tabimage)
            maxtab=np.max(tabimage)
            if(mintab>-0.0001 ) :
                if (maxtab<255.0001):
                    testchar=1
                    nomimagetot=nomimage+'.ima'
                    ucima=np.uint8(tabimage)
                    fp=open(nomimagetot,'wb')
                    for iut in range(nlig):
                        for jut in range(ncol):
                            record=struct.pack( 'B', ucima[iut][jut])
                            fp.write(record)
                    
                else :
                    nomimagetot=nomimage+'.IMF'
                    fp=open(nomimagetot,'wb')
                    for iut in range(nlig):
                        for jut in range(ncol):
                            fbuff=float(tabimage[iut][jut])
                            record=struct.pack( typecode, fbuff)
                            fp.write(record)
            fp.close()
                
        
        
        
               
    if ndim==3:    
        nplan=np.size(tabimage,2)       
        imode=np.iscomplex(tabimage[0][0][0])
        mintab=np.min(tabimage)
        maxtab=np.max(tabimage)
        if(mintab>-0.0001 ) :
            if (maxtab<255.0001):
                testchar=1
        
        
        fp=open(nomdim,'w')      
        fp.write('%d'%ncol+'  %d'%nlig+'  %d'%nplan+'   1'+'\n')
        if imode==True :
            fp.write('-type CFLOAT')
        else :
            if testchar==0:
                fp.write('-type FLOAT')
            if testchar==1:
                fp.write('-type U8')
        fp.close()
        
        nomimagetot=nomimage+'.IMA'
        fp=open(nomimagetot,'wb')
        if imode==True :
            for lut in range(nplan):
                for iut in range(nlig):
                    for jut in range(ncol):
                        fbuff=float(tabimage.real[iut][jut][lut])
                        record=struct.pack( typecode, fbuff)
                        fp.write(record)
                        fbuff=float(tabimage.imag[iut][jut][lut])
                        record=struct.pack( typecode, fbuff)
                        fp.write(record)
        else :
            if(testchar==1) :
                for lut in range(nplan):
                    ucima=np.uint8(tabimage[:,:,lut])
                    for iut in range(nlig):
                        for jut in range(ncol):
                            record=struct.pack( 'B', ucima[iut][jut])
                            fp.write(record)
            else :
                for lut in range(nplan):
                    for iut in range(nlig):
                        for jut in range(ncol):
                            fbuff=float(tabimage[iut][jut][lut])
                            record=struct.pack( typecode, fbuff)
                            fp.write(record)
                            
                            
        fp.close()
                
    if taghdr==1 :
        noffset=0
        nomhdr=nomimagetot+'.hdr'
        fp=open(nomhdr,'w')   
        fp.write('ENVI \n')         
        fp.write('{Fichier produit par tiilab.mat2imz (python) } \n')
        fp.write('samples = %d'%ncol+'\n')
        fp.write('lines = %d'%nlig+'\n')
        fp.write('bands = %d'%nplan+'\n')
        fp.write('header offset = %d'%noffset+'\n')
        fp.write('file type = ENVI Standard \n')
        if imode==True :
            fp.write('data type = 6 \n')
        else :
            if(testchar==1) : 
                fp.write('data type = 1  \n')
            else : 
                fp.write('data type = 4  \n')
            
        fp.write('interleave = bsq \n')
        fp.write(hdrcode+'\n')
        if imode==True :
            fp.write('complex function = Magnitude  \n')
        
        
        fp.close()
        
        
        
        
######################################################
######################################################
######################################################

    
    