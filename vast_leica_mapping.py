from scyjava import jimport
import os, pathlib, glob, sys
import math
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image
from PIL import Image
import imagej
from skimage.io import imread
ij = imagej.init('sc.fiji:fiji')#, mode='interactive')
from somiteCounting.orientfish import orient_fish

print(f"ImageJ2 version: {ij.getVersion()}")

ImageReader   = jimport('loci.formats.ImageReader')
IFormatReader = jimport('loci.formats.IFormatReader')
IMetadata     = jimport('loci.formats.meta.IMetadata')
MetadataTools = jimport('loci.formats.MetadataTools')
ImageReader   = jimport('loci.formats.ImageReader')
DebugTools    = jimport('loci.common.DebugTools')
DateTimeZone  = jimport('org.joda.time.DateTimeZone')



def map_well_to_vast(data_path, experiment_name, use_nplanes=False):
    """
    Maps the well to the VAST file and prints metadata information.
    """
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        return
    print(f"Processing data path: {data_path}")

    if not os.path.exists(os.path.join(data_path, experiment_name)):
        print(f"{os.path.join(data_path, experiment_name)} is not a valid file.")
        return
    print(f"Processing experiment: {experiment_name}")
    
    file_path = os.path.join(data_path, experiment_name, 'Leica', f"{experiment_name}.lif")
    if not os.path.exists(file_path):
        print(f"Leica file {file_path} does not exist.")
        return
    print(f"Using Leica file: {file_path}")

    vast_path = os.path.join(data_path, experiment_name, 'VAST images')
    if not os.path.exists(vast_path):
        print(f"VAST path {vast_path} does not exist.")
        return
    print(f"Using VAST path: {vast_path}")

    plates = glob.glob(os.path.join(data_path, experiment_name, 'VAST images', 'Plate*'))
    print(f"Found plates: {plates}")
    well_dict={}
    plate_dict={}
    for p in plates:
        if not os.path.isdir(p):
            print(f"Skipping {p} as it is not a directory.")
            continue
        print(f"Processing plate: {p}")
        wells = glob.glob(os.path.join(p, 'Well_*'))
        print(f"Found wells: {wells}")
        if not wells:
            print(f"No wells found in {p}.")
            continue
        plate_dict[p]=len(wells)
        for w in wells:
            print(f"Processing well: {w}")
            vast_files = glob.glob(os.path.join(w, '*.tiff'))
            if not vast_files:
                print(f"No VAST files found in {p}.")
                continue
            ti_c=0
            ti_m=0
            n_found=0
            for vf in vast_files:
                if '.tiff' not in vf:continue
                ti_c += os.path.getctime(vf)
                ti_m += os.path.getmtime(vf)
                n_found += 1
            ti_m = ti_m / n_found
            ti_c = ti_c / n_found
            ti_m = int(ti_m)
            ti_c = int(ti_c)
            print(ti_m,'  ',datetime.fromtimestamp(ti_m)," ",ti_c," ",datetime.fromtimestamp(ti_c),'  ',vf)
            
            well_dict[w] = ti_m
    
    print('well_dict=', well_dict)

    reader = ImageReader()
    meta   = MetadataTools.createOMEXMLMetadata()
    DebugTools.enableLogging("OFF")

    reader.setMetadataStore(meta)
    reader.setId(file_path)
    meta    = reader.getMetadataStore() # Retrieves the metadata object
    nSeries = reader.getSeriesCount()

    print('nSeries=', nSeries)
    for iImage in range(nSeries):
        reader.setSeries(iImage)
        imname=str(meta.getImageName(iImage))

        if imname==None or imname=='':
            print('Image name is empty, skipping this image.')
            continue

        if 'Plate' not in imname and 'plate' not in imname and use_nplanes==False:
            print(f"Skipping image {iImage} with name {imname} as it does not contain 'Plate' or 'plate'.")
            continue

        #if use_nplanes==True:
        #    nPlanes = meta.getPlaneCount(iImage) # Number of Planes within the image
        #    for p in plate_dict:
        #        if math.fabs(nPlanes-plate_dict[p])/plate_dict[p]<0.2:
        #            print(f"Image {iImage} with name {imname} has {nPlanes} planes, which is more than 20% off nPlanes={nPlanes} nwell={plate_dict[p]}.")
        #            continue
        #        else:
        #            print(f"Image {iImage} with name {imname} matched to plate {p} with {plate_dict[p]} wells based on nPlanes={nPlanes}.")



        print(' --->>> Processing Image name:', imname)
        im_type = imname.split('/')[-1]
        if 'Trigger' in im_type:
            print(f"Skipping image {iImage} with type {im_type} as it contains 'Trigger'.")
            continue

        nPlanes = meta.getPlaneCount(iImage) # Number of Planes within the image
        
        print("Image #",iImage,"  imagename=",meta.getImageName(iImage), " nPlanes=",nPlanes)
        print("acq date =", meta.getImageAcquisitionDate(iImage).asDateTime(DateTimeZone.UTC).getMillis())
        print("acq date =", meta.getImageAcquisitionDate(iImage).asDateTime(DateTimeZone.UTC))
        timestamp=int(meta.getImageAcquisitionDate(iImage).asDateTime(DateTimeZone.UTC).getMillis())/1000.
        dt_object = datetime.fromtimestamp(timestamp)
        print('datetime=',dt_object)
        for iPlane in range(nPlanes):
            zct = reader.getZCTCoords(iPlane)
            #print("Plane C:",zct[0]," Z:",zct[1]," T:",zct[2])
            if meta.getPlaneDeltaT(iImage,iPlane)==None:
                print(f"Skipping plane {iPlane} for image {iImage} as PlaneDeltaT is None.")
                continue
            dt_object = datetime.fromtimestamp(timestamp+float(meta.getPlaneDeltaT(iImage,iPlane).value().doubleValue()))
            sel_well=None
            minDeltaT=9999999999999999
            for w in well_dict:
                delta_t = math.fabs(timestamp+float(meta.getPlaneDeltaT(iImage,iPlane).value().doubleValue())-well_dict[w])
                if delta_t<minDeltaT:
                    sel_well=w
                    minDeltaT=delta_t
            print("   plane= ", iPlane,"    Acquired at ",meta.getPlaneDeltaT(iImage,iPlane).value().doubleValue(), '  ',dt_object,'  ',sel_well, '   delta_t=',minDeltaT)
            print('   well time= ', well_dict[sel_well],'  ',datetime.fromtimestamp(well_dict[sel_well]))
            print('   timestamp= ', timestamp)
            #bytes_arr=np.array(reader.openBytes(iPlane), np.uint8)
            bytes_arr=np.array(reader.openBytes(iPlane))
            arr = np.frombuffer(bytes_arr, dtype=np.uint16)  # or dtype='<u2' / '>u2' if needed
            #arr = bytes_arr.view(np.uint16)

            im_arr = np.reshape(arr,  (2048,2048))

            # safe min/max and avoid division by zero
            minv = float(im_arr.min())
            maxv = float(im_arr.max())
            if maxv == minv:
                # constant image -> choose 0 or 255 (here we use 0)
                arr8 = np.zeros_like(im_arr, dtype=np.uint8)
                arr16 = np.zeros_like(im_arr, dtype=np.uint16)
            else:
                # linear scale to 0..255
                arr8 = ((im_arr - minv) / (maxv - minv) * 255.0).round().astype(np.uint8)
                arr16 = ((im_arr - minv) / (maxv - minv) * 65535.0).round().astype(np.uint16)



            im = Image.fromarray(im_arr, mode='I;16')
            im_norm = Image.fromarray(arr16, mode='I;16')
            im_norm8 = Image.fromarray(arr8, mode='L')
            mkdir_out = sel_well.replace('VAST images', 'Leica images')
            if not os.path.exists(mkdir_out):
                os.makedirs(mkdir_out)
                print(f"Created output directory: {mkdir_out}")
            out_path = os.path.join(mkdir_out, "Leica_{}.tiff".format(im_type))
            out_path_norm = os.path.join(mkdir_out, "Leica_{}_norm.tiff".format(im_type))
            out_path_norm8 = os.path.join(mkdir_out, "Leica_{}_norm8.tiff".format(im_type))
            im.save(out_path)
            im_norm.save(out_path_norm)
            im_norm8.save(out_path_norm8)


def scan_lif(data_path, experiment_name):
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        return
    print(f"Processing data path: {data_path}")

    if not os.path.exists(os.path.join(data_path, experiment_name)):
        print(f"{os.path.join(data_path, experiment_name)} is not a valid file.")
        return
    print(f"Processing experiment: {experiment_name}")
    
    file_path = os.path.join(data_path, experiment_name, 'Leica', f"{experiment_name}.lif")

    if not os.path.exists(file_path):
        print(f"Leica file {file_path} does not exist.")
        return
    print(f"Using Leica file: {file_path}")


    reader = ImageReader()
    meta   = MetadataTools.createOMEXMLMetadata()
    DebugTools.enableLogging("OFF")

    reader.setMetadataStore(meta)
    reader.setId(file_path)
    meta    = reader.getMetadataStore() # Retrieves the metadata object
    nSeries = reader.getSeriesCount()

    print('nSeries=', nSeries)
    for iImage in range(nSeries):
        reader.setSeries(iImage)
        imname=str(meta.getImageName(iImage))

        if imname==None or imname=='':
            print('Image name is empty, skipping this image.')
            continue

        print(' --->>> Processing Image name:', imname)
        im_type = imname.split('/')[-1]

        nPlanes = meta.getPlaneCount(iImage) # Number of Planes within the image
        
        print("Image #",iImage,"  imagename=",meta.getImageName(iImage), " nPlanes=",nPlanes)
        print("acq date =", meta.getImageAcquisitionDate(iImage).asDateTime(DateTimeZone.UTC).getMillis())
        print("acq date =", meta.getImageAcquisitionDate(iImage).asDateTime(DateTimeZone.UTC))
        timestamp=int(meta.getImageAcquisitionDate(iImage).asDateTime(DateTimeZone.UTC).getMillis())/1000.
        dt_object = datetime.fromtimestamp(timestamp)
        print('datetime=',dt_object)
        for iPlane in range(nPlanes):
            zct = reader.getZCTCoords(iPlane)
            print("Plane C:",zct[0]," Z:",zct[1]," T:",zct[2])
            if meta.getPlaneDeltaT(iImage,iPlane)==None:
                print(f"Skipping plane {iPlane} for image {iImage} as PlaneDeltaT is None.")
                continue
            dt_object = datetime.fromtimestamp(timestamp+float(meta.getPlaneDeltaT(iImage,iPlane).value().doubleValue()))
            
            print("   plane= ", iPlane,"    Acquired at ",meta.getPlaneDeltaT(iImage,iPlane).value().doubleValue(), '  ',dt_object)



if __name__ == '__main__':
    """
    Main function to execute the mapping.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Map well to VAST files and orient fish images.")
    parser.add_argument('--data_path', type=str, default="data", help="Path to the data directory.")
    parser.add_argument('--experiment_name', type=str, default="VAST_2025-07-08", help="Name of the experiment.")
    parser.add_argument('--scan_lif', action='store_true', help="If set, only scan the LIF file for metadata.")
    parser.add_argument('--use_nplanes', action='store_true', help="If set, use the planes to match.")

    args = parser.parse_args()

    file_path = args.data_path
    experiment_name = args.experiment_name
    if args.scan_lif:
        scan_lif(file_path, experiment_name)
    else:
        map_well_to_vast(file_path, experiment_name, use_nplanes=args.use_nplanes)
        #orient_fish(data_path=file_path, experiment_name=experiment_name)
