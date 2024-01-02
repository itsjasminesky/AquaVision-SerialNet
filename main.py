import os
import argparse
import re
import pandas as pd

from easyocr2filter import EasyOCR2Filter
from extraction import ExtractionAlgo
from utils import resize_image


class ExtractData:
    def __init__(self, image_dir, output_file, gpu) -> None:
        """Constructor."""
        self.image_dir = image_dir
        self.output_file = output_file
        self.gpu = gpu
        return None
    
    def write_data(self, data):
        """Writing data to output file."""
        # MASKING OUTPUT DIR
        output_dir = os.path.dirname(self.output_file)
        if output_dir != '':
            os.makedirs(output_dir, exist_ok=True)

        columns = ['imageFile', 'deployYear', 'imageSerialNumber', 'imageAssetID', 'imageSerialNumber Match', 'imageAssetID Match']
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(self.output_file)
        return None

    def main(self):
        """Main function of application."""
        # RESIZING IMAGE
        resize_image(self.image_dir)
        
        # PERFORMING OCR
        easyocr = EasyOCR2Filter(self.image_dir, gpu=self.gpu)
        anno_list = easyocr.perform_ocr()
        
        # PERFORMING EXTRACTION
        ext_algo = ExtractionAlgo()
        data = []
        for anno in anno_list:
            # GETTING RESUT
            asset_id, serial_num = ext_algo.get_result(anno)
            
            # GETTING IMAGE NAME
            image_name = anno['fileInfo']['imageName']
            
            # GETTING YEAR
            year = re.findall(r'_(\d{4})\d{4}_', image_name)
            if year:
                year = year[0]
            else:
                year = 'UNKNOWN'
            
            # GETTING SERIAL NUMBER
            sns = serial_num[-1]
            sn = ''
            for s in sns:
                _s = re.split(r'SN ?: ?', s)[-1]
                sn += ' ' + _s
            sn = sn.strip()
            
            # APPENDING DATA
            _data = [
                image_name,
                year,
                sn,
                ' '.join(asset_id[-1]),
                serial_num[0],
                asset_id[0],
            ]
            print(_data)
            data.append(_data)
        
        # WRITING FILE
        self.write_data(data)
        return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='This application will extract asset id and sn from images.')
    argparser.add_argument('image_dir', type=str, help='Path to directory that contain images.')
    argparser.add_argument('output_file', type=str, help='Path to csv file in which result will save.')
    argparser.add_argument('--gpu', default=False, action='store_true', help='Enabling GPU for inference.')
    args = argparser.parse_args()

    ed = ExtractData(args.image_dir, args.output_file, args.gpu)
    ed.main()