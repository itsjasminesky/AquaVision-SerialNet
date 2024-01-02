import os
import easyocr
import tqdm
import shutil
import numpy as np

from utils import read_image, list_image_path, write_json


class EasyOCR2Filter:
    def __init__(self, image_dir, output_dir=None, gpu=False) -> None:
        """Constructor of the application."""
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.gpu = gpu
        return None

    def perfromOCR(self, image_path_list):
        """Performing ocr image path list."""
        # GETING OCR
        reader = easyocr.Reader(['en'], gpu=self.gpu)
        
        result_list = []
        for image_path in tqdm.tqdm(image_path_list, 'Performing OCR'):
            result = reader.readtext(image_path)
            result_list.append(result)
        return result_list
    
    def convert(self, image_path_list, result_list):
        """Converting result to google vision filter response."""
        # LOOPING THROUGH EACH DATA
        data = []
        for image_path, result in zip(image_path_list, result_list):
            # GETTING FILE INFORMATION
            image_dir, image_name = os.path.split(image_path)
            image = read_image(image_path)
            h, w, _ = image.shape
            file_info = {
                "width": w,
                "height": h,
                "imageDirPath": image_dir,
                "imageName": image_name
            }
            
            # GETTING TEXT INFORMATION
            complete_text = ''
            individual_text = []
            for poly, text, confidence in result:
                individual_text.append({
                    "confidence": confidence,
                    "description": text,
                    "boundingPoly": np.array(poly, dtype=int).tolist(),
                    "symbols": []
                })
                complete_text += text + '\n'
            
            # APPENDING DATA
            data.append({
                'fileInfo': file_info,
                'textAnnotations': {'completeText': complete_text, 'individualText': individual_text}
            })
        return data
    
    def writeData(self, data):
        """Write data to output dir."""
        os.makedirs(self.output_dir, exist_ok=True)

        # LOOPING THROUGH EACH DATA
        for d in data:
            # COPYING IMAGE
            image_dir, image_name = d['fileInfo']['imageDirPath'], d['fileInfo']['imageName']
            src_path = os.path.join(image_dir, image_name)
            dst_path = os.path.join(self.output_dir, image_name)
            shutil.copyfile(src_path, dst_path)
            
            # WRITING ANNOTATION FILE
            image_name = os.path.basename(image_name)
            image_case = os.path.splitext(image_name)[0]
            file_path = os.path.join(self.output_dir, image_case + '.json')
            write_json(file_path, d)
        return None
    
    def perform_ocr(self):
        # GETING IMAGE PATHS
        image_path_list = list_image_path(self.image_dir)

        # PERFORMING OCR
        result_list = self.perfromOCR(image_path_list)
        
        # CONVETING RESULT
        data = self.convert(image_path_list, result_list)
        return data
    
    def main(self):
        """Main function of the application."""
        # Performing OCR
        data = self.perform_ocr()
        
        # WRITING DATA
        self.writeData(data)
        return None

if __name__ == "__main__":
    # THIS IS USED FOR UNIT TESTING
    image_dir = 'INPUT/processed2'
    output_dir = 'INPUT/ocr_filter2'
    
    eg = EasyOCR2Filter(image_dir, output_dir)
    eg.main()