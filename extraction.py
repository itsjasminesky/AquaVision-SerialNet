import os
import glob
import re
from natsort import natsorted
import math
import numpy as np
import json

class ExtractionAlgo:
    def __init__(self, annotation_dir=None) -> None:
        """Constructor of the application."""
        self.annotation_dir = annotation_dir
        
        self._num_repl = {
            'O': '0',
            'o': '0',
        }
        
        self._asset_id_regex = [
            ('Exact', r'\b[A-Z]+-?\d{5}-\d+'), # ATAPL-70150-00006, ATAPL66943-004
            ('Exact Num Replace', r'\b[A-Z]+-\d{5}-\d+'), # ATAPL-66662-O001
            ('Spaced', r'\b[A-Z]{5}-?\d{5} ?\d+'), # ATAPL-78438 00002
            ('Partial Start', r'\b[A-Z]{5}-?\d+-?'), # ATAPL67, ATAPL-58337
            ('Partial End', r'\d{5}-?\d{5}'), # 58320-00003
        ] 
        
        self._serial_num_regex = [
            ('Exact', r'\b\d{2}-\d{5}'), # 16-50122
            ('Partial Start', r'\bSN ?: ?'), # SN : , SN:, SN :,
            ('Poor', r' \d{3} '), # ` 125 `
        ]
        return None
    
    def read_json(file_name):
        with open(file_name, 'rb') as f:
            data_dict = json.load(f)
        return data_dict
    
    def get_annotation_file_path(self):
        """Getting list of annotation files."""
        return glob.glob(os.path.join(self.annotation_dir, '*.json'))
    
    def get_complete_text(self, anno):
        """Getting complete text string."""
        text = anno['textAnnotations']['completeText']
        text = re.sub('\n', ' ', text)
        return text
    
    def _replace_with_number(self, text):
        """Replacing in correct char with number"""
        for k, v in self._num_repl.items():
            text = text.replace(k, v)
        return text
    
    def _find_distance_btw_points(self, p1, p2):
        """Finding distance between two points."""
        distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        return distance
    
    def _complet_match(self, matches, anno):
        """Completing match result."""
        # GETTING WORD LIST
        words = anno['textAnnotations']['individualText']
        # FIRSTING FINDING CURRENT WORD
        current_idxs = []
        current_words = []
        for match in matches:
            for idx, word in enumerate(words):
                description = word['description']
                if match in description:
                    if idx not in current_idxs:
                        current_idxs.append(idx)
                        current_words.append(word)
        
        # FINDING CLOSEST WORD AND ADDING TO CURRENT
        new_matches = []
        for current_idx, current_word in zip(current_idxs, current_words):
            distances = []
            current_tr = current_word['boundingPoly'][1]
            for idx, word in enumerate(words):
                # SKIPPING CURRECT IDX
                if idx == current_idx:
                    distances.append(99999999) # Very large number to provent it to be appear as closest.
                    continue
                
                # FINDING DISTANCE
                next_tl = word['boundingPoly'][0]
                distance = self._find_distance_btw_points(current_tr, next_tl)
                
                # APPENDING DISTANCE
                distances.append(distance)
                
            # FINDING CLOSEST WORD
            closest_idx = np.argmin(distances)
            closest_word = words[closest_idx]
            
            # JOINING INFORMATION
            new_matches.append(current_word['description'] + closest_word['description'])
        return new_matches
    
    def _get_asset_id(self, anno):
        """Getting asset id using different algorithms."""
        # GETTING COMPLETE TEXT
        text = self.get_complete_text(anno)
        
        match_type, match = 'No', []
        # MATCHING
        for mt, regex in self._asset_id_regex:
            if mt == 'Exact Num Replace':
                text = self._replace_with_number(text)
            
            match = re.findall(regex, text)
            if match:
                match_type = mt
                
                # CORRECT MATCH FOR PARTAL START
                if mt == 'Partial Start':
                    match = [m.strip() for m in match]
                    match = self._complet_match(match, anno)
                break
        return (match_type, match)
    
    def _get_serial_number(self, anno):
        """Getting serial number using different algorithms."""
        # GETTING COMPLETE TEXT
        text = self.get_complete_text(anno)
        
        match_type, match = 'No', []
        # MATCHING
        for mt, regex in self._serial_num_regex:
            if mt == 'Exact Num Replace':
                text = self._replace_with_number(text)
            
            match = re.findall(regex, text)
            if match:
                match_type = mt
                
                # CORRECT MATCH FOR PARTAL START
                if mt == 'Partial Start':
                    match = [m.strip() for m in match]
                    match = self._complet_match(match, anno)
                break
        return (match_type, match)

    def get_result(self, anno):
        """Getting desire information."""
        # GETTING ASSET ID
        asset_id = self._get_asset_id(anno)
        
        # GETTING SERIAL NUMBER+
        serial_num = self._get_serial_number(anno)
        return asset_id, serial_num
    
    def main(self):
        """Main function of the application."""
        # GETTING LIST OF ANNOTATION PATH
        anno_file_paths = self.get_annotation_file_path()

        # GETTING DATA
        assert_id_count = 0
        serial_num_count = 0
        for file_path in natsorted(anno_file_paths):
            prefix = os.path.basename(file_path).split('.')[0]
            # if prefix != 'CTDPFB301_20230913_085400_recovered':
            #     continue
            
            # READING FILE
            anno = self.read_json(file_path)
            
            # GETTING RESUT
            asset_id, serial_num = self.get_result(anno)

            # PRINTING
            # print(f'{prefix} -> {asset_id}')
            if asset_id[-1]:
                assert_id_count += 1
                
            print(f'{prefix} -> {serial_num}')
            if serial_num[-1]:
                serial_num_count += 1
        print(f'Asset ID count: {assert_id_count}')
        print(f'SN ID count: {serial_num_count}')
        return None


if __name__ == "__main__":
    # IT IS USED FOR UNIT TESTING
    anno_dir = 'INPUT/ocr_filter2'
    
    ex = ExtractionAlgo(anno_dir)
    ex.main()