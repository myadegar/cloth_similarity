import os
import mimetypes
from PIL import Image
import numpy as np
import argparse

from rembg import remove, new_session
from deep_person_reid.torchreid.utils import FeatureExtractor
from deep_person_reid.torchreid import metrics

class ClothSimilarity():
    def __init__(self):
        self.session = new_session(model_name="u2net_cloth_seg")
        self.feature_extractor = FeatureExtractor(model_name='osnet_ain_x0_5',
            model_path='./osnet_ain_x0_5_imagenet.pth', device='cuda')

    def cloth_segmentation(self, input_dir_path, output_dir_path):
        result_data = {}
        os.makedirs(output_dir_path, exist_ok=True)
        files = os.listdir(input_dir_path)
        images_file = [file for file in files if str(mimetypes.guess_type(file)[0]).startswith('image')]
        images_path = [os.path.join(input_dir_path, image_file) for image_file in images_file]
        for image_path, image_file in zip(images_path, images_file):
            print(f' process on file: {image_path}')
            result_data[image_file] = {}
            file_name, ext = os.path.splitext(image_file)
            input_image = Image.open(image_path)
            segment_images, mask_images = remove(input_image, session=self.session, alpha_matting=True,
                                  post_process_mask=True, only_mask=False, concat_parts=False)
            segment_img = {}
            mask_img = {}
            crop_img = {}
            bbox_points = {}
            for idx, segment_image in enumerate(segment_images):
                binary_image = segment_image.convert('1')
                sum_pixel = np.sum(binary_image)
                H, W = segment_image.size
                pixel_ratio = sum_pixel / (H * W + 1)
                if pixel_ratio < 0.05:
                    continue
                cloth_mask_img = mask_images[idx]
                cloth_bbox = segment_image.getbbox()
                cloth_segment_img = segment_image.crop(cloth_bbox)
                cloth_crop_img = input_image.crop(cloth_bbox)
                ##
                if idx == 0: # upper body
                    segment_img['upper'] = cloth_segment_img
                    mask_img['upper'] = cloth_mask_img
                    crop_img['upper'] = cloth_crop_img
                    bbox_points['upper'] = cloth_bbox
                elif idx == 1: # lower body
                    segment_img['lower'] = cloth_segment_img
                    mask_img['lower'] = cloth_mask_img
                    crop_img['lower'] = cloth_crop_img
                    bbox_points['lower'] = cloth_bbox
                elif idx == 2: # other part
                    segment_img['other'] = cloth_segment_img
                    mask_img['other'] = cloth_mask_img
                    crop_img['other'] = cloth_crop_img
                    bbox_points['other'] = cloth_bbox
                    # modify upper segment
                    if 'upper' in segment_img.keys():
                        upper_segment_img = Image.composite(segment_images[2], segment_images[0], mask_images[2])
                        upper_cloth_mask_img = Image.composite(mask_images[2], mask_images[0], mask_images[2])
                        upper_cloth_bbox = upper_segment_img.getbbox()
                        upper_cloth_segment_img = upper_segment_img.crop(upper_cloth_bbox)
                        upper_cloth_crop_img = input_image.crop(upper_cloth_bbox)
                        ##
                        segment_img['upper'] = upper_cloth_segment_img
                        mask_img['upper'] = upper_cloth_mask_img
                        crop_img['upper'] = upper_cloth_crop_img
                        bbox_points['upper'] = upper_cloth_bbox
            ##
            result_data[image_file]['segment_img'] = segment_img
            result_data[image_file]['mask_img'] = mask_img
            result_data[image_file]['crop_img'] = crop_img
            result_data[image_file]['bbox_points'] = bbox_points
            #
            for body_part, img in segment_img.items():
                dir_path = os.path.join(output_dir_path, 'segment')
                os.makedirs(dir_path, exist_ok=True)
                output_path = os.path.join(dir_path, file_name + '_segment' + '_' + body_part + '.png')
                img.save(output_path)
            for body_part, img in mask_img.items():
                dir_path = os.path.join(output_dir_path, 'mask')
                os.makedirs(dir_path, exist_ok=True)
                output_path = os.path.join(dir_path, file_name + '_mask' + '_' + body_part + '.png')
                img.save(output_path)
            for body_part, img in crop_img.items():
                dir_path = os.path.join(output_dir_path, 'crop')
                os.makedirs(dir_path, exist_ok=True)
                output_path = os.path.join(dir_path, file_name + '_crop' + '_' + body_part + '.png')
                img.save(output_path)

        return result_data

    def similarity_calculation(self, result_cloth, result_person, cloth_mode, similarity_method, similarity_mode):
        cloth_image_files = []
        cloth_segment_images = []
        for image_file, data in result_cloth.items():
            segment_img = data['segment_img'].get(cloth_mode.lower())
            if segment_img is not None:
                cloth_image_files.append(image_file)
                segment_img = segment_img.convert(mode='RGB')
                cloth_segment_images.append(np.array(segment_img))
        ##
        person_image_files = []
        person_segment_images = []
        for image_file, data in result_person.items():
            segment_img = data['segment_img'].get(cloth_mode)
            if segment_img is not None:
                person_image_files.append(image_file)
                segment_img = segment_img.convert(mode='RGB')
                person_segment_images.append(np.array(segment_img))
        ##
        if similarity_method.lower() == 'reid':
            features_cloth = self.feature_extractor(cloth_segment_images)
            features_person = self.feature_extractor(person_segment_images)
        elif similarity_method.lower() == 'vae':
            pass
        else:
            raise Exception('Incorrect similarity method !')
        ##
        similarity_matrix = metrics.distance.cosine_distance(features_cloth, features_person)
        similarity_matrix = 1.0 - similarity_matrix.cpu().numpy()
        ##
        if similarity_mode.lower() == 'mean':
            similarity_scores = np.mean(similarity_matrix, axis=0)
        elif similarity_mode.lower() == 'max':
            similarity_scores = np.max(similarity_matrix, axis=0)
        else:
            raise Exception('Incorrect similarity mode !')
        ##
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_similarity_scores = similarity_scores[sorted_indices]
        sorted_person_image_files = [person_image_files[idx] for idx in sorted_indices]

        return sorted_person_image_files, sorted_similarity_scores


    def __call__(self, cloth_dir, person_dir, result_dir, cloth_mode='upper', similarity_method= 'reid', similarity_mode='mean'):
        cloth_output_dir = os.path.join(result_dir, 'cloth')
        person_output_dir = os.path.join(result_dir, 'person')
        result_cloth = self.cloth_segmentation(input_dir_path=cloth_dir, output_dir_path=cloth_output_dir)
        result_person = self.cloth_segmentation(input_dir_path=person_dir, output_dir_path=person_output_dir)
        sorted_person_image_files, sorted_similarity_scores = self.similarity_calculation(result_cloth, result_person,
                                                                                          cloth_mode, similarity_method, similarity_mode)
        return sorted_person_image_files, sorted_similarity_scores


if __name__ == "__main__":
    cloth_similarity = ClothSimilarity()
    ###
    # cloth_dir = './input_images/cloth'
    # person_dir = './input_images/person'
    # result_dir = './result_images'
    # cloth_mode = 'upper'
    # similarity_method = 'reid'
    # similarity_mode = 'mean'
    ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cloth_dir', type=str, default='./input_images/cloth', help='Input directory path of cloth images.')
    parser.add_argument('-p', '--person_dir', type=str, default='./input_images/person', help='Input directory path of person images.')
    parser.add_argument('-r', '--result_dir', type=str, default='./result_images', help='Output directory of result.')
    parser.add_argument('-x', '--cloth_mode', type=str, default='upper', help='segment part of body for similarity calculation (upper | lower)')
    parser.add_argument('-y', '--sim_method', type=str, default='reid', help='feature extraction method (reid | vae)')
    parser.add_argument('-z', '--sim_mode', type=str, default='mean', help='compare similarity mode between clothes and persons (mean | max)')
    args = parser.parse_args()
    ##
    cloth_dir = args.cloth_dir
    person_dir = args.person_dir
    result_dir = args.result_dir
    cloth_mode = args.cloth_mode
    similarity_method = args.sim_method
    similarity_mode = args.sim_mode

    sorted_person_image_files, sorted_similarity_scores = cloth_similarity(cloth_dir,
                                                                           person_dir, result_dir, cloth_mode, similarity_method, similarity_mode)
    print('sorted image files: \n', sorted_person_image_files)
    print('sorted similarity scores \n', sorted_similarity_scores)
